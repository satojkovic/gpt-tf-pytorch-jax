import tensorflow as tf
import argparse
import numpy as np

from picoGPT.utils import load_encoder_hparams_and_params


class MaskedMultiSelfAttention(tf.keras.layers.Layer):
    def __init__(self, h_dim, max_T, n_heads, drop_p):
        super(MaskedMultiSelfAttention, self).__init__()
        self.n_heads = n_heads

        self.c_attn = tf.keras.layers.Dense(3 * h_dim)

        self.c_proj = tf.keras.layers.Dense(h_dim)

        self.attn_drop = tf.keras.layers.Dropout(drop_p)
        self.proj_drop = tf.keras.layers.Dropout(drop_p)

        # Create lower triangle mask
        mask = tf.linalg.band_part(tf.ones((max_T, max_T)), -1, 0)
        mask = tf.reshape(mask, (1, 1, max_T, max_T))

        # Set mask as non-trainable variable
        self.mask = tf.Variable(mask, trainable=False)

    def call(self, x):
        B, T, C = x.shape
        N, D = self.n_heads, C // self.n_heads

        qkv = self.c_attn(x)
        q, k, v = tf.split(qkv, 3, axis=-1)
        q = tf.reshape(q, (B, T, N, D))
        k = tf.reshape(k, (B, T, N, D))
        v = tf.reshape(v, (B, T, N, D))

        q = tf.transpose(q, perm=[0, 2, 1, 3])
        k = tf.transpose(k, perm=[0, 2, 1, 3])
        v = tf.transpose(v, perm=[0, 2, 1, 3])

        weights = tf.matmul(q, k, transpose_b=True) / tf.math.sqrt(
            tf.cast(D, dtype=tf.float32)
        )

        # Apply mask
        weights += (1 - self.mask) * -1e9

        normalized_weights = tf.nn.softmax(weights, axis=-1)
        attention = tf.matmul(self.attn_drop(normalized_weights), v)
        attention = tf.transpose(attention, perm=[0, 2, 1, 3])
        attention = tf.reshape(attention, (B, T, C))

        out = self.c_proj(self.proj_drop(attention))
        return out


class TransformerDecoderBlock(tf.keras.Model):
    def __init__(self, h_dim, max_T, n_heads, drop_p):
        super().__init__()

        self.attn = MaskedMultiSelfAttention(h_dim, max_T, n_heads, drop_p)
        self.mlp = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(units=4 * h_dim, activation="gelu"),
                tf.keras.layers.Dense(units=h_dim),
                tf.keras.layers.Dropout(rate=drop_p),
            ]
        )
        self.ln1 = tf.keras.layers.LayerNormalization()
        self.ln2 = tf.keras.layers.LayerNormalization()

    def call(self, x):
        x = self.attn(x) + x
        x = self.ln1(x)
        x = self.mlp(x) + x
        x = self.ln2(x)
        return x


class GPT2(tf.keras.Model):
    def __init__(self, params, hparams, max_T=1024, drop_p=0.1):
        super(GPT2, self).__init__()
        self.params = params
        self.hparams = hparams
        self.drop_p = drop_p
        # Input embeddings
        self.wte = tf.convert_to_tensor(self.params["wte"])
        self.wpe = tf.convert_to_tensor(self.params["wpe"])
        # Decoder blocks
        self.blocks = []
        for i in range(self.hparams["n_head"]):
            block = TransformerDecoderBlock(
                h_dim=self.hparams["n_embd"],
                max_T=max_T,
                n_heads=self.hparams["n_head"],
                drop_p=self.drop_p,
            )
            self.blocks.append(block)
        # LayerNorm
        self.layer_norm = tf.keras.layers.LayerNormalization()

    def set_pretrained_weights(self):
        for i, block in enumerate(self.blocks):
            self._set_block_weights(i, block)
        self._set_layernorm_weights()

    def _set_block_weights(self, layer_idx, block):
        # MaskedMultiSelfAttention.c_attn, c_proj
        block.layers[0].c_attn.set_weights(
            [
                self.params["blocks"][layer_idx]["attn"]["c_attn"]["w"],
                self.params["blocks"][layer_idx]["attn"]["c_attn"]["b"],
            ]
        )
        block.layers[0].c_proj.set_weights(
            [
                self.params["blocks"][layer_idx]["attn"]["c_proj"]["w"],
                self.params["blocks"][layer_idx]["attn"]["c_proj"]["b"],
            ]
        )
        # MLP
        block.layers[1].layers[0].set_weights(
            [
                self.params["blocks"][layer_idx]["mlp"]["c_fc"]["w"],
                self.params["blocks"][layer_idx]["mlp"]["c_fc"]["b"],
            ]
        )
        block.layers[1].layers[1].set_weights(
            [
                self.params["blocks"][layer_idx]["mlp"]["c_proj"]["w"],
                self.params["blocks"][layer_idx]["mlp"]["c_proj"]["b"],
            ]
        )
        # Layernorm(ln1, ln2)
        block.layers[2].set_weights(
            [
                self.params["blocks"][layer_idx]["ln_1"]["g"],
                self.params["blocks"][layer_idx]["ln_1"]["b"],
            ]
        )
        block.layers[3].set_weights(
            [
                self.params["blocks"][layer_idx]["ln_2"]["g"],
                self.params["blocks"][layer_idx]["ln_2"]["b"],
            ]
        )

    def _set_layernorm_weights(self):
        self.layer_norm.set_weights(
            [self.params["ln_f"]["g"], self.params["ln_f"]["b"]]
        )

    def call(self, input_ids):
        input_ids = tf.cast(input_ids, tf.int32)
        x = tf.gather(self.wte, input_ids) + tf.gather(
            self.wpe, range(input_ids.shape[1])
        )
        for block in self.blocks:
            x = block(x)
        x = self.layer_norm(x)
        return tf.matmul(x, self.params["wte"].T)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", required=True, help="Input text")
    parser.add_argument(
        "--n_tokens_to_generate", default=40, help="number of tokens to generate"
    )
    args = parser.parse_args()

    model_size = "124M"
    models_dir = "models"
    encoder, hparams, params = load_encoder_hparams_and_params(model_size, models_dir)
    print("hparams:", hparams)
    print("params:", params.keys())
    print("params[blocks]:", params["blocks"][0].keys())
    print("params[blocks][attn]:", params["blocks"][0]["attn"].keys())
    print("params[blocks][ln_1]:", params["blocks"][0]["ln_1"].keys())
    print("params[blocks][ln_2]:", params["blocks"][0]["ln_2"].keys())
    print("params[blocks][mlp]:", params["blocks"][0]["mlp"].keys())

    print("prompt:", args.prompt)
    input_ids = encoder.encode(args.prompt)
    print("input_ids:", input_ids)

    model = GPT2(params, hparams, max_T=len(input_ids))
    model.build(input_shape=(1, len(input_ids)))
    model.set_pretrained_weights()
    model.summary()

    # Predict next words
    logits = model(tf.expand_dims(input_ids, axis=0))
    next_id = tf.argmax(tf.squeeze(logits, axis=0)[-1], axis=-1)
    output_text = encoder.decode(input_ids + [next_id.numpy().item()])
    print(output_text)
