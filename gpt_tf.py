import tensorflow as tf
import argparse

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

    B, T, D = 1, hparams["n_ctx"], hparams["n_embd"]
    n_heads = hparams["n_head"]
    block = TransformerDecoderBlock(h_dim=D, max_T=T, n_heads=n_heads, drop_p=0.1)
    block.build(input_shape=(B, T, D))
    block.summary()
