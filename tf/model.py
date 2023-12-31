import tensorflow as tf


class MaskedMultiSelfAttention(tf.keras.layers.Layer):
    def __init__(self, h_dim, n_heads, drop_p):
        super(MaskedMultiSelfAttention, self).__init__()
        self.n_heads = n_heads

        self.c_attn = tf.keras.layers.Dense(3 * h_dim)

        self.c_proj = tf.keras.layers.Dense(h_dim)

        self.attn_drop = tf.keras.layers.Dropout(drop_p)
        self.proj_drop = tf.keras.layers.Dropout(drop_p)

    def call(self, x):
        B, T, C = x.shape
        N, D = self.n_heads, C // self.n_heads

        # Create lower triangle mask
        mask = tf.linalg.band_part(tf.ones((T, T)), -1, 0)
        mask = tf.reshape(mask, (1, 1, T, T))

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
        weights += (1 - mask) * -1e9

        normalized_weights = tf.nn.softmax(weights, axis=-1)
        attention = self.attn_drop(tf.matmul(normalized_weights, v))
        attention = tf.transpose(attention, perm=[0, 2, 1, 3])
        attention = tf.reshape(attention, (B, T, C))

        out = self.proj_drop(self.c_proj(attention))
        return out


class TransformerDecoderBlock(tf.keras.Model):
    def __init__(self, h_dim, n_heads, drop_p):
        super().__init__()

        self.attn = MaskedMultiSelfAttention(h_dim, n_heads, drop_p)
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
        x = self.attn(self.ln1(x)) + x
        x = self.mlp(self.ln2(x)) + x
        return x


class GPT2(tf.keras.Model):
    def __init__(self, params, hparams, drop_p=0.1):
        super(GPT2, self).__init__()
        self.params = params
        self.hparams = hparams
        self.drop_p = drop_p
        # Input embeddings
        self.wte = tf.convert_to_tensor(self.params["wte"])
        self.wpe = tf.convert_to_tensor(self.params["wpe"])
        # Decoder blocks
        self.blocks = []
        for _ in range(self.hparams["n_layer"]):
            block = TransformerDecoderBlock(
                h_dim=self.hparams["n_embd"],
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
        # Text and Position Embedding
        input_ids = tf.cast(input_ids, tf.int32)
        x = tf.gather(self.wte, input_ids) + tf.gather(
            self.wpe, range(input_ids.shape[1])
        )
        # Transformer Block (Decoder only)
        for block in self.blocks:
            x = block(x)
        # Additional LayerNorm
        x = self.layer_norm(x)
        # Linear
        return tf.matmul(x, self.params["wte"].T)
