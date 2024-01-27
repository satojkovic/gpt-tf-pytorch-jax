import jax.numpy as jnp
from flax import linen as nn
import math
import jax


class MaskedMultiSelfAttention(nn.Module):
    h_dim: int
    n_heads: int
    drop_p: float

    def setup(self):
        self.c_attn = nn.Dense(self.h_dim * 3)
        self.c_proj = nn.Dense(self.h_dim)

        self.attn_drop = nn.Dropout(self.drop_p)
        self.proj_drop = nn.Dropout(self.drop_p)

    def __call__(self, x, deterministic=None):
        B, T, C = x.shape
        N, D = self.n_heads, C // self.n_heads

        qkv = self.c_attn(x)
        q, k, v = jnp.array_split(qkv, 3, axis=-1)
        q = q.reshape(B, T, N, D).transpose(0, 2, 1, 3)
        k = k.reshape(B, T, N, D).transpose(0, 2, 1, 3)
        v = v.reshape(B, T, N, D).transpose(0, 2, 1, 3)

        # Returns: [B, 1, T, T] shaped causal mask
        mask = nn.make_causal_mask(jnp.ones((B, T)), dtype=jnp.int32)

        # calc attention weights (B, N, T, T)
        weights = jnp.matmul(q, jnp.swapaxes(k, -2, -1)) / math.sqrt(D)
        weights += (1 - mask) * 1e-9
        normalized_weights = nn.softmax(weights, axis=-1)

        # calc attention (B, N, T, D)
        attn = jnp.matmul(normalized_weights, v)
        attn = self.attn_drop(attn, deterministic=deterministic)

        # gather heads (B, T, N, D) -> (B, T, N*D)
        attn = attn.transpose(0, 2, 1, 3).reshape(B, T, N * D)

        out = self.proj_drop(self.c_proj(attn), deterministic=deterministic)
        return out


class MLP(nn.Module):
    h_dim: int
    drop_p: float

    @nn.compact
    def __call__(self, x, deterministic=None):
        x = nn.Dense(4 * self.h_dim)(x)
        x = nn.gelu(x)
        x = nn.Dense(self.h_dim)(x)
        x = nn.Dropout(rate=self.drop_p, deterministic=deterministic)(x)
        return x


class TransformerDecoderBlock(nn.Module):
    h_dim: int
    n_heads: int
    drop_p: float

    def setup(self):
        self.attn = MaskedMultiSelfAttention(
            h_dim=self.h_dim, n_heads=self.n_heads, drop_p=self.drop_p
        )
        self.mlp = MLP(h_dim=self.h_dim, drop_p=self.drop_p)
        self.ln1 = nn.LayerNorm(epsilon=1e-5, use_bias=True)
        self.ln2 = nn.LayerNorm(epsilon=1e-5, use_bias=True)

    def __call__(self, x, deterministic=None):
        x = self.attn(self.ln1(x), deterministic=deterministic) + x
        x = self.mlp(self.ln2(x), deterministic=deterministic) + x
        return x


class GPT2(nn.Module):
    params: dict
    hparams: dict
    drop_p: float

    def setup(self):
        self.wpe = self.params["wpe"]
        self.wte = self.params["wte"]

        self.blocks = [
            TransformerDecoderBlock(
                h_dim=self.hparams["n_embd"],
                n_heads=self.hparams["n_head"],
                drop_p=self.drop_p,
            )
            for _ in range(self.hparams["n_layer"])
        ]
        self.layer_norm = nn.LayerNorm(epsilon=1e-5)

    def assign_weights(self, model_params):
        def get_param(layer_idx, module, dense):
            return {
                "kernel": self.params["blocks"][layer_idx][module][dense]["w"],
                "bias": self.params["blocks"][layer_idx][module][dense]["b"],
            }

        def assign_decoder_block_weights(layer_idx, block):
            # attn module
            block_attn = block["attn"]
            block_attn["c_attn"] = get_param(layer_idx, module="attn", dense="c_attn")
            block_attn["c_proj"] = get_param(layer_idx, module="attn", dense="c_proj")
            # mlp module
            block_mlp = block["mlp"]
            block_mlp["Dense_0"] = get_param(layer_idx, module="mlp", dense="c_fc")
            block_mlp["Dense_1"] = get_param(layer_idx, module="mlp", dense="c_proj")
            # layer norm
            block["ln1"]["scale"], block["ln1"]["bias"] = (
                self.params["blocks"][layer_idx]["ln_1"]["g"],
                self.params["blocks"][layer_idx]["ln_1"]["b"],
            )
            block["ln2"]["scale"], block["ln2"]["bias"] = (
                self.params["blocks"][layer_idx]["ln_2"]["g"],
                self.params["blocks"][layer_idx]["ln_2"]["b"],
            )

        # DecoderBlock
        for i in range(self.hparams["n_layer"]):
            assign_decoder_block_weights(i, model_params[f"blocks_{i}"])

        # layer_norm
        model_params["layer_norm"]["scale"], model_params["layer_norm"]["bias"] = (
            self.params["ln_f"]["g"],
            self.params["ln_f"]["b"],
        )

        return model_params

    def __call__(self, input_ids, deterministic=None):
        x = self.wte[input_ids] + self.wpe[list(range(input_ids.shape[1]))]
        for block in self.blocks:
            x = block(x, deterministic=deterministic)
        x = self.layer_norm(x)
        out = jnp.matmul(x, self.wte.transpose(1, 0))
        return out
