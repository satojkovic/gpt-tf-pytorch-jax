import jax.numpy as jnp
from flax import linen as nn


class MaskedMultiSelfAttention(nn.Module):
    h_dim: int
    n_heads: int
    drop_p: float

    def setup(self):
        pass

    def __call__(self, x, deterministic=None):
        pass


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


class GPT2(nn.Module):
    params: dict
    hparams: dict
    drop_p: float

    def setup(self):
        pass

    def __call__(self, input_ids):
        pass
