import jax.numpy as jnp
from flax import linen as nn


class GPT2(nn.Module):
    params: dict
    hparams: dict
    drop_p: float

    def setup(self):
        pass

    def __call__(self, input_ids):
        pass
