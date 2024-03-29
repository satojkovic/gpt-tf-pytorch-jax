import torch
import torch.nn as nn
import math
from torch.nn import functional as F


class MaskedMultiSelfAttention(nn.Module):
    def __init__(self, h_dim, n_heads, drop_p):
        super().__init__()
        self.n_heads = n_heads
        self.c_attn = nn.Linear(h_dim, 3 * h_dim)
        self.c_proj = nn.Linear(h_dim, h_dim)
        self.attn_drop = nn.Dropout(drop_p)
        self.proj_drop = nn.Dropout(drop_p)

    def forward(self, x):
        B, T, C = x.shape
        N, D = self.n_heads, C // self.n_heads

        # Create lower triangle mask
        mask = torch.tril(torch.ones((T, T)))
        mask = mask.reshape((1, 1, T, T))

        q, k, v = self.c_attn(x).chunk(3, dim=-1)
        q = q.view(B, T, N, D).transpose(1, 2)
        k = k.view(B, T, N, D).transpose(1, 2)
        v = v.view(B, T, N, D).transpose(1, 2)

        weights = q @ k.transpose(2, 3) / math.sqrt(D)

        # Apply mask
        weights += (1 - mask) * -1e9

        normalized_weights = F.softmax(weights, dim=-1)
        attention = self.attn_drop(normalized_weights @ v)
        attention = attention.transpose(1, 2).contiguous().view(B, T, C)

        out = self.proj_drop(self.c_proj(attention))
        return out


class TransformerDecoderBlock(nn.Module):
    def __init__(self, h_dim, n_heads, drop_p):
        super().__init__()

        self.attn = MaskedMultiSelfAttention(h_dim, n_heads, drop_p)
        self.mlp = nn.Sequential(
            nn.Linear(h_dim, 4 * h_dim),
            nn.GELU(),
            nn.Linear(4 * h_dim, h_dim),
            nn.Dropout(drop_p),
        )
        self.ln1 = nn.LayerNorm(h_dim)
        self.ln2 = nn.LayerNorm(h_dim)

    def forward(self, x):
        x = self.attn(self.ln1(x)) + x
        x = self.mlp(self.ln2(x)) + x
        return x


class GPT2(nn.Module):
    def __init__(self, params, hparams, drop_p=0.1):
        super().__init__()
        self.params = params
        self.hparams = hparams
        self.drop_p = drop_p
        self.h_dim = hparams["n_embd"]
        self.n_heads = hparams["n_head"]
        self.n_layer = hparams["n_layer"]

        self.wte = self.params["wte.weight"]
        self.wpe = self.params["wpe.weight"]

        self.blocks = nn.ModuleList(
            [
                TransformerDecoderBlock(
                    h_dim=self.h_dim, n_heads=self.n_heads, drop_p=self.drop_p
                )
                for _ in range(self.n_layer)
            ]
        )
        self.layer_norm = nn.LayerNorm(self.h_dim)

        self._set_block_weights()
        self._set_layernorm_weights()

    def _set_block_weights(self):
        def get_param(layer_idx, param_string):
            return {
                "weight": self.params[f"h.{layer_idx}.{param_string}.weight"].T,
                "bias": self.params[f"h.{layer_idx}.{param_string}.bias"],
            }

        for i, block in enumerate(self.blocks):
            block.attn.c_attn.load_state_dict(get_param(i, "attn.c_attn"))
            block.attn.c_proj.load_state_dict(get_param(i, "attn.c_proj"))
            block.ln1.load_state_dict(get_param(i, "ln_1"))
            block.ln2.load_state_dict(get_param(i, "ln_2"))
            block.mlp[0].load_state_dict(get_param(i, "mlp.c_fc"))
            block.mlp[2].load_state_dict(get_param(i, "mlp.c_proj"))

    def _set_layernorm_weights(self):
        d = {"weight": self.params["ln_f.weight"], "bias": self.params["ln_f.bias"]}
        self.layer_norm.load_state_dict(d)

    def forward(self, input_ids):
        x = self.wte[input_ids] + self.wpe[list(range(input_ids.shape[1]))]
        for block in self.blocks:
            x = block(x)
        x = self.layer_norm(x)
        out = x @ self.wte.T
        return out


if __name__ == "__main__":
    import torchsummary

    n_heads = 12
    h_dim = 768
    T = 10
    dummy_x = torch.randn((1, T, h_dim))
    model = TransformerDecoderBlock(h_dim, n_heads, drop_p=0.1)
    out = model(dummy_x)
    torchsummary.summary(model, input_size=(T, h_dim))
