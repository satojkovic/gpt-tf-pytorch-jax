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
    def __init__(self):
        super().__init__()

    def forward(self, x):
        pass


if __name__ == "__main__":
    import torchsummary

    n_heads = 12
    h_dim = 768
    T = 10
    dummy_x = torch.randn((1, T, h_dim))
    model = TransformerDecoderBlock(h_dim, n_heads, drop_p=0.1)
    out = model(dummy_x)
    torchsummary.summary(model, input_size=(T, h_dim))
