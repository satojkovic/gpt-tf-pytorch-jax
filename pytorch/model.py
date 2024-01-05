import torch.nn as nn


class MaskedMultiSelfAttention(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        pass


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
