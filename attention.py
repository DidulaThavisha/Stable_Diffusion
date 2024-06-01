import torch
import torch.nn as nn
import torch.nn.functional as F

class SelfAttention(nn.Module):
    def __init__(self, n_heads, n_embed, in_proj_bias=True, out_proj_bias=True) -> None:
        super().__init__()
        self.query = nn.Linear(n_embed, n_embed, bias=in_proj_bias)
        self.key = nn.Linear(n_embed, n_embed, bias=in_proj_bias)
        self.value = nn.Linear(n_embed, n_embed, bias=in_proj_bias)
        self.n_heads = n_heads
        self.n_embed = n_embed
        self.head_size = n_embed // n_heads
        self.ln = nn.Linear(n_embed, n_embed, bias=out_proj_bias)

    def forward(self, x: torch.Tensor, causal_mask=False) -> torch.Tensor:
        B, P, C = x.shape

        q = self.query(x)
        k = self.key(x)
        v = self.value(x)

        q = q.view((B, P, self.n_heads, self.head_size)).transpose(1, 2)
        k = k.view((B, P, self.n_heads, self.head_size)).transpose(1, 2)
        v = v.view((B, P, self.n_heads, self.head_size)).transpose(1, 2)

        weights = q @ k.transpose(-2, -1) * (self.head_size ** -0.5)
        if causal_mask:
            mask = torch.ones_like(weights, dtype=torch.bool).triu(1)
            weights = weights.masked_fill(mask, -torch.inf)

        weights = F.softmax(weights, dim=-1)
        y = weights @ v
        y = y.transpose(1, 2).reshape((B, P, C))

        y = self.ln(y)
        return y
    

class CrossAttention(nn.Module):
    def __init__(self, n_heads: int, n_embed: int, d_cross: int, in_proj_bias=True, out_proj_bias=True) -> None:
        super().__init__()
        self.query = nn.Linear(n_embed, n_embed, bias=in_proj_bias)
        self.key = nn.Linear(d_cross, n_embed, bias=in_proj_bias)
        self.value = nn.Linear(d_cross, n_embed, bias=in_proj_bias)
        self.n_heads = n_heads
        self.n_embed = n_embed
        self.head_size = n_embed // n_heads
        self.ln = nn.Linear(n_embed, n_embed, bias=out_proj_bias)

    def forward(self, x, y) -> torch.Tensor:
        # x -> (B, T, C) latent         Q is calculated from x
        # y -> (B, T, C) context        K and V are calculated from y
        B, T, C = x.shape

        q = self.query(x)
        k = self.key(y)
        v = self.value(y)

        q = q.view((B, T, self.n_heads, self.head_size)).transpose(1, 2)
        k = k.view((B, T, self.n_heads, self.head_size)).transpose(1, 2)
        v = v.view((B, T, self.n_heads, self.head_size)).transpose(1, 2)

        weights = q @ k.transpose(-2, -1) * (self.head_size ** -0.5)
        weights = F.softmax(weights, dim=-1)

        y = weights @ v

        y = y.transpose(1, 2).reshape((B, T, C))
        y = self.ln(y)
        return y