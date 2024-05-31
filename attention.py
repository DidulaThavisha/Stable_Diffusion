import torch
import torch.nn as nn
import torch.nn.functional as F

class SelfAttention(nn.Module):
    def __init__(self, n_heads, n_embed) -> None:
        super().__init__()
        self.query = nn.Linear(n_embed, n_embed)
        self.key = nn.Linear(n_embed, n_embed)
        self.value = nn.Linear(n_embed, n_embed)
        self.n_heads = n_heads
        self.n_embed = n_embed
        self.head_size = n_embed // n_heads
        self.ln = nn.Linear(n_embed, n_embed)

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
    

