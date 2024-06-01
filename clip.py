import torch
import torch.nn as nn
import torch.nn.functional as F
from attention import SelfAttention


class CLIPEmbedding(nn.Module):
    def __init__(self, n_vocab: int, n_embed: int, n_tokens: int) -> None:
        super().__init__()
        self.token_embediing = nn.Embedding(n_vocab, n_embed)
        self.position_embedding = nn.Parameter(torch.zeros(n_tokens, n_embed))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (B, T)
        x = self.token_embediing(x) + self.position_embedding
        return x


class ClipLayer(nn.Module):
    def __init__(self, n_head: int, n_embed: int) -> None:
        super().__init__()
        self.layer_norm_1 = nn.LayerNorm(n_embed)
        self.attention = SelfAttention(n_head, n_embed)
        self.layer_norm_2 = nn.LayerNorm(n_embed)
        self.linear_1 = nn.Linear(n_embed, 4 * n_embed)
        self.linear_2 = nn.Linear(4 * n_embed, n_embed)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x -> (B, T, C)
        # Self_Attention
        y = x
        x = self.layer_norm_1(x)
        x = self.attention(x, causal_mask=True)
        x = x + y
        # Feed-Forward Connection
        y = x
        x = self.layer_norm_2(x)
        x = self.linear_1(x)
        x = x * (F.sigmoid(x * 1.702))
        x = self.linear_2(x)
        return x + y


class CLIP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embedding = CLIPEmbedding(49408, 768, 77)
        self.layers = nn.Sequential(ClipLayer(12, 768) for _ in range(12))
        self.layer_norm = nn.LayerNorm(768)

    def foward(self, x: torch.Tensor):
        x = x.to(torch.long)  # Because vocab indices
        x = self.embedding(x)
        x = self.layers(x)
        x = self.layer_norm(x)
        return x
