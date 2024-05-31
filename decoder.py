import torch 
import torch.nn as nn
import torch.nn.functional as F
from attention import SelfAttention


class VAE_ResidualBlock(nn.Module):
    def __init__(self, in_channel, out_channel) -> None:
        super().__init__()
        self.sequential = nn.Sequential(
            nn.GroupNorm(32, in_channel),
            nn.SiLU(),
            nn.Conv2d(in_channel, out_channel, kernel_size=3, padding=1),
            nn.GroupNorm(32, out_channel),
            nn.SiLU(),
            nn.Conv2d(out_channel, out_channel, kernel_size=3, padding=1),
        )
        if in_channel == out_channel:
            self.residual_layer = nn.Identity()
        else:
            self.residual_layer = nn.Conv2d(in_channel, out_channel, kernel_size=1)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.sequential(x) + self.residual_layer(x)




class VAE_AttentionBlock(nn.Module):
    def __init__(self, in_channel) -> None:
        super().__init__()
        self.group_norm = nn.GroupNorm(32, in_channel)
        self.attention = SelfAttention(1, in_channel)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = x
        x = self.group_norm(x)
        B, C, H, W = x.shape
        x = x.view((B, C, H*W))                                                     # (B, C, H*W) 
        x = x.transpose(-1, -2)                                                     # (B, H*W, C)
        x = self.attention(x)                                                       # (B, H*W, C)
        x = x.transpose(-1, -2)                                                     # (B, C, H*W)
        x = x.view((B, C, H, W))                                                    # (B, C, H, W)
        return x + y


class Decoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.sequence = nn.Sequential(
            nn.Conv2d(4, 4, kernel_size=1, padding=0),
            nn.Conv2d(4, 512, kernel_size=3, padding=1),            
            VAE_ResidualBlock(512, 512),             
            VAE_AttentionBlock(512),             
            VAE_ResidualBlock(512, 512),             
            VAE_ResidualBlock(512, 512),             
            VAE_ResidualBlock(512, 512),             
            VAE_ResidualBlock(512, 512),             
            nn.Upsample(scale_factor=2),            
            nn.Conv2d(512, 512, kernel_size=3, padding=1),             
            VAE_ResidualBlock(512, 512),             
            VAE_ResidualBlock(512, 512),             
            VAE_ResidualBlock(512, 512),             
            nn.Upsample(scale_factor=2),             
            nn.Conv2d(512, 512, kernel_size=3, padding=1),             
            VAE_ResidualBlock(512, 256),             
            VAE_ResidualBlock(256, 256),             
            VAE_ResidualBlock(256, 256),             
            nn.Upsample(scale_factor=2),             
            nn.Conv2d(256, 256, kernel_size=3, padding=1),             
            VAE_ResidualBlock(256, 128),             
            VAE_ResidualBlock(128, 128),             
            VAE_ResidualBlock(128, 128),             
            nn.GroupNorm(32, 128),             
            nn.SiLU(),             
            nn.Conv2d(128, 3, kernel_size=3, padding=1), 
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x/0.18215
        x = self.sequence(x)
       
        return x

