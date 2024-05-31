import torch 
import torch.nn as nn
import torch.nn.functional as F
from decoder import VAE_AttentionBlock, VAE_ResidualBlock

class Encoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 128, kernel_size=3, padding=1),                                    # (B, 128, H, H)
            VAE_ResidualBlock(128, 128),                                                    # (B, 128, H, H)
            VAE_ResidualBlock(128, 128),                                                    # (B, 128, H, H)
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=0),                        # (B, 128, H/2, H/2)
            VAE_ResidualBlock(128, 256),                                                    # (B, 256, H/2, H/2)
            VAE_ResidualBlock(256, 256),                                                    # (B, 256, H/2, H/2)
            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=0),                        # (B, 256, H/4, H/4)
            VAE_ResidualBlock(256, 512),                                                    # (B, 512, H/4, H/4)
            VAE_ResidualBlock(512, 512),                                                    # (B, 512, H/4, H/4)
            nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=0),                        # (B, 512, H/8, H/8)
            VAE_ResidualBlock(512, 512),                                                    # (B, 512, H/8, H/8)
            VAE_ResidualBlock(512, 512),                                                    # (B, 512, H/8, H/8)
            VAE_AttentionBlock(512),                                                   # (B, 512, H/8, H/8)
            VAE_ResidualBlock(512, 512),                                                    # (B, 512, H/8, H/8)
            nn.GroupNorm(32, 512),                                                          # (B, 512, H/8, H/8)
            nn.SiLU(),                                                                      
            nn.Conv2d(512, 8, kernel_size=3, padding=1),                                    # (B, 8, H/8, H/8)
            nn.Conv2d(8, 8, kernel_size=1),                                                 # (B, 8, H/8, H/8)
        )
    
    def forward(self, x: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        for layer in self.encoder:
            if getattr(layer, "stride", None) == 2:
                x = F.pad(x, (0, 1, 0, 1))            
            x = layer(x)
        
        mean , log_variance = torch.chunk(x, 2, dim=1)                                           # (B, 4, H/8, H/8) each
        log_variance = torch.clamp(log_variance, -30, 20)                                        
        variance = torch.exp(log_variance)
        std = torch.sqrt(variance)

        x = mean + std * noise                                                                   # (B, 4, H/8, H/8) each mean and std
        x *= 0.18215
        return x
    
# How does this two chunks have produced the mean and log_variance?
# How can we say log_variance and not varience or std?
