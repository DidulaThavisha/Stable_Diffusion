import torch
import torch.nn as nn
import torch.nn.functional as F
from attention import SelfAttention, CrossAttention

 
class TimeEmbedding(nn.Module):
    def __init__(self, n_embed: int) -> None:
        super().__init__()
        self.linear_1 = nn.Linear(n_embed, 4 * n_embed)
        self.linear_2 = nn.Linear(n_embed * 4, n_embed * 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x = (1,320)
        x = self.linear_1(x)
        x = F.silu(x)
        x = self.linear_2(x)
        return x 


class SwitchSequential(nn.Sequential):
    def forward(self, x: torch.Tensor, context: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
        for layer in self:
            if isinstance(layer, UNET_AttentionBlock):
                x = layer(x, context)
            elif isinstance(layer, UNET_ResidualBlock):
                x = layer(x, time)
            else:
                x = layer(x)
        return x


class UpSample(nn.Module):
    def __init__(self, in_channel) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channel, in_channel, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        x = self.conv(x)        
        return x


class UNET_ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, n_time=1280) -> None:
        super().__init__()
        self.group_norm = nn.GroupNorm(32, in_channels)
        self.conv_feature = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.linear = nn.Linear(n_time, out_channels)

        self.group_norm_merged = nn.GroupNorm(32, out_channels)
        self.conv_merged = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

        if in_channels == out_channels:
            self.residual_layer = nn.Identity()
        else:
            self.residual_layer = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, feature: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
        x = feature
        feature = self.group_norm(feature)
        feature = F.silu(feature)
        feature = self.conv_feature(feature)
        time = F.silu(time)
        time = self.linear(time)
        merged = feature + time.unsqueeze(-1).unsqueeze(-1)
        merged = self.group_norm_merged(merged)
        merged  = F.silu(merged)
        merged = self.conv_merged(merged)
        return merged + self.residual_layer(x)
    

class UNET_AttentionBlock(nn.Module):
    def __init__(self, n_head: int, n_embed: int, d_context = 768) -> None:
        super().__init__()
        channels = n_embed * n_head
        self.group_norm = nn.GroupNorm(32, channels, 1e-6)
        self.conv_input = nn.Conv2d(channels, channels, kernel_size=1)

        self.layer_norm_1 = nn.LayerNorm(channels)
        self.attention_1 = SelfAttention(n_head, channels, in_proj_bias = False)
        self.layer_norm_2 = nn.LayerNorm(channels)
        self.attention_2 = CrossAttention(n_head, channels, d_context, in_proj_bias = False)
        self.layer_norm_3 = nn.LayerNorm(channels)
        self.linear_geglu_1 = nn.Linear(channels, 4 * channels * 2)
        self.linear_geglu_2 = nn.Linear(4 * channels, channels)
        self.conv_output = nn.Conv2d(channels, channels, kernel_size=1)

    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        long_residual_connection = x
        x = self.group_norm(x)
        x = self.conv_input(x)
        B,C,H,W = x.shape
        x = x.view((B, C, H*W))
        x = x.transpose(-1, -2)
        short_residual_connection = x
        x = self.layer_norm_1(x)
        x = self.attention_1(x)
        x = x + short_residual_connection
        short_residual_connection = x
        x = self.layer_norm_2(x)
        x = self.attention_2(x, context)
        x = x + short_residual_connection
        short_residual_connection = x
        x = self.layer_norm_3(x)
        x, gate = self.linear_geglu_1(x).chunk(2, dim=-1)
        x = x * F.gelu(gate)
        x = self.linear_geglu_2(x)
        x = x + short_residual_connection
        x = x.transpose(-1, -2)
        x = x.view((B, C, H, W))
        x = self.conv_output(x)
        return x + long_residual_connection        


class UNET_Output(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.group_norm = nn.GroupNorm(32, in_channels)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.group_norm(x)
        x = F.silu(x)
        x = self.conv(x)
        return x


class UNET(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        
        self.encoder = nn.Module([
            SwitchSequential(nn.Conv2d(4, 320, kernel_size=3, padding=1)),
            SwitchSequential(UNET_ResidualBlock(320, 320), UNET_AttentionBlock(8, 40)),
            SwitchSequential(UNET_ResidualBlock(320, 320), UNET_AttentionBlock(8, 40)),
            SwitchSequential(nn.Conv2d(320, 320, kernel_size=3, stride=2, padding=1)),
            SwitchSequential(UNET_ResidualBlock(320, 640), UNET_AttentionBlock(8, 80)),
            SwitchSequential(UNET_ResidualBlock(640, 640), UNET_AttentionBlock(8, 80)),
            SwitchSequential(nn.Conv2d(640, 640, kernel_size=3, stride=2, padding=1)),
            SwitchSequential(UNET_ResidualBlock(640, 1280), UNET_AttentionBlock(8, 160)),
            SwitchSequential(UNET_ResidualBlock(1280, 1280), UNET_AttentionBlock(8, 160)),
            SwitchSequential(nn.Conv2d(1280, 1280, kernel_size=3, stride=2, padding=1)),
            SwitchSequential(UNET_ResidualBlock(1280, 1280)),
            SwitchSequential(UNET_ResidualBlock(1280, 1280))
        ])
        self.bottleneck = SwitchSequential(
            UNET_ResidualBlock(1280, 1280),
            UNET_AttentionBlock(8, 160),
            UNET_ResidualBlock(1280, 1280)
        )
        self.decoder = nn.Module([
            SwitchSequential(UNET_ResidualBlock(2560, 1280)),
            SwitchSequential(UNET_ResidualBlock(2560, 1280)),
            SwitchSequential(UNET_ResidualBlock(2560, 1280), UpSample(1280)),
            SwitchSequential(UNET_ResidualBlock(2560, 1280), UNET_AttentionBlock(8, 160)),
            SwitchSequential(UNET_ResidualBlock(2560, 1280), UNET_AttentionBlock(8, 160)),
            SwitchSequential(UNET_ResidualBlock(2560, 1280), UNET_AttentionBlock(8, 160)),
            SwitchSequential(UNET_ResidualBlock(1920, 1280), UNET_AttentionBlock(8, 160), UpSample(1280)),
            SwitchSequential(UNET_ResidualBlock(1920, 640), UNET_AttentionBlock(8, 80)),
            SwitchSequential(UNET_ResidualBlock(1280, 640), UNET_AttentionBlock(8, 80)),
            SwitchSequential(UNET_ResidualBlock(960, 640), UNET_AttentionBlock(8, 40), UpSample(640)),
            SwitchSequential(UNET_ResidualBlock(960, 320), UNET_AttentionBlock(8, 40)),
            SwitchSequential(UNET_ResidualBlock(640, 320), UNET_AttentionBlock(8, 80)),
            SwitchSequential(UNET_ResidualBlock(640, 320), UNET_AttentionBlock(8, 40)),
        ])  

    def forward(self, x):
        return x  


class Diffusion(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.time_embedding = TimeEmbedding(320)
        self.unet = UNET()
        self.unet_final = UNET_Output()

    def forward(self, latent: torch.Tensor, context: torch.Tensor,  time: torch.Tensor) -> torch.Tensor:
        # latent = (B, 4, H/8, H/8)
        # context = (B, T, C) ; C = 768
        # time = (1, 320)
        time = self.time_embedding(time)
        return 
    

