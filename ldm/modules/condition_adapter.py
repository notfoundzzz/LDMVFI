import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
        )

    def forward(self, x):
        return x + self.block(x)


class SRConditionAdapter(nn.Module):
    def __init__(
        self,
        in_channels: int = 9,
        cond_channels: int = 6,
        hidden_channels: int = 32,
        num_res_blocks: int = 2,
        scale: float = 1.0,
    ):
        super().__init__()
        self.scale = scale
        layers = [
            nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1),
            nn.SiLU(),
        ]
        for _ in range(num_res_blocks):
            layers.append(ResidualBlock(hidden_channels))
            layers.append(nn.SiLU())
        self.body = nn.Sequential(*layers)
        self.proj = nn.Conv2d(hidden_channels, cond_channels, kernel_size=3, padding=1)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, prev_sr, next_sr, target_hw):
        diff = torch.abs(prev_sr - next_sr)
        x = torch.cat([prev_sr, next_sr, diff], dim=1)
        if x.shape[-2:] != target_hw:
            x = F.interpolate(x, size=target_hw, mode="bilinear", align_corners=False)
        return self.proj(self.body(x)) * self.scale
