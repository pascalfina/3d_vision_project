from __future__ import annotations

import torch
import torch.nn as nn

from src.models.backbones.unet import ResBlock3d, norm_layer


class SmallSLATCompletionCNN(nn.Module):
    def __init__(
        self,
        in_channels: int = 9,
        hidden_channels: int = 32,
        out_channels: int = 8,
        num_blocks: int = 3,
    ) -> None:
        super().__init__()
        if hidden_channels % 32 != 0:
            raise ValueError(
                f"hidden_channels must be divisible by 32 for GroupNorm, got {hidden_channels}"
            )
        self.input_layer = nn.Conv3d(in_channels, hidden_channels, kernel_size=3, padding=1)
        self.blocks = nn.Sequential(
            *[
                ResBlock3d(hidden_channels, hidden_channels, norm_type="group")
                for _ in range(num_blocks)
            ]
        )
        self.output_layer = nn.Sequential(
            norm_layer("group", hidden_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(hidden_channels, out_channels, kernel_size=3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.input_layer(x)
        h = self.blocks(h)
        return self.output_layer(h)
