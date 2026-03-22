from __future__ import annotations

import torch
import torch.nn as nn

from src.models.backbones.unet import ResBlock3d


class SmallSLATCompletionCNN(nn.Module):
    def __init__(
        self,
        in_channels: int = 9,
        hidden_channels: int = 32,
        out_channels: int = 8,
        num_blocks: int = 3,
    ) -> None:
        super().__init__()
        self.input_layer = nn.Conv3d(in_channels, hidden_channels, kernel_size=3, padding=1)
        self.blocks = nn.Sequential(
            *[ResBlock3d(hidden_channels, hidden_channels, norm_type="batch") for _ in range(num_blocks)]
        )
        self.output_layer = nn.Sequential(
            nn.BatchNorm3d(hidden_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(hidden_channels, out_channels, kernel_size=3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.input_layer(x)
        h = self.blocks(h)
        return self.output_layer(h)
