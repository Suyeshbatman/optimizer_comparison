"""ResNet8: a small residual network for CIFAR-10 (32x32x3 → 10 classes).

The key idea of ResNets: instead of learning a function H(x), each block
learns the *residual* F(x) = H(x) - x, so the output is x + F(x).
If F(x) is close to zero, the block acts like an identity. The gradient
flows straight through the skip connection. This solves the "vanishing
gradient" problem in deep networks.

This version has ~75k-100k parameters, small enough to train in seconds
on a GPU, large enough to show real optimizer differences.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    """One residual block: two conv layers with a skip connection.

    Input → Conv → BN → ReLU → Conv → BN → (+input) → ReLU → Output
                                               ↑
                                          skip connection

    If input and output have different channel counts (e.g. 16 → 32),
    the skip connection uses a 1x1 conv to match dimensions.
    """

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3,
            stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3,
            stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        # Skip connection: if dimensions change, adapt with 1x1 conv
        self.skip = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.skip = nn.Sequential(
                nn.Conv2d(
                    in_channels, out_channels, kernel_size=1,
                    stride=stride, bias=False
                ),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x):
        identity = self.skip(x)        # transform input if dimensions differ
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + identity            # THE skip connection — this is the key!
        out = F.relu(out)
        return out


class ResNet8(nn.Module):
    """ResNet with 2 stages of 2 residual blocks each = 8 conv layers total.

    Architecture:
        Conv(3→16) → BN → ReLU
        Stage 1: ResBlock(16→16) × 2    (32×32 spatial)
        Stage 2: ResBlock(16→32) × 2    (16×16 spatial, stride=2 downsamples)
        Global Average Pool → Linear(32→10)
    """

    def __init__(self, num_classes=10):
        super().__init__()
        # Initial convolution: 3 RGB channels → 16 feature maps
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)

        # Stage 1: spatial size stays 32×32, channels stay 16
        self.stage1 = nn.Sequential(
            ResidualBlock(16, 16),
            ResidualBlock(16, 16),
        )
        # Stage 2: spatial size halves to 16×16, channels double to 32
        # stride=2 in the first block causes the downsampling
        self.stage2 = nn.Sequential(
            ResidualBlock(16, 32, stride=2),
            ResidualBlock(32, 32),
        )

        # Global average pooling: takes the average of each feature map,
        # reducing (B, 32, 16, 16) → (B, 32). Much cheaper than flattening!
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(32, num_classes)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))  # (B, 16, 32, 32)
        x = self.stage1(x)                   # (B, 16, 32, 32)
        x = self.stage2(x)                   # (B, 32, 16, 16)
        x = self.avg_pool(x)                 # (B, 32, 1, 1)
        x = x.flatten(start_dim=1)           # (B, 32)
        x = self.fc(x)                       # (B, 10)
        return x