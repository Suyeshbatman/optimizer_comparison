"""SmallCNN — a compact convolutional network for MNIST (28x28 grayscale digits)."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class SmallCNN(nn.Module):
    """Two conv layers + two fully-connected layers. ~20k parameters.

    Input shape:  (batch, 1, 28, 28)   — 1 channel because MNIST is grayscale
    Output shape: (batch, 10)          — one logit per digit class (0-9)
    """

    def __init__(self, num_classes: int = 10):
        super().__init__()
        # Conv layer: 1 input channel -> 16 feature maps, 3x3 kernel, padding=1 keeps spatial size
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding=1)
        # Second conv: 16 -> 32 feature maps
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        # After two 2x2 max-pools, 28x28 -> 14x14 -> 7x7. 32 channels * 7 * 7 = 1568 features.
        self.fc1 = nn.Linear(32 * 7 * 7, 64)
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Conv -> ReLU -> 2x2 max-pool. Repeats the standard "feature extractor" pattern.
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)   # (B, 16, 14, 14)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)   # (B, 32, 7, 7)
        # Flatten the 3D feature map into a 1D vector per example.
        x = x.flatten(start_dim=1)                   # (B, 1568)
        x = F.relu(self.fc1(x))                      # (B, 64)
        x = self.fc2(x)                              # (B, 10)  raw logits
        return x