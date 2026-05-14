"""DeepMLP: a configurable multi-layer perceptron for regression.

Used for California Housing: 8 input features → hidden layers → 1 output.
'Deep' because it has 4 hidden layers, not just 1. This gives optimizers
something substantial to work with (vanishing gradients become possible).
"""
import torch.nn as nn


class DeepMLP(nn.Module):
    """Fully-connected network with configurable hidden layers.

    Parameters
    ----------
    in_features : int
        Number of input features (8 for California Housing).
    hidden : tuple of int
        Sizes of hidden layers. Default (128, 128, 64, 32) gives 4 layers
        that progressively narrow — a common pattern for regression.
    out_features : int
        Number of outputs (1 for single-target regression).
    dropout : float
        Dropout probability. 0.0 = no dropout. Dropout randomly zeroes
        some neurons during training to prevent overfitting.
    """

    def __init__(self, in_features=8, hidden=(128, 128, 64, 32),
                 out_features=1, dropout=0.0):
        super().__init__()

        layers = []
        prev_size = in_features

        for h in hidden:
            # Each hidden layer: Linear → ReLU → (optional Dropout)
            layers.append(nn.Linear(prev_size, h))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev_size = h

        # Final output layer — no activation! For regression, we want
        # raw predicted values (any real number), not probabilities.
        layers.append(nn.Linear(prev_size, out_features))

        # nn.Sequential chains all layers so forward() is just one call.
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)