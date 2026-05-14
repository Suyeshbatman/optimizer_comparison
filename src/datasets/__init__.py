"""Dataset registry — maps config names to builder functions."""
from src.datasets.mnist import build as _mnist
from src.datasets.cifar10 import build as _cifar10
from src.datasets.california import build as _california

DATASETS = {
    "mnist": _mnist,
    "cifar10": _cifar10,
    "california": _california,
}


def build_dataset(name: str, cfg: dict):
    """Look up a dataset by name and build train/test loaders.

    Returns (train_loader, test_loader, meta).
    """
    if name not in DATASETS:
        raise ValueError(f"Unknown dataset '{name}'. Available: {list(DATASETS.keys())}")
    return DATASETS[name](cfg)