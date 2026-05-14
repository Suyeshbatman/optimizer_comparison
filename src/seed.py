"""Reproducibility utilities."""
import os
import random
import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Pin every RNG we touch so a run is reproducible.

    PyTorch, NumPy, and Python's `random` module each have their own
    random number generator. If we don't seed all three, results vary
    run-to-run even with identical code and data.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # no-op if no GPU; safe to always call

    # Trade a tiny bit of speed for determinism on GPU.
    # cuDNN's auto-tuner picks different convolution algorithms on
    # different runs otherwise, which slightly changes results.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False