"""MNIST dataset loader — 28x28 grayscale handwritten digits, 10 classes."""
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


def build(cfg: dict) -> tuple:
    """Download MNIST and return train/test loaders + metadata.

    Parameters
    ----------
    cfg : dict
        Experiment config loaded from YAML. We read:
        - cfg["batch_size"]  (e.g. 128)
        - cfg.get("data_dir", "data")  where to cache downloaded files

    Returns
    -------
    (train_loader, test_loader, meta)
        meta is a dict with task type and number of classes, so the
        trainer knows whether to use cross-entropy or MSE loss.
    """
    batch_size = cfg["batch_size"]
    data_dir = cfg.get("data_dir", "data")

    # Compose transforms into a pipeline that runs on each image:
    #   1. ToTensor()  — converts PIL Image (0-255) to FloatTensor (0.0-1.0)
    #   2. Normalize() — shifts & scales so pixel values have mean≈0, std≈1
    #      0.1307 and 0.3081 are the global mean/std of the MNIST dataset.
    #      Normalization helps optimizers converge faster because gradients
    #      are better scaled.
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])

    if cfg.get("augment", False):
        transform_train = transforms.Compose([
            transforms.RandomRotation(10),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ])
    else:
        transform_train = transform_test

    train_set = datasets.MNIST(
        root=data_dir, train=True, download=True, transform=transform_train
    )
    test_set = datasets.MNIST(
        root=data_dir, train=False, download=True, transform=transform_test
    )

    # DataLoader handles:
    #   - batching: groups samples into chunks of batch_size
    #   - shuffling: randomizes order each epoch (train only, not test)
    #   - num_workers: loads data in parallel threads (0 = main thread only,
    #     safest on Windows; >0 can cause multiprocessing issues on Windows)
    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True, num_workers=0
    )
    test_loader = DataLoader(
        test_set, batch_size=batch_size, shuffle=False, num_workers=0
    )

    meta = {
        "task": "classification",
        "num_classes": 10,
        "input_shape": (1, 28, 28),  # channels, height, width
    }

    return train_loader, test_loader, meta