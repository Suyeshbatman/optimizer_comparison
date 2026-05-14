"""CIFAR-10 dataset loader — 32x32 color images, 10 classes."""
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


def build(cfg: dict) -> tuple:
    """Download CIFAR-10 and return train/test loaders + metadata.

    CIFAR-10 has 50k training and 10k test images across 10 classes:
    airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck.

    Images are 32x32 with 3 color channels (RGB), unlike MNIST which
    is 28x28 with 1 channel (grayscale).
    """
    batch_size = cfg["batch_size"]
    data_dir = cfg.get("data_dir", "data")

    # These mean/std values are pre-computed over the entire CIFAR-10
    # training set, per channel (R, G, B).
    # Normalizing per-channel helps because different channels can have
    # very different intensity distributions.
    if cfg.get("augment", False):
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.4914, 0.4822, 0.4465),
                std=(0.2470, 0.2435, 0.2616),
            ),
        ])
    else:
        transform_train = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.4914, 0.4822, 0.4465),
                std=(0.2470, 0.2435, 0.2616),
            ),
        ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.4914, 0.4822, 0.4465),
            std=(0.2470, 0.2435, 0.2616),
        ),
    ])

    train_set = datasets.CIFAR10(
        root=data_dir, train=True, download=True, transform=transform_train
    )
    test_set = datasets.CIFAR10(
        root=data_dir, train=False, download=True, transform=transform_test
    )

    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True, num_workers=0
    )
    test_loader = DataLoader(
        test_set, batch_size=batch_size, shuffle=False, num_workers=0
    )

    meta = {
        "task": "classification",
        "num_classes": 10,
        "input_shape": (3, 32, 32),
    }

    return train_loader, test_loader, meta