"""California Housing dataset — tabular regression (8 features → 1 target)."""
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def build(cfg: dict) -> tuple:
    """Load California Housing and return train/test loaders + metadata.

    Unlike MNIST/CIFAR which come from torchvision, this dataset comes
    from scikit-learn. We:
    1. Load the raw numpy arrays.
    2. Split into 80% train / 20% test.
    3. Standardize features (zero mean, unit variance) — critical for
       gradient-based optimizers because wildly different feature scales
       cause wildly different gradient magnitudes.
    4. Wrap as PyTorch TensorDatasets.

    The target is median house value in units of $100,000.
    """
    batch_size = cfg["batch_size"]

    # fetch_california_housing returns a Bunch with .data and .target
    data = fetch_california_housing()
    X, y = data.data, data.target  # X: (20640, 8), y: (20640,)

    # Split BEFORE scaling — if you scale on the full dataset and then
    # split, test data leaks into the training statistics ("data leakage").
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=cfg.get("data_split_seed", 0)
    )

    # StandardScaler: for each feature, subtract mean and divide by std.
    # Fit on train only, transform both — this is the correct way.
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)   # learn mean/std from train
    X_test = scaler.transform(X_test)         # apply same mean/std to test

    # Convert numpy arrays to PyTorch tensors.
    # .float() ensures 32-bit (PyTorch default). numpy defaults to 64-bit.
    train_dataset = TensorDataset(
        torch.from_numpy(X_train).float(),
        torch.from_numpy(y_train).float().unsqueeze(1),  # (N,) -> (N,1)
    )
    test_dataset = TensorDataset(
        torch.from_numpy(X_test).float(),
        torch.from_numpy(y_test).float().unsqueeze(1),
    )

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=0
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=0
    )

    meta = {
        "task": "regression",
        "num_features": 8,
        "num_targets": 1,
    }

    return train_loader, test_loader, meta