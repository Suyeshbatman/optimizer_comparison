"""Model registry — maps config names to model classes."""
from src.models.cnn_mnist import SmallCNN
from src.models.resnet8 import ResNet8
from src.models.mlp import DeepMLP

MODELS = {
    "small_cnn": SmallCNN,
    "resnet8": ResNet8,
    "deep_mlp": DeepMLP,
}


def build_model(name: str, meta: dict):
    """Instantiate a model by name, using dataset metadata for sizing.

    Parameters
    ----------
    name : str
        Key into MODELS (e.g. "small_cnn").
    meta : dict
        From the dataset builder. Contains task-specific info like
        num_classes or num_features so we can size the model correctly.
    """
    if name not in MODELS:
        raise ValueError(f"Unknown model '{name}'. Available: {list(MODELS.keys())}")

    cls = MODELS[name]

    if name == "small_cnn":
        return cls(num_classes=meta.get("num_classes", 10))
    elif name == "resnet8":
        return cls(num_classes=meta.get("num_classes", 10))
    elif name == "deep_mlp":
        return cls(
            in_features=meta.get("num_features", 8),
            out_features=meta.get("num_targets", 1),
        )
    else:
        return cls()