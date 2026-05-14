"""Optimizer registry — maps string names to torch optimizer factories.

Usage:
    optimizer = build_optimizer("adam", model.parameters(), {"lr": 1e-3})
"""
import torch.optim as optim


def _sgd(params, hparams):
    """Vanilla SGD updates weights by: w = w - lr * gradient."""
    return optim.SGD(params, lr=hparams["lr"])


def _sgd_momentum(params, hparams):
    """SGD + Momentum adds a 'velocity' term so updates accumulate
    direction over steps, like a ball rolling downhill. Converges faster
    than vanilla SGD on most problems."""
    return optim.SGD(
        params,
        lr=hparams["lr"],
        momentum=hparams.get("momentum", 0.9),
    )


def _nesterov(params, hparams):
    """Nesterov Accelerated Gradient, a smarter version of momentum.
    It 'looks ahead' by computing the gradient at the anticipated next
    position rather than the current one. Slightly better convergence."""
    return optim.SGD(
        params,
        lr=hparams["lr"],
        momentum=hparams.get("momentum", 0.9),
        nesterov=True,
    )


def _adagrad(params, hparams):
    """Adagrad, adapts the learning rate per-parameter. Parameters that
    receive large gradients get smaller updates, and vice versa. Good for
    sparse data, but the learning rate can shrink too aggressively over time."""
    return optim.Adagrad(params, lr=hparams["lr"])


def _rmsprop(params, hparams):
    """RMSprop, fixes Adagrad's shrinking learning rate by using an
    exponential moving average of squared gradients instead of the full
    history. A precursor to Adam."""
    return optim.RMSprop(
        params,
        lr=hparams["lr"],
        alpha=hparams.get("alpha", 0.99),
    )


def _adam(params, hparams):
    """Adam, combines momentum (moving avg of gradients) with RMSprop
    (moving avg of squared gradients). The most popular optimizer in
    deep learning. Uses two hyperparameters beyond lr: beta1 and beta2."""
    return optim.Adam(
        params,
        lr=hparams["lr"],
        betas=(hparams.get("beta1", 0.9), hparams.get("beta2", 0.999)),
    )


def _adamw(params, hparams):
    """AdamW, Adam with 'decoupled' weight decay. Standard Adam applies
    weight decay inside the adaptive gradient step, which interacts badly
    with the per-parameter scaling. AdamW applies it separately, which
    gives better generalization. This is the optimizer behind most modern
    LLM training (GPT, BERT, etc.)."""
    return optim.AdamW(
        params,
        lr=hparams["lr"],
        betas=(hparams.get("beta1", 0.9), hparams.get("beta2", 0.999)),
        weight_decay=hparams.get("weight_decay", 1e-2),
    )


def _novel(params, hparams):
    """Placeholder for novel optimizer variant — not designed yet."""
    raise NotImplementedError(
        "Novel optimizer not implemented yet. "
        "Remove 'novel' from your YAML config or implement src/optim/novel.py."
    )


# ── The registry itself ─────────────────────────────────────────────
# Keys are the exact strings you put in the YAML config.
OPTIMIZERS = {
    "sgd":          _sgd,
    "sgd_momentum": _sgd_momentum,
    "nesterov":     _nesterov,
    "adagrad":      _adagrad,
    "rmsprop":      _rmsprop,
    "adam":         _adam,
    "adamw":        _adamw,
    "novel":        _novel,
}


def build_optimizer(name: str, params, hparams: dict):
    """Look up an optimizer by name and instantiate it.

    Parameters
    ----------
    name : str
        Key into OPTIMIZERS (e.g. "adam", "sgd_momentum").
    params : iterable
        model.parameters() — the weights the optimizer will update.
    hparams : dict
        Hyperparameters from the YAML grid (must include at least "lr").

    Returns
    -------
    torch.optim.Optimizer
    """
    if name not in OPTIMIZERS:
        raise ValueError(
            f"Unknown optimizer '{name}'. "
            f"Available: {list(OPTIMIZERS.keys())}"
        )
    return OPTIMIZERS[name](params, hparams)