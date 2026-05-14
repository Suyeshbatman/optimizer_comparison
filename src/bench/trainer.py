"""Generic train/eval loop for the optimizer benchmark.

Handles both classification (cross-entropy + accuracy) and regression
(MSE + R²) based on the dataset's meta["task"] field.

The trainer is deliberately simple. No learning rate schedulers, no
early stopping, no gradient clipping. We want to measure the raw
behavior of each optimizer without extra tricks.
"""
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import r2_score, precision_recall_fscore_support, confusion_matrix

from src.bench.metrics import (
    Timer, MemoryProbe, optimizer_state_bytes,
    grad_norm, param_norm, steps_to_threshold,
)
from src.bench.logger import RunLogger


def train_one_config(model, train_loader, test_loader, optimizer, cfg, meta, logger):
    """Train a model and collect all benchmark metrics.

    Parameters
    ----------
    model : nn.Module
        Already moved to the correct device.
    train_loader, test_loader : DataLoader
        From src/datasets/*.py.
    optimizer : torch.optim.Optimizer
        From src/optim/registry.py.
    cfg : dict
        Experiment config (needs "epochs", "device", and optionally "threshold").
    meta : dict
        Dataset metadata (needs "task"; "classification" or "regression").
    logger : RunLogger
        Writes curves.csv and summary.json.

    Returns
    -------
    dict
        Summary of the run (same data written to summary.json).
    """
    device = cfg["device"]
    epochs = cfg["epochs"]
    task = meta["task"]

    # ── Pick loss function based on task ─────────────────────────
    # CrossEntropyLoss expects raw logits (no softmax) and integer labels.
    # MSELoss expects float predictions and float targets.
    if task == "classification":
        criterion = nn.CrossEntropyLoss()
    else:
        criterion = nn.MSELoss()

    # ── Set up instrumentation ───────────────────────────────────
    total_timer = Timer()
    epoch_timer = Timer()
    memory_probe = MemoryProbe(device)

    # Track curves for convergence analysis
    test_metric_curve = []

    total_timer.start()

    for epoch in range(1, epochs + 1):
        # ── TRAIN phase ──────────────────────────────────────────
        model.train()  # enables dropout, batch norm in training mode
        epoch_timer.start()
        memory_probe.reset_gpu_peak()

        running_loss = 0.0
        correct = 0
        total = 0
        all_preds = []    # for R² in regression
        all_targets = []

        for batch_x, batch_y in train_loader:
            # Move data to GPU/CPU to match the model
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            # For regression, targets might need reshaping
            if task == "regression":
                batch_y = batch_y.float()
                if batch_y.dim() == 1:
                    batch_y = batch_y.unsqueeze(1)  # (B,) -> (B,1)

            # -- The four core steps of training --

            # 1. Zero gradients from the previous step.
            #    Without this, gradients accumulate (add up) across steps,
            #    which is almost never what you want.
            optimizer.zero_grad()

            # 2. Forward pass: feed data through the model.
            outputs = model(batch_x)

            # 3. Compute loss: how wrong are the predictions?
            loss = criterion(outputs, batch_y)

            # 4. Backward pass + optimizer step:
            #    .backward() computes gradients.
            #    .step() uses those gradients to update weights.
            loss.backward()
            optimizer.step()

            # -- Track batch statistics --
            running_loss += loss.item() * batch_x.size(0)
            total += batch_x.size(0)

            if task == "classification":
                # outputs is (B, num_classes). argmax gives predicted class.
                preds = outputs.argmax(dim=1)
                correct += (preds == batch_y).sum().item()
            else:
                all_preds.extend(outputs.detach().cpu().numpy().flatten())
                all_targets.extend(batch_y.detach().cpu().numpy().flatten())

        # Compute epoch-level training metrics
        train_loss = running_loss / total
        if task == "classification":
            train_metric = correct / total  # accuracy
        else:
            train_metric = r2_score(all_targets, all_preds)  # R²

        current_grad_norm = grad_norm(model)
        current_param_norm = param_norm(model)
        epoch_time = epoch_timer.stop()

        # ── EVAL phase ───────────────────────────────────────────
        test_loss, test_metric, test_preds, test_targets = evaluate(
            model, test_loader, criterion, device, task
        )
        test_metric_curve.append(test_metric)

        # ── Log this epoch ───────────────────────────────────────
        logger.log_epoch({
            "epoch": epoch,
            "train_loss": round(train_loss, 6),
            "test_loss": round(test_loss, 6),
            "train_metric": round(train_metric, 6),
            "test_metric": round(test_metric, 6),
            "epoch_time_s": round(epoch_time, 3),
            "peak_gpu_mb": round(memory_probe.peak_gpu_mb(), 1),
            "grad_norm": round(current_grad_norm, 4),
            "param_norm": round(current_param_norm, 4),
        })

        print(
            f"  Epoch {epoch}/{epochs} — "
            f"train_loss: {train_loss:.4f}, "
            f"test_metric: {test_metric:.4f}, "
            f"time: {epoch_time:.1f}s"
        )

    total_time = total_timer.stop()

    # ── Convergence analysis ─────────────────────────────────────
    threshold_cfg = cfg.get("threshold", None)
    convergence_step = -1
    if threshold_cfg:
        convergence_step = steps_to_threshold(
            test_metric_curve,
            target=threshold_cfg["value"],
            mode="max" if task == "classification" else "min",
        )

    # ── Build summary ────────────────────────────────────────────
    best_metric = max(test_metric_curve) if task == "classification" else min(test_metric_curve)

    summary = {
        "optimizer": cfg["optimizer_name"],
        "lr": cfg["lr"],
        "seed": cfg["seed"],
        "epochs": epochs,
        "total_time_s": round(total_time, 2),
        "best_test_metric": round(best_metric, 6),
        "final_test_metric": round(test_metric_curve[-1], 6),
        "steps_to_threshold": convergence_step,
        "optimizer_state_bytes": optimizer_state_bytes(optimizer),
        "peak_gpu_mb": round(memory_probe.peak_gpu_mb(), 1),
        "peak_cpu_mb": round(memory_probe.cpu_rss_mb(), 0),
        "final_grad_norm": round(current_grad_norm, 4),
        "final_param_norm": round(current_param_norm, 4),
    }

    if task == "classification":
        p, r, f1, sup = precision_recall_fscore_support(
            test_targets, test_preds, average=None, zero_division=0)
        mp, mr, mf1, _ = precision_recall_fscore_support(
            test_targets, test_preds, average="macro", zero_division=0)
        wp, wr, wf1, _ = precision_recall_fscore_support(
            test_targets, test_preds, average="weighted", zero_division=0)
        cm = confusion_matrix(test_targets, test_preds)
        summary["classification_metrics"] = {
            "per_class_precision": [round(v, 4) for v in p.tolist()],
            "per_class_recall": [round(v, 4) for v in r.tolist()],
            "per_class_f1": [round(v, 4) for v in f1.tolist()],
            "per_class_support": sup.tolist(),
            "macro_precision": round(float(mp), 4),
            "macro_recall": round(float(mr), 4),
            "macro_f1": round(float(mf1), 4),
            "weighted_precision": round(float(wp), 4),
            "weighted_recall": round(float(wr), 4),
            "weighted_f1": round(float(wf1), 4),
            "confusion_matrix": cm.tolist(),
        }
    else:
        errors = test_preds - test_targets
        summary["regression_metrics"] = {
            "mean_error": round(float(np.mean(errors)), 6),
            "std_error": round(float(np.std(errors)), 6),
            "mae": round(float(np.mean(np.abs(errors))), 6),
            "mse": round(float(np.mean(errors ** 2)), 6),
        }

    logger.log_summary(summary)
    return summary


def evaluate(model, test_loader, criterion, device, task):
    """Run the model on the test set without updating weights.

    torch.no_grad() disables gradient tracking — saves memory and
    speeds up computation. We never call .backward() during eval.
    """
    model.eval()  # disables dropout, switches batch norm to eval mode

    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            if task == "regression":
                batch_y = batch_y.float()
                if batch_y.dim() == 1:
                    batch_y = batch_y.unsqueeze(1)

            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)

            running_loss += loss.item() * batch_x.size(0)
            total += batch_x.size(0)

            if task == "classification":
                preds = outputs.argmax(dim=1)
                correct += (preds == batch_y).sum().item()
                all_preds.extend(preds.cpu().numpy().flatten())
                all_targets.extend(batch_y.cpu().numpy().flatten())
            else:
                all_preds.extend(outputs.cpu().numpy().flatten())
                all_targets.extend(batch_y.cpu().numpy().flatten())

    avg_loss = running_loss / total
    if task == "classification":
        metric = correct / total
    else:
        metric = r2_score(all_targets, all_preds)

    return avg_loss, metric, np.array(all_preds), np.array(all_targets)