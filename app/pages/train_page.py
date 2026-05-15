"""Page 1 — Train & Benchmark: configure and run optimizer sweeps."""
import os
import sys
from datetime import datetime
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import torch
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.seed import set_seed
from src.device import get_device
from src.datasets import build_dataset
from src.models import build_model
from src.optim.registry import build_optimizer, OPTIMIZERS
from src.bench.metrics import (
    Timer, MemoryProbe, optimizer_state_bytes,
    grad_norm, param_norm, steps_to_threshold,
)


# ── Color palette ────────────────────────────────────────────────
COLORS = {
    "sgd":          "#1f77b4",
    "sgd_momentum": "#ff7f0e",
    "nesterov":     "#2ca02c",
    "adagrad":      "#d62728",
    "rmsprop":      "#9467bd",
    "adam":         "#8c564b",
    "adamw":        "#e377c2",
}

PRESETS = {
    "MNIST — SmallCNN (Classification)": {
        "dataset": "mnist",
        "model": "small_cnn",
        "experiment": "mnist_cnn",
        "default_epochs": 5,
        "default_batch_size": 128,
        "task": "classification",
        "threshold_metric": "test_acc",
        "threshold_value": 0.97,
        "description": "Handwritten digit recognition (28×28 grayscale, 10 classes). Fast to train — good starting point.",
    },
    "CIFAR-10 — ResNet8 (Classification)": {
        "dataset": "cifar10",
        "model": "resnet8",
        "experiment": "cifar_resnet8",
        "default_epochs": 10,
        "default_batch_size": 128,
        "task": "classification",
        "threshold_metric": "test_acc",
        "threshold_value": 0.70,
        "description": "Color image classification (32×32 RGB, 10 classes). Harder task — takes longer to train.",
    },
    "California Housing — DeepMLP (Regression)": {
        "dataset": "california",
        "model": "deep_mlp",
        "experiment": "housing_mlp",
        "default_epochs": 20,
        "default_batch_size": 64,
        "task": "regression",
        "threshold_metric": "test_mse",
        "threshold_value": 0.35,
        "description": "Predict median house prices from 8 features. Regression task — metric is R² instead of accuracy.",
    },
}

DEFAULT_LRS = {
    "sgd":          [0.01, 0.03, 0.1],
    "sgd_momentum": [0.01, 0.03],
    "nesterov":     [0.01, 0.03],
    "adagrad":      [0.01, 0.03],
    "rmsprop":      [0.001, 0.003],
    "adam":         [0.0003, 0.001],
    "adamw":        [0.0003, 0.001],
}


def _evaluate(model, test_loader, criterion, device, task):
    """Evaluate model on test set."""
    from sklearn.metrics import r2_score

    model.eval()
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


def _compute_classification_metrics(preds, targets, num_classes=None):
    """Compute precision, recall, F1 from predictions and targets."""
    from sklearn.metrics import precision_recall_fscore_support, confusion_matrix

    precision, recall, f1, support = precision_recall_fscore_support(
        targets, preds, average=None, zero_division=0,
    )
    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
        targets, preds, average="macro", zero_division=0,
    )
    weighted_p, weighted_r, weighted_f1, _ = precision_recall_fscore_support(
        targets, preds, average="weighted", zero_division=0,
    )
    cm = confusion_matrix(targets, preds)

    return {
        "per_class_precision": [round(v, 4) for v in precision.tolist()],
        "per_class_recall": [round(v, 4) for v in recall.tolist()],
        "per_class_f1": [round(v, 4) for v in f1.tolist()],
        "per_class_support": support.tolist(),
        "macro_precision": round(macro_p, 4),
        "macro_recall": round(macro_r, 4),
        "macro_f1": round(macro_f1, 4),
        "weighted_precision": round(weighted_p, 4),
        "weighted_recall": round(weighted_r, 4),
        "weighted_f1": round(weighted_f1, 4),
        "confusion_matrix": cm.tolist(),
    }


def _compute_regression_metrics(preds, targets):
    """Compute bias-variance style metrics for regression."""
    errors = preds - targets
    return {
        "mean_error": round(float(np.mean(errors)), 6),
        "std_error": round(float(np.std(errors)), 6),
        "mae": round(float(np.mean(np.abs(errors))), 6),
        "mse": round(float(np.mean(errors ** 2)), 6),
    }


def _build_scheduler(optimizer, scheduler_cfg, epochs):
    """Build a LR scheduler from config. Returns None if scheduler_name is 'None'."""
    from torch.optim import lr_scheduler

    name = scheduler_cfg.get("name", "None")
    if name == "None":
        return None
    elif name == "StepLR":
        return lr_scheduler.StepLR(
            optimizer,
            step_size=scheduler_cfg.get("step_size", 5),
            gamma=scheduler_cfg.get("gamma", 0.5),
        )
    elif name == "CosineAnnealing":
        return lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=scheduler_cfg.get("T_max", epochs),
        )
    elif name == "ReduceOnPlateau":
        return lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",
            patience=scheduler_cfg.get("patience", 3),
            factor=scheduler_cfg.get("factor", 0.5),
        )
    return None


def train_model_with_progress(model, train_loader, test_loader, optimizer,
                              cfg, meta, progress_bar, status_text,
                              loss_chart=None, metric_chart=None,
                              scheduler_cfg=None, early_stopping_cfg=None):
    """Train a model with live Streamlit progress. Returns (summary, epoch_data) or None if stopped."""
    from sklearn.metrics import r2_score
    import torch.nn as nn

    device = cfg["device"]
    epochs = cfg["epochs"]
    task = meta["task"]
    metric_name = "Accuracy" if task == "classification" else "R²"

    criterion = nn.CrossEntropyLoss() if task == "classification" else nn.MSELoss()

    scheduler = None
    if scheduler_cfg:
        scheduler = _build_scheduler(optimizer, scheduler_cfg, epochs)

    total_timer = Timer()
    epoch_timer = Timer()
    memory_probe = MemoryProbe(device)

    test_metric_curve = []
    epoch_data = []

    total_timer.start()

    es_enabled = early_stopping_cfg and early_stopping_cfg.get("enabled", False)
    es_patience = early_stopping_cfg.get("patience", 5) if es_enabled else 0
    es_min_delta = early_stopping_cfg.get("min_delta", 0.0) if es_enabled else 0
    es_best = float('-inf')
    es_counter = 0
    es_triggered = False

    for epoch in range(1, epochs + 1):
        if st.session_state.get("stop_training", False):
            status_text.text("⏹️ Training stopped by user.")
            break

        model.train()
        epoch_timer.start()
        memory_probe.reset_gpu_peak()

        running_loss = 0.0
        correct = 0
        total = 0
        all_preds = []
        all_targets = []

        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            if task == "regression":
                batch_y = batch_y.float()
                if batch_y.dim() == 1:
                    batch_y = batch_y.unsqueeze(1)

            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * batch_x.size(0)
            total += batch_x.size(0)

            if task == "classification":
                preds = outputs.argmax(dim=1)
                correct += (preds == batch_y).sum().item()
            else:
                all_preds.extend(outputs.detach().cpu().numpy().flatten())
                all_targets.extend(batch_y.detach().cpu().numpy().flatten())

        train_loss = running_loss / total
        if task == "classification":
            train_metric = correct / total
        else:
            train_metric = r2_score(all_targets, all_preds)

        g_norm = grad_norm(model)
        p_norm = param_norm(model)
        epoch_time = epoch_timer.stop()

        test_loss, test_metric, test_preds, test_targets = _evaluate(model, test_loader, criterion, device, task)
        test_metric_curve.append(test_metric)

        # Step the LR scheduler
        if scheduler is not None:
            from torch.optim.lr_scheduler import ReduceLROnPlateau
            if isinstance(scheduler, ReduceLROnPlateau):
                scheduler.step(test_metric)
            else:
                scheduler.step()

        epoch_data.append({
            "epoch": epoch,
            "train_loss": round(train_loss, 6),
            "test_loss": round(test_loss, 6),
            "train_metric": round(train_metric, 6),
            "test_metric": round(test_metric, 6),
            "epoch_time_s": round(epoch_time, 3),
            "peak_gpu_mb": round(memory_probe.peak_gpu_mb(), 1),
            "grad_norm": round(g_norm, 4),
            "param_norm": round(p_norm, 4),
        })

        progress_bar.progress(epoch / epochs)
        status_text.text(
            f"Epoch {epoch}/{epochs} — Loss: {train_loss:.4f}, "
            f"Test {metric_name}: {test_metric:.4f}, Time: {epoch_time:.1f}s"
        )

        # Update live charts
        if loss_chart is not None:
            chart_df = pd.DataFrame(epoch_data)
            loss_chart.line_chart(
                chart_df.set_index("epoch")[["train_loss", "test_loss"]],
                use_container_width=True,
            )
        if metric_chart is not None:
            chart_df = pd.DataFrame(epoch_data)
            metric_chart.line_chart(
                chart_df.set_index("epoch")[["train_metric", "test_metric"]],
                use_container_width=True,
            )

        if es_enabled:
            if test_metric > es_best + es_min_delta:
                es_best = test_metric
                es_counter = 0
            else:
                es_counter += 1
                if es_counter >= es_patience:
                    status_text.text(f"Early stopping: no improvement for {es_patience} epochs.")
                    es_triggered = True
                    break

    total_time = total_timer.stop()

    if not epoch_data:
        return None, []

    # Convergence
    threshold_cfg = cfg.get("threshold", None)
    convergence_step = -1
    if threshold_cfg and test_metric_curve:
        mode = "max" if task == "classification" else "min"
        convergence_step = steps_to_threshold(
            test_metric_curve, target=threshold_cfg["value"], mode=mode,
        )

    best_metric = max(test_metric_curve) if task == "classification" else min(test_metric_curve)

    sched_name = scheduler_cfg.get("name", "None") if scheduler_cfg else "None"

    summary = {
        "optimizer": cfg["optimizer_name"],
        "lr": cfg["lr"],
        "seed": cfg["seed"],
        "scheduler": sched_name,
        "early_stopped": es_triggered,
        "epochs": len(epoch_data),
        "total_time_s": round(total_time, 2),
        "best_test_metric": round(best_metric, 6),
        "final_test_metric": round(test_metric_curve[-1], 6),
        "steps_to_threshold": convergence_step,
        "optimizer_state_bytes": optimizer_state_bytes(optimizer),
        "peak_gpu_mb": round(memory_probe.peak_gpu_mb(), 1),
        "peak_cpu_mb": round(memory_probe.cpu_rss_mb(), 0),
        "final_grad_norm": round(g_norm, 4),
        "final_param_norm": round(p_norm, 4),
    }

    if task == "classification":
        summary["classification_metrics"] = _compute_classification_metrics(test_preds, test_targets)
    else:
        summary["regression_metrics"] = _compute_regression_metrics(test_preds, test_targets)

    return summary, epoch_data


def _make_comparison_plots(all_results, all_curves, task):
    """Generate comparison plots from collected run data."""
    figs = []
    metric_name = "Accuracy" if task == "classification" else "R²"

    # ── 1. Loss curves ──────────────────────────────────────────
    fig1, ax1 = plt.subplots(figsize=(10, 5))
    for run_key, curves in all_curves.items():
        df = pd.DataFrame(curves)
        opt_name = run_key.split(" ")[0]  # "adam (lr=0.001)" -> "adam"
        color = COLORS.get(opt_name, "#333333")
        ax1.plot(df["epoch"], df["train_loss"], label=run_key,
                 color=color, linewidth=2, marker="o", markersize=4)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Training Loss")
    ax1.set_title("Training Loss Comparison")
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)
    fig1.tight_layout()
    figs.append(("Training Loss", fig1))

    # ── 2. Test metric curves ────────────────────────────────────
    fig2, ax2 = plt.subplots(figsize=(10, 5))
    for run_key, curves in all_curves.items():
        df = pd.DataFrame(curves)
        opt_name = run_key.split(" ")[0]
        color = COLORS.get(opt_name, "#333333")
        ax2.plot(df["epoch"], df["test_metric"], label=run_key,
                 color=color, linewidth=2, marker="o", markersize=4)
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel(f"Test {metric_name}")
    ax2.set_title(f"Test {metric_name} Comparison")
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)
    fig2.tight_layout()
    figs.append((f"Test {metric_name}", fig2))

    # ── 3. Test loss curves ──────────────────────────────────────
    fig3, ax3 = plt.subplots(figsize=(10, 5))
    for run_key, curves in all_curves.items():
        df = pd.DataFrame(curves)
        opt_name = run_key.split(" ")[0]
        color = COLORS.get(opt_name, "#333333")
        ax3.plot(df["epoch"], df["test_loss"], label=run_key,
                 color=color, linewidth=2, marker="o", markersize=4)
    ax3.set_xlabel("Epoch")
    ax3.set_ylabel("Test Loss")
    ax3.set_title("Test Loss Comparison")
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3)
    fig3.tight_layout()
    figs.append(("Test Loss", fig3))

    # ── 4. Memory comparison ─────────────────────────────────────
    fig4, ax4 = plt.subplots(figsize=(8, max(4, len(all_results) * 0.5)))
    labels = [f"{r['optimizer']} (lr={r['lr']})" for r in all_results]
    mem_kb = [r["optimizer_state_bytes"] / 1024 for r in all_results]
    colors = [COLORS.get(r["optimizer"], "#333333") for r in all_results]
    bars = ax4.barh(labels, mem_kb, color=colors)
    for bar, val in zip(bars, mem_kb):
        ax4.text(bar.get_width() + 1, bar.get_y() + bar.get_height() / 2,
                 f"{val:.0f} KB", va="center", fontsize=9)
    ax4.set_xlabel("Optimizer State (KB)")
    ax4.set_title("Optimizer Memory Overhead")
    fig4.tight_layout()
    figs.append(("Memory Overhead", fig4))

    # ── 5. Wall-clock time ───────────────────────────────────────
    fig5, ax5 = plt.subplots(figsize=(8, max(4, len(all_results) * 0.5)))
    times = [r["total_time_s"] for r in all_results]
    bars = ax5.barh(labels, times, color=colors)
    for bar, val in zip(bars, times):
        ax5.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
                 f"{val:.1f}s", va="center", fontsize=9)
    ax5.set_xlabel("Total Time (seconds)")
    ax5.set_title("Training Time Comparison")
    fig5.tight_layout()
    figs.append(("Training Time", fig5))

    # ── 6. Convergence ───────────────────────────────────────────
    fig6, ax6 = plt.subplots(figsize=(8, max(4, len(all_results) * 0.5)))
    steps = [r["steps_to_threshold"] for r in all_results]
    max_s = max((s for s in steps if s >= 0), default=1)
    if max_s == 0:
        max_s = 1
    display = [s if s > 0 else (0.15 if s == 0 else max_s * 1.3) for s in steps]
    bar_colors = ["#cccccc" if s < 0 else COLORS.get(r["optimizer"], "#333333")
                  for s, r in zip(steps, all_results)]
    bars = ax6.barh(labels, display, color=bar_colors)
    for bar, val in zip(bars, steps):
        text = "did not converge" if val < 0 else (
            "epoch 0 (immediate)" if val == 0 else f"epoch {val}")
        ax6.text(bar.get_width() + 0.05, bar.get_y() + bar.get_height() / 2,
                 text, va="center", fontsize=9)
    ax6.set_xlabel("Epochs to Threshold")
    ax6.set_title("Convergence Speed")
    ax6.set_xlim(0, max_s * 1.6)
    fig6.tight_layout()
    figs.append(("Convergence", fig6))

    return figs


def render():
    """Main render function for the Train & Benchmark page."""
    st.title("🏋️ Train & Benchmark")
    st.markdown("Configure an experiment, select optimizers, and watch them train.")

    # ── Initialize session state ─────────────────────────────────
    if "stop_training" not in st.session_state:
        st.session_state.stop_training = False

    # ── Experiment selection ─────────────────────────────────────
    st.subheader("1. Choose Experiment")
    preset_name = st.selectbox("Experiment preset", list(PRESETS.keys()))
    preset = PRESETS[preset_name]
    st.info(f"📋 {preset['description']}")

    # ── Training config ──────────────────────────────────────────
    st.subheader("2. Training Configuration")
    col1, col2 = st.columns(2)
    with col1:
        epochs = st.number_input("Epochs", min_value=1, max_value=100,
                                 value=preset["default_epochs"])
    with col2:
        batch_size = st.number_input("Batch size", min_value=16, max_value=512,
                                     value=preset["default_batch_size"], step=16)

    seeds_str = st.text_input(
        "Random seeds (comma-separated)",
        value="42, 43, 44",
        help="Train each optimizer+lr combo with every seed. More seeds = better stability analysis on the Compare page.",
    )
    try:
        seeds = [int(s.strip()) for s in seeds_str.split(",") if s.strip()]
    except ValueError:
        st.error("Seeds must be integers separated by commas.")
        return
    if not seeds:
        st.error("Enter at least one seed.")
        return

    run_mode = st.radio(
        "Run mode",
        ["⚡ Quick test (1 learning rate each)", "🔬 Full grid (all learning rates)"],
        horizontal=True,
    )

    # ── Optimizer selection ──────────────────────────────────────
    st.subheader("3. Select Optimizers")
    available = [k for k in OPTIMIZERS.keys() if k != "novel"]
    selected_opts = st.multiselect(
        "Choose optimizers to benchmark",
        available,
        default=["sgd", "adam", "adamw"],
    )

    if not selected_opts:
        st.warning("Select at least one optimizer.")
        return

    # ── Learning rate config ─────────────────────────────────────
    st.subheader("4. Learning Rates")
    opt_lr_map = {}
    if run_mode.startswith("⚡"):
        cols = st.columns(min(len(selected_opts), 4))
        for i, opt_name in enumerate(selected_opts):
            with cols[i % 4]:
                default_lr = DEFAULT_LRS.get(opt_name, [0.001])[0]
                lr = st.number_input(
                    f"{opt_name} lr", value=default_lr,
                    format="%.4f", key=f"lr_{opt_name}",
                    min_value=0.00001, max_value=1.0,
                )
                opt_lr_map[opt_name] = [lr]
    else:
        for opt_name in selected_opts:
            defaults = DEFAULT_LRS.get(opt_name, [0.001])
            lr_str = st.text_input(
                f"{opt_name} learning rates (comma-separated)",
                value=", ".join(str(x) for x in defaults),
                key=f"lr_grid_{opt_name}",
            )
            try:
                opt_lr_map[opt_name] = [float(x.strip()) for x in lr_str.split(",")]
            except ValueError:
                st.error(f"Invalid learning rates for {opt_name}")
                return

    total_runs = sum(len(lrs) for lrs in opt_lr_map.values()) * len(seeds)
    st.markdown(f"**Total runs: {total_runs}** ({len(seeds)} seed{'s' if len(seeds) > 1 else ''} x {total_runs // len(seeds)} optimizer/lr combos)")

    # ── Learning rate scheduler ─────────────────────────────────
    st.subheader("5. Learning Rate Scheduler (optional)")
    scheduler_options = {
        "None": "Fixed learning rate throughout training",
        "StepLR": "Decay LR by gamma every step_size epochs",
        "CosineAnnealing": "Smoothly decay LR to near zero following a cosine curve",
        "ReduceOnPlateau": "Reduce LR when test metric stops improving",
    }
    scheduler_name = st.selectbox(
        "Scheduler",
        list(scheduler_options.keys()),
        help="Schedulers adjust the learning rate during training",
    )
    st.caption(scheduler_options[scheduler_name])

    scheduler_cfg = {"name": scheduler_name}
    if scheduler_name == "StepLR":
        sc1, sc2 = st.columns(2)
        with sc1:
            scheduler_cfg["step_size"] = st.number_input(
                "Step size (epochs)", min_value=1, max_value=50, value=5, key="sched_step")
        with sc2:
            scheduler_cfg["gamma"] = st.number_input(
                "Gamma (decay factor)", min_value=0.01, max_value=1.0, value=0.5,
                step=0.05, format="%.2f", key="sched_gamma")
    elif scheduler_name == "CosineAnnealing":
        scheduler_cfg["T_max"] = epochs
    elif scheduler_name == "ReduceOnPlateau":
        sc1, sc2 = st.columns(2)
        with sc1:
            scheduler_cfg["patience"] = st.number_input(
                "Patience (epochs)", min_value=1, max_value=20, value=3, key="sched_patience")
        with sc2:
            scheduler_cfg["factor"] = st.number_input(
                "Factor (decay)", min_value=0.01, max_value=1.0, value=0.5,
                step=0.05, format="%.2f", key="sched_factor")

    # ── Early stopping ──────────────────────────────────────────
    st.subheader("6. Early Stopping (optional)")
    early_stopping_enabled = st.checkbox(
        "Enable early stopping",
        value=False,
        help="Stop training when the test metric stops improving",
        key="early_stopping",
    )
    early_stopping_cfg = {"enabled": early_stopping_enabled}
    if early_stopping_enabled:
        es_col1, es_col2 = st.columns(2)
        with es_col1:
            early_stopping_cfg["patience"] = st.number_input(
                "Patience (epochs)", min_value=1, max_value=50, value=5, key="es_patience")
        with es_col2:
            early_stopping_cfg["min_delta"] = st.number_input(
                "Min improvement", min_value=0.0, max_value=1.0, value=0.001,
                step=0.001, format="%.4f", key="es_min_delta")
        st.caption("Training stops if the test metric doesn't improve by at least 'min improvement' for 'patience' consecutive epochs.")

    # ── Data augmentation ────────────────────────────────────────
    st.subheader("7. Data Augmentation (optional)")
    augment_enabled = False
    if preset["task"] == "classification" and preset["dataset"] in ("mnist", "cifar10"):
        augment_enabled = st.checkbox(
            "Enable data augmentation",
            value=False,
            help="Apply random transforms to training images for better generalization",
            key="augment_toggle",
        )
        if augment_enabled:
            if preset["dataset"] == "cifar10":
                st.caption("CIFAR-10: Random crop (32x32 with 4px padding) + Random horizontal flip")
            else:
                st.caption("MNIST: Random rotation (+-10 degrees) + Random translation (+-10%)")
    else:
        st.caption("Data augmentation is available for image classification tasks (MNIST, CIFAR-10).")

    # ── Control buttons ──────────────────────────────────────────
    st.markdown("---")
    btn_col1, btn_col2 = st.columns([3, 1])
    with btn_col1:
        start_clicked = st.button("🚀 Start Training", type="primary", use_container_width=True)
    with btn_col2:
        stop_clicked = st.button("⏹️ Stop", type="secondary", use_container_width=True)

    if stop_clicked:
        st.session_state.stop_training = True
        st.warning("Stop requested — training will halt after the current epoch.")
        return

    if not start_clicked:
        return

    # Reset stop flag when starting
    st.session_state.stop_training = False

    # ── Training loop ────────────────────────────────────────────
    device = get_device()
    st.markdown(f"**Device:** `{device}`")

    with st.spinner(f"Loading {preset['dataset']} dataset..."):
        cfg_data = {"batch_size": batch_size, "augment": augment_enabled}
        train_loader, test_loader, meta = build_dataset(preset["dataset"], cfg_data)
    st.success(f"✅ Dataset loaded: {preset['dataset']}")

    all_results = []
    all_curves = {}

    overall_progress = st.progress(0)

    run_idx = 0
    for opt_name in selected_opts:
        for lr in opt_lr_map[opt_name]:
            for seed in seeds:
                if st.session_state.get("stop_training", False):
                    st.warning("⏹️ Sweep stopped by user.")
                    break

                run_idx += 1
                run_key = f"{opt_name} (lr={lr}, seed={seed})"

                st.markdown(f"### Run {run_idx}/{total_runs}: **{run_key}**")

                status_text = st.empty()
                progress_bar = st.progress(0)

                chart_col1, chart_col2 = st.columns(2)
                with chart_col1:
                    st.caption("Loss")
                    loss_chart = st.empty()
                with chart_col2:
                    metric_name_label = "Accuracy" if meta["task"] == "classification" else "R²"
                    st.caption(metric_name_label)
                    metric_chart = st.empty()

                set_seed(seed)
                model = build_model(preset["model"], meta).to(device)
                optimizer = build_optimizer(opt_name, model.parameters(), {"lr": lr})

                run_cfg = {
                    "epochs": epochs,
                    "device": device,
                    "optimizer_name": opt_name,
                    "lr": lr,
                    "seed": seed,
                    "threshold": {
                        "metric": preset["threshold_metric"],
                        "value": preset["threshold_value"],
                    },
                }

                try:
                    summary, epoch_data = train_model_with_progress(
                        model, train_loader, test_loader, optimizer,
                        run_cfg, meta, progress_bar, status_text,
                        loss_chart=loss_chart, metric_chart=metric_chart,
                        scheduler_cfg=scheduler_cfg,
                        early_stopping_cfg=early_stopping_cfg,
                    )

                    if summary is None:
                        st.warning(f"⏹️ {run_key} — stopped early.")
                        continue

                    all_results.append(summary)
                    all_curves[run_key] = epoch_data

                    model_save_dir = os.path.join("runs", preset["experiment"], "models")
                    os.makedirs(model_save_dir, exist_ok=True)
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    metric_tag = "acc" if meta["task"] == "classification" else "r2"
                    metric_val = summary["best_test_metric"]
                    model_fname = f"{opt_name}_lr{lr}_seed{seed}_ep{epochs}_{metric_tag}{metric_val:.4f}_{timestamp}.pt"
                    model_path = os.path.join(model_save_dir, model_fname)
                    torch.save({
                        "model_state_dict": model.state_dict(),
                        "model_name": preset["model"],
                        "dataset": preset["dataset"],
                        "optimizer": opt_name,
                        "lr": lr,
                        "seed": seed,
                        "epochs": epochs,
                        "meta": meta,
                        "summary": summary,
                    }, model_path)

                    metric_name = "Accuracy" if meta["task"] == "classification" else "R²"
                    st.success(
                        f"✅ {run_key} — "
                        f"Best {metric_name}: {summary['best_test_metric']:.4f} | "
                        f"Time: {summary['total_time_s']:.1f}s | "
                        f"State: {summary['optimizer_state_bytes'] / 1024:.0f} KB"
                    )

                except Exception as e:
                    st.error(f"❌ {run_key} failed: {e}")
                    import traceback
                    st.code(traceback.format_exc())

                overall_progress.progress(run_idx / total_runs)

            if st.session_state.get("stop_training", False):
                break
        if st.session_state.get("stop_training", False):
            break

    # ── Save results to disk for Compare page ────────────────────
    if all_results:
        import csv
        import json

        experiment_dir = os.path.join("runs", preset["experiment"])
        os.makedirs(experiment_dir, exist_ok=True)
        results_path = os.path.join(experiment_dir, "results.csv")

        fieldnames = [
            "run_id", "optimizer", "lr", "seed", "best_test_metric",
            "final_test_metric", "total_time_s", "steps_to_threshold",
            "optimizer_state_bytes", "peak_gpu_mb", "final_grad_norm",
            "macro_precision", "macro_recall", "macro_f1",
            "weighted_f1", "mae", "status",
        ]

        file_exists = os.path.exists(results_path)
        with open(results_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            if not file_exists:
                writer.writeheader()

            for summary in all_results:
                cls_m = summary.get("classification_metrics", {})
                reg_m = summary.get("regression_metrics", {})
                writer.writerow({
                    "run_id": f"{summary['optimizer']}_lr{summary['lr']}_seed{summary['seed']}",
                    "optimizer": summary["optimizer"],
                    "lr": summary["lr"],
                    "seed": summary["seed"],
                    "best_test_metric": summary["best_test_metric"],
                    "final_test_metric": summary["final_test_metric"],
                    "total_time_s": summary["total_time_s"],
                    "steps_to_threshold": summary["steps_to_threshold"],
                    "optimizer_state_bytes": summary["optimizer_state_bytes"],
                    "peak_gpu_mb": summary["peak_gpu_mb"],
                    "final_grad_norm": summary["final_grad_norm"],
                    "macro_precision": cls_m.get("macro_precision", ""),
                    "macro_recall": cls_m.get("macro_recall", ""),
                    "macro_f1": cls_m.get("macro_f1", ""),
                    "weighted_f1": cls_m.get("weighted_f1", ""),
                    "mae": reg_m.get("mae", ""),
                    "status": "ok",
                })

        # Save per-run curves and detailed metrics JSON
        for run_key, curves in all_curves.items():
            import re as _re
            m = _re.match(r"(.+?) \(lr=(.+?), seed=(\d+)\)", run_key)
            if m:
                run_id = f"{m.group(1)}_lr{m.group(2)}_seed{m.group(3)}"
            else:
                run_id = run_key.replace(" ", "_")
            run_dir = os.path.join(experiment_dir, run_id)
            os.makedirs(run_dir, exist_ok=True)

            curves_df = pd.DataFrame(curves)
            curves_df.to_csv(os.path.join(run_dir, "curves.csv"), index=False)

        # Save detailed metrics (per-class, confusion matrix) as JSON per run
        for summary in all_results:
            run_id = f"{summary['optimizer']}_lr{summary['lr']}_seed{summary['seed']}"
            run_dir = os.path.join(experiment_dir, run_id)
            os.makedirs(run_dir, exist_ok=True)
            detail = {}
            if "classification_metrics" in summary:
                detail["classification_metrics"] = summary["classification_metrics"]
            if "regression_metrics" in summary:
                detail["regression_metrics"] = summary["regression_metrics"]
            if detail:
                with open(os.path.join(run_dir, "detailed_metrics.json"), "w") as f:
                    json.dump(detail, f, indent=2)

    # ── Results section ──────────────────────────────────────────
    if not all_results:
        st.error("No runs completed.")
        return

    st.markdown("---")
    st.header("📊 Results")

    # Best model recommendation
    metric_name = "Accuracy" if meta["task"] == "classification" else "R²"
    best = max(all_results, key=lambda r: r["best_test_metric"])
    best_fname = f"{best['optimizer']}_lr{best['lr']}_seed{best['seed']}_ep{epochs}_{('acc' if meta['task'] == 'classification' else 'r2')}{best['best_test_metric']:.4f}"
    st.success(
        f"**Recommended Model:** `{best_fname}`\n\n"
        f"Optimizer: **{best['optimizer']}** | LR: **{best['lr']}** | "
        f"Seed: **{best['seed']}** | Epochs: **{epochs}** | "
        f"Best {metric_name}: **{best['best_test_metric']:.4f}** | "
        f"Time: {best['total_time_s']:.1f}s"
    )

    # Summary table
    st.subheader("Summary Table")
    results_df = pd.DataFrame(all_results)
    display_cols = [
        "optimizer", "lr", "seed", "best_test_metric", "total_time_s",
        "optimizer_state_bytes", "steps_to_threshold", "peak_gpu_mb",
    ]
    display_df = results_df[display_cols].copy()
    display_df["state_KB"] = (display_df["optimizer_state_bytes"] / 1024).round(0)
    display_df = display_df.drop(columns=["optimizer_state_bytes"])
    display_df = display_df.sort_values("best_test_metric", ascending=False)
    st.dataframe(display_df, use_container_width=True)

    csv_data = results_df.to_csv(index=False)
    st.download_button(
        "📥 Download results CSV",
        csv_data,
        file_name=f"{preset['experiment']}_results.csv",
        mime="text/csv",
    )

    # ── Classification metrics table ───────────────────────────────
    if meta["task"] == "classification" and any("classification_metrics" in r for r in all_results):
        st.subheader("Precision / Recall / F1")
        cls_rows = []
        for r in all_results:
            cm = r.get("classification_metrics", {})
            if cm:
                cls_rows.append({
                    "Optimizer": r["optimizer"],
                    "LR": r["lr"],
                    "Accuracy": f"{r['best_test_metric']:.4f}",
                    "Macro Precision": f"{cm['macro_precision']:.4f}",
                    "Macro Recall": f"{cm['macro_recall']:.4f}",
                    "Macro F1": f"{cm['macro_f1']:.4f}",
                    "Weighted F1": f"{cm['weighted_f1']:.4f}",
                })
        if cls_rows:
            st.dataframe(pd.DataFrame(cls_rows), use_container_width=True, hide_index=True)

    if meta["task"] == "regression" and any("regression_metrics" in r for r in all_results):
        st.subheader("Regression Error Metrics")
        reg_rows = []
        for r in all_results:
            rm = r.get("regression_metrics", {})
            if rm:
                reg_rows.append({
                    "Optimizer": r["optimizer"],
                    "LR": r["lr"],
                    "R²": f"{r['best_test_metric']:.4f}",
                    "MAE": f"{rm['mae']:.4f}",
                    "MSE": f"{rm['mse']:.4f}",
                    "Mean Error (Bias)": f"{rm['mean_error']:.4f}",
                    "Std Error": f"{rm['std_error']:.4f}",
                })
        if reg_rows:
            st.dataframe(pd.DataFrame(reg_rows), use_container_width=True, hide_index=True)

    # ── Comparison plots ─────────────────────────────────────────
    st.subheader("Comparison Charts")

    figs = _make_comparison_plots(all_results, all_curves, meta["task"])

    # Display in 2-column layout
    for i in range(0, len(figs), 2):
        cols = st.columns(2)
        for j, col in enumerate(cols):
            if i + j < len(figs):
                title, fig = figs[i + j]
                with col:
                    st.markdown(f"**{title}**")
                    st.pyplot(fig, use_container_width=True)
                    plt.close(fig)

    st.success("🎉 Benchmark complete! Head to **🔮 Try the Model** to test predictions.")