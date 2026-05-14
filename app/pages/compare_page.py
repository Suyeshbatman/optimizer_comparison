"""Page 3 — Compare Results: view and compare across all experiments."""
import os
import sys
import glob
import json
import io
import base64
from datetime import datetime
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

COLORS = {
    "sgd":          "#1f77b4",
    "sgd_momentum": "#ff7f0e",
    "nesterov":     "#2ca02c",
    "adagrad":      "#d62728",
    "rmsprop":      "#9467bd",
    "adam":         "#8c564b",
    "adamw":        "#e377c2",
}

EXPERIMENT_LABELS = {
    "mnist_cnn":      "MNIST — SmallCNN",
    "cifar_resnet8":  "CIFAR-10 — ResNet8",
    "housing_mlp":    "Housing — DeepMLP",
}


def _discover_experiments(runs_dir="runs"):
    """Find all experiments that have a results.csv."""
    experiments = {}
    if not os.path.isdir(runs_dir):
        return experiments
    for name in sorted(os.listdir(runs_dir)):
        results_path = os.path.join(runs_dir, name, "results.csv")
        if os.path.exists(results_path):
            label = EXPERIMENT_LABELS.get(name, name)
            experiments[label] = {
                "name": name,
                "results_path": results_path,
                "dir": os.path.join(runs_dir, name),
            }
    return experiments


def _load_run_curves(experiment_dir, run_id):
    """Load curves.csv for a single run."""
    path = os.path.join(experiment_dir, run_id, "curves.csv")
    if os.path.exists(path):
        return pd.read_csv(path)
    return None


def _load_run_summary(experiment_dir, run_id):
    """Load summary.json for a single run."""
    path = os.path.join(experiment_dir, run_id, "summary.json")
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return None


def _load_detailed_metrics(experiment_dir, run_id):
    """Load detailed_metrics.json for a single run."""
    path = os.path.join(experiment_dir, run_id, "detailed_metrics.json")
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return None


def _section_overview(results, experiment_name):
    """Show high-level stats for one experiment."""
    total_runs = len(results)
    optimizers = results["optimizer"].nunique()
    best_row = results.loc[results["best_test_metric"].idxmax()]
    worst_row = results.loc[results["best_test_metric"].idxmin()]

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Runs", total_runs)
    with col2:
        st.metric("Optimizers Tested", optimizers)
    with col3:
        st.metric("Best Metric",
                   f"{best_row['best_test_metric']:.4f}",
                   f"{best_row['optimizer']} (lr={best_row['lr']})")
    with col4:
        st.metric("Worst Metric",
                   f"{worst_row['best_test_metric']:.4f}",
                   f"{worst_row['optimizer']} (lr={worst_row['lr']})")


def _section_leaderboard(results):
    """Ranked table of optimizer performance."""
    st.markdown("#### 🏆 Leaderboard")
    display = _build_leaderboard_df(results)
    st.dataframe(display, use_container_width=True, hide_index=True)
    return display


def _section_loss_curves(results, experiment_dir):
    """Interactive loss curve comparison."""
    st.markdown("#### 📉 Loss Curves")

    # Let user pick which optimizers to show
    all_opts = sorted(results["optimizer"].unique())
    selected = st.multiselect(
        "Select optimizers to compare",
        all_opts,
        default=all_opts,
        key="compare_loss_opts",
    )

    if not selected:
        st.info("Select at least one optimizer.")
        return

    # For each selected optimizer, pick the best lr (highest metric)
    show_all_lrs = st.checkbox("Show all learning rates (not just best)", key="show_all_lrs")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    for opt in selected:
        opt_runs = results[results["optimizer"] == opt]

        if show_all_lrs:
            runs_to_plot = opt_runs
        else:
            # Pick best lr
            best_lr_idx = opt_runs.groupby("lr")["best_test_metric"].mean().idxmax()
            runs_to_plot = opt_runs[opt_runs["lr"] == best_lr_idx]

        # Average across seeds for same (optimizer, lr)
        for lr, lr_group in runs_to_plot.groupby("lr"):
            all_train_loss = []
            all_test_metric = []

            for _, row in lr_group.iterrows():
                curves = _load_run_curves(experiment_dir, row["run_id"])
                if curves is not None:
                    all_train_loss.append(curves["train_loss"].values)
                    all_test_metric.append(curves["test_metric"].values)

            if not all_train_loss:
                continue

            # Average across seeds
            min_len = min(len(x) for x in all_train_loss)
            avg_loss = np.mean([x[:min_len] for x in all_train_loss], axis=0)
            avg_metric = np.mean([x[:min_len] for x in all_test_metric], axis=0)
            epochs = range(1, min_len + 1)

            color = COLORS.get(opt, "#333333")
            label = f"{opt} (lr={lr})"
            linestyle = "-" if not show_all_lrs else None

            ax1.plot(epochs, avg_loss, label=label, color=color,
                     linewidth=2, marker="o", markersize=3)
            ax2.plot(epochs, avg_metric, label=label, color=color,
                     linewidth=2, marker="o", markersize=3)

    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Training Loss")
    ax1.set_title("Training Loss")
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)

    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Test Metric")
    ax2.set_title("Test Metric")
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


def _section_memory(results):
    """Memory comparison charts."""
    st.markdown("#### 💾 Memory Comparison")

    col1, col2 = st.columns(2)

    with col1:
        # Optimizer state size
        state_avg = results.groupby("optimizer")["optimizer_state_bytes"].mean()
        state_avg = state_avg.sort_values()

        fig1, ax1 = plt.subplots(figsize=(6, max(3, len(state_avg) * 0.5)))
        colors = [COLORS.get(n, "#333333") for n in state_avg.index]
        bars = ax1.barh(state_avg.index, state_avg.values / 1024, color=colors)
        for bar, val in zip(bars, state_avg.values):
            ax1.text(bar.get_width() + 1, bar.get_y() + bar.get_height() / 2,
                     f"{val / 1024:.0f} KB", va="center", fontsize=9)
        ax1.set_xlabel("State Size (KB)")
        ax1.set_title("Optimizer State Overhead")
        fig1.tight_layout()
        st.pyplot(fig1)
        plt.close(fig1)

    with col2:
        # GPU memory
        gpu_avg = results.groupby("optimizer")["peak_gpu_mb"].mean()
        gpu_avg = gpu_avg.sort_values()

        fig2, ax2 = plt.subplots(figsize=(6, max(3, len(gpu_avg) * 0.5)))
        colors = [COLORS.get(n, "#333333") for n in gpu_avg.index]
        bars = ax2.barh(gpu_avg.index, gpu_avg.values, color=colors)
        for bar, val in zip(bars, gpu_avg.values):
            ax2.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
                     f"{val:.1f} MB", va="center", fontsize=9)
        ax2.set_xlabel("Peak GPU (MB)")
        ax2.set_title("Peak GPU Memory")
        fig2.tight_layout()
        st.pyplot(fig2)
        plt.close(fig2)


def _section_timing(results):
    """Wall-clock time comparison."""
    st.markdown("#### ⏱️ Training Time")

    time_avg = results.groupby("optimizer")["total_time_s"].mean()
    time_avg = time_avg.sort_values()

    fig, ax = plt.subplots(figsize=(8, max(3, len(time_avg) * 0.5)))
    colors = [COLORS.get(n, "#333333") for n in time_avg.index]
    bars = ax.barh(time_avg.index, time_avg.values, color=colors)
    for bar, val in zip(bars, time_avg.values):
        ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
                f"{val:.1f}s", va="center", fontsize=9)
    ax.set_xlabel("Avg Total Time (seconds)")
    ax.set_title("Training Time by Optimizer")
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


def _section_convergence(results):
    """Convergence speed comparison."""
    st.markdown("#### 🎯 Convergence Speed")

    # Best lr per optimizer
    best_per_opt = (
        results.groupby(["optimizer", "lr"]).agg(
            metric_mean=("best_test_metric", "mean"),
            convergence=("steps_to_threshold", "mean"),
        ).reset_index()
        .sort_values("metric_mean", ascending=False)
        .drop_duplicates("optimizer", keep="first")
    )

    fig, ax = plt.subplots(figsize=(8, max(3, len(best_per_opt) * 0.5)))

    labels = [f"{r['optimizer']} (lr={r['lr']})" for _, r in best_per_opt.iterrows()]
    steps = best_per_opt["convergence"].tolist()

    max_s = max((s for s in steps if s >= 0), default=1)
    if max_s == 0:
        max_s = 1

    display = [s if s > 0 else (0.15 if s == 0 else max_s * 1.3) for s in steps]
    bar_colors = [
        "#cccccc" if s < 0 else COLORS.get(r["optimizer"], "#333333")
        for s, (_, r) in zip(steps, best_per_opt.iterrows())
    ]

    bars = ax.barh(labels, display, color=bar_colors)
    for bar, val in zip(bars, steps):
        if val < 0:
            text = "did not converge"
        elif val == 0:
            text = "epoch 0 (immediate)"
        else:
            text = f"epoch {val:.1f}"
        ax.text(bar.get_width() + 0.05, bar.get_y() + bar.get_height() / 2,
                text, va="center", fontsize=9)
    ax.set_xlabel("Epochs to Threshold")
    ax.set_title("Convergence Speed (best LR per optimizer)")
    ax.set_xlim(0, max_s * 1.6)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


def _section_stability(results):
    """Stability analysis — variance across seeds."""
    st.markdown("#### 🔬 Stability (Seed Variance)")

    grouped = results.groupby(["optimizer", "lr"]).agg(
        metric_mean=("best_test_metric", "mean"),
        metric_std=("best_test_metric", "std"),
        grad_mean=("final_grad_norm", "mean"),
        grad_std=("final_grad_norm", "std"),
        num_seeds=("seed", "count"),
    ).reset_index()

    grouped["metric_std"] = grouped["metric_std"].fillna(0)
    grouped["grad_std"] = grouped["grad_std"].fillna(0)

    # Only show entries with multiple seeds
    multi_seed = grouped[grouped["num_seeds"] > 1].copy()

    if multi_seed.empty:
        st.info("Run experiments with multiple seeds to see stability analysis. "
                "Use the YAML configs which specify seeds: [42, 43, 44].")
        return

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Metric Variance** (lower = more consistent)")
        multi_seed_sorted = multi_seed.sort_values("metric_std")
        labels = [f"{r['optimizer']} (lr={r['lr']})"
                  for _, r in multi_seed_sorted.iterrows()]
        stds = multi_seed_sorted["metric_std"].values

        fig1, ax1 = plt.subplots(figsize=(6, max(3, len(labels) * 0.4)))
        colors = [COLORS.get(r["optimizer"], "#333333")
                  for _, r in multi_seed_sorted.iterrows()]
        bars = ax1.barh(labels, stds, color=colors)
        for bar, val in zip(bars, stds):
            ax1.text(bar.get_width() + 0.0001, bar.get_y() + bar.get_height() / 2,
                     f"{val:.4f}", va="center", fontsize=8)
        ax1.set_xlabel("Std Dev of Best Metric")
        ax1.set_title("Metric Consistency")
        fig1.tight_layout()
        st.pyplot(fig1)
        plt.close(fig1)

    with col2:
        st.markdown("**Gradient Norm Variance** (lower = more stable training)")
        grad_sorted = multi_seed.sort_values("grad_std")
        labels_g = [f"{r['optimizer']} (lr={r['lr']})"
                    for _, r in grad_sorted.iterrows()]
        grad_stds = grad_sorted["grad_std"].values

        fig2, ax2 = plt.subplots(figsize=(6, max(3, len(labels_g) * 0.4)))
        colors_g = [COLORS.get(r["optimizer"], "#333333")
                    for _, r in grad_sorted.iterrows()]
        bars = ax2.barh(labels_g, grad_stds, color=colors_g)
        for bar, val in zip(bars, grad_stds):
            ax2.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height() / 2,
                     f"{val:.4f}", va="center", fontsize=8)
        ax2.set_xlabel("Std Dev of Final Grad Norm")
        ax2.set_title("Training Stability")
        fig2.tight_layout()
        st.pyplot(fig2)
        plt.close(fig2)


def _section_cross_experiment(all_experiments):
    """Compare best results across experiments."""
    st.markdown("#### 🌐 Cross-Experiment Comparison")

    rows = []
    for label, info in all_experiments.items():
        df = pd.read_csv(info["results_path"])
        best = df.loc[df["best_test_metric"].idxmax()]
        fastest = df.loc[df["total_time_s"].idxmin()]
        leanest = df.loc[df["optimizer_state_bytes"].idxmin()]

        # Among runs with state > 0, find cheapest
        nonzero = df[df["optimizer_state_bytes"] > 0]
        if not nonzero.empty:
            cheapest_adaptive = nonzero.loc[nonzero["optimizer_state_bytes"].idxmin()]
        else:
            cheapest_adaptive = leanest

        rows.append({
            "Experiment": label,
            "Best Metric": f"{best['best_test_metric']:.4f}",
            "Best Optimizer": f"{best['optimizer']} (lr={best['lr']})",
            "Fastest": f"{fastest['optimizer']} ({fastest['total_time_s']:.1f}s)",
            "Leanest (zero state)": leanest["optimizer"],
            "Cheapest Adaptive": f"{cheapest_adaptive['optimizer']} "
                                 f"({cheapest_adaptive['optimizer_state_bytes'] / 1024:.0f} KB)",
            "Total Runs": len(df),
        })

    summary_df = pd.DataFrame(rows)
    st.dataframe(summary_df, use_container_width=True, hide_index=True)


def _section_precision_recall_f1(results, experiment_dir):
    """Per-class and macro precision/recall/F1 comparison."""
    st.markdown("#### 🎯 Precision / Recall / F1")

    has_csv_metrics = "macro_f1" in results.columns and results["macro_f1"].notna().any()

    run_ids = sorted(results["run_id"].unique())
    all_detailed = {}
    for rid in run_ids:
        dm = _load_detailed_metrics(experiment_dir, rid)
        if dm and "classification_metrics" in dm:
            all_detailed[rid] = dm["classification_metrics"]

    if not has_csv_metrics and not all_detailed:
        st.info("No precision/recall/F1 data found. Retrain models to generate these metrics.")
        return

    # Macro metrics comparison table
    if has_csv_metrics:
        subset = results[results["macro_f1"].notna() & (results["macro_f1"] != "")].copy()
        if not subset.empty:
            for col in ["macro_precision", "macro_recall", "macro_f1", "weighted_f1"]:
                if col in subset.columns:
                    subset[col] = pd.to_numeric(subset[col], errors="coerce")

            grouped = subset.groupby(["optimizer", "lr"]).agg(
                macro_precision=("macro_precision", "mean"),
                macro_recall=("macro_recall", "mean"),
                macro_f1=("macro_f1", "mean"),
                weighted_f1=("weighted_f1", "mean"),
                accuracy=("best_test_metric", "mean"),
            ).reset_index().sort_values("macro_f1", ascending=False)

            grouped_display = grouped.copy()
            for col in ["macro_precision", "macro_recall", "macro_f1", "weighted_f1", "accuracy"]:
                grouped_display[col] = grouped_display[col].apply(lambda x: f"{x:.4f}")
            grouped_display.columns = [
                "Optimizer", "LR", "Macro Precision", "Macro Recall",
                "Macro F1", "Weighted F1", "Accuracy",
            ]
            st.dataframe(grouped_display, use_container_width=True, hide_index=True)

    # F1 comparison bar chart
    if has_csv_metrics:
        subset = results[results["macro_f1"].notna() & (results["macro_f1"] != "")].copy()
        subset["macro_f1"] = pd.to_numeric(subset["macro_f1"], errors="coerce")
        f1_avg = subset.groupby("optimizer")["macro_f1"].mean().sort_values()

        if not f1_avg.empty:
            fig, ax = plt.subplots(figsize=(8, max(3, len(f1_avg) * 0.5)))
            colors = [COLORS.get(n, "#333333") for n in f1_avg.index]
            bars = ax.barh(f1_avg.index, f1_avg.values, color=colors)
            for bar, val in zip(bars, f1_avg.values):
                ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height() / 2,
                        f"{val:.4f}", va="center", fontsize=9)
            ax.set_xlabel("Macro F1 Score")
            ax.set_title("Macro F1 by Optimizer")
            ax.set_xlim(0, min(1.0, f1_avg.max() * 1.15))
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

    # Per-class breakdown and confusion matrix for a selected run
    if all_detailed:
        st.markdown("**Per-class breakdown (select a run):**")
        detail_run = st.selectbox(
            "Run", list(all_detailed.keys()), key="f1_detail_run",
        )
        dm = all_detailed[detail_run]

        num_classes = len(dm["per_class_f1"])
        class_labels = [str(i) for i in range(num_classes)]
        per_class_df = pd.DataFrame({
            "Class": class_labels,
            "Precision": dm["per_class_precision"],
            "Recall": dm["per_class_recall"],
            "F1": dm["per_class_f1"],
            "Support": dm["per_class_support"],
        })
        st.dataframe(per_class_df, use_container_width=True, hide_index=True)

        # Confusion matrix heatmap
        cm = np.array(dm["confusion_matrix"])
        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
        fig.colorbar(im, ax=ax, shrink=0.8)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title(f"Confusion Matrix — {detail_run}")
        tick_marks = range(num_classes)
        ax.set_xticks(tick_marks)
        ax.set_yticks(tick_marks)
        ax.set_xticklabels(class_labels, fontsize=8)
        ax.set_yticklabels(class_labels, fontsize=8)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)


def _section_regression_error(results, experiment_dir):
    """Regression error metrics: MAE, MSE, bias."""
    st.markdown("#### 📐 Regression Error Analysis")

    has_csv_metrics = "mae" in results.columns and results["mae"].notna().any()

    run_ids = sorted(results["run_id"].unique())
    all_detailed = {}
    for rid in run_ids:
        dm = _load_detailed_metrics(experiment_dir, rid)
        if dm and "regression_metrics" in dm:
            all_detailed[rid] = dm["regression_metrics"]

    if not has_csv_metrics and not all_detailed:
        st.info("No regression error data found. Retrain models to generate these metrics.")
        return

    if all_detailed:
        rows = []
        for rid, rm in all_detailed.items():
            parts = rid.split("_lr")
            opt = parts[0] if parts else rid
            rows.append({
                "Run": rid,
                "Optimizer": opt,
                "MAE": rm.get("mae", 0),
                "MSE": rm.get("mse", 0),
                "Mean Error (Bias)": rm.get("mean_error", 0),
                "Std Error": rm.get("std_error", 0),
            })
        err_df = pd.DataFrame(rows)
        st.dataframe(err_df, use_container_width=True, hide_index=True)

        # MAE comparison bar chart
        mae_avg = err_df.groupby("Optimizer")["MAE"].mean().sort_values()
        fig, ax = plt.subplots(figsize=(8, max(3, len(mae_avg) * 0.5)))
        colors = [COLORS.get(n, "#333333") for n in mae_avg.index]
        bars = ax.barh(mae_avg.index, mae_avg.values, color=colors)
        for bar, val in zip(bars, mae_avg.values):
            ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height() / 2,
                    f"{val:.4f}", va="center", fontsize=9)
        ax.set_xlabel("Mean Absolute Error")
        ax.set_title("MAE by Optimizer")
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)


def _section_bias_variance(results, experiment_dir):
    """Bias-variance decomposition using multi-seed runs."""
    st.markdown("#### ⚖️ Bias-Variance Analysis")

    multi_seed = results.groupby(["optimizer", "lr"]).filter(lambda g: g["seed"].nunique() > 1)
    if multi_seed.empty:
        st.info(
            "Bias-variance analysis requires multiple seeds per (optimizer, lr) combination. "
            "Run experiments with multiple seeds (e.g., seeds: [42, 43, 44]) to enable this."
        )
        return

    # Load all detailed_metrics.json for multi-seed runs
    grouped = multi_seed.groupby(["optimizer", "lr"])

    bv_rows = []
    for (opt, lr), group in grouped:
        all_run_metrics = []
        for _, row in group.iterrows():
            dm = _load_detailed_metrics(experiment_dir, row["run_id"])
            if dm:
                all_run_metrics.append(dm)

        if len(all_run_metrics) < 2:
            continue

        metrics = group["best_test_metric"].values
        mean_metric = np.mean(metrics)
        var_metric = np.var(metrics)

        # Classification: per-class F1 variance across seeds
        if "classification_metrics" in all_run_metrics[0]:
            f1_arrays = [np.array(m["classification_metrics"]["per_class_f1"]) for m in all_run_metrics]
            mean_f1 = np.mean(f1_arrays, axis=0)
            var_f1 = np.mean(np.var(f1_arrays, axis=0))
            bias_term = 1.0 - np.mean(mean_f1)

            bv_rows.append({
                "Optimizer": opt,
                "LR": lr,
                "Mean Accuracy": round(mean_metric, 4),
                "Bias (1 - mean F1)": round(bias_term, 4),
                "Variance (F1)": round(var_f1, 6),
                "Metric Variance": round(var_metric, 6),
                "Seeds": len(group),
            })

        # Regression: error-based decomposition
        elif "regression_metrics" in all_run_metrics[0]:
            mean_errors = [m["regression_metrics"]["mean_error"] for m in all_run_metrics]
            mses = [m["regression_metrics"]["mse"] for m in all_run_metrics]
            bias_sq = np.mean(mean_errors) ** 2
            variance = np.var(mses)

            bv_rows.append({
                "Optimizer": opt,
                "LR": lr,
                "Mean R²": round(mean_metric, 4),
                "Bias²": round(bias_sq, 6),
                "Variance (MSE)": round(variance, 6),
                "Metric Variance": round(var_metric, 6),
                "Seeds": len(group),
            })

    if not bv_rows:
        # Fallback: use metric variance from CSV even without detailed JSONs
        for (opt, lr), group in grouped:
            metrics = group["best_test_metric"].values
            bv_rows.append({
                "Optimizer": opt,
                "LR": lr,
                "Mean Metric": round(np.mean(metrics), 4),
                "Metric Std": round(np.std(metrics), 4),
                "Metric Variance": round(np.var(metrics), 6),
                "Seeds": len(group),
            })

    if bv_rows:
        bv_df = pd.DataFrame(bv_rows)
        st.dataframe(bv_df, use_container_width=True, hide_index=True)

        # Bias vs Variance chart
        fig, ax = plt.subplots(figsize=(8, 5))
        for _, row in bv_df.iterrows():
            opt = row["Optimizer"]
            color = COLORS.get(opt, "#333333")
            bias_col = next((c for c in ["Bias (1 - mean F1)", "Bias²", "Metric Std"] if c in row.index), None)
            var_col = next((c for c in ["Variance (F1)", "Variance (MSE)", "Metric Variance"] if c in row.index), None)
            if bias_col and var_col:
                ax.scatter(row[bias_col], row[var_col], color=color, s=100, zorder=5)
                ax.annotate(f"{opt}\n(lr={row['LR']})", (row[bias_col], row[var_col]),
                            fontsize=8, ha="left", va="bottom",
                            xytext=(5, 5), textcoords="offset points")

        ax.set_xlabel("Bias")
        ax.set_ylabel("Variance")
        ax.set_title("Bias vs Variance by Optimizer")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        st.caption(
            "**Interpretation:** Lower-left is ideal (low bias + low variance). "
            "High bias = model consistently underperforms. "
            "High variance = model is sensitive to random seed / initialization."
        )


def _section_training_history(results, experiment_dir):
    """Training history timeline showing all runs chronologically."""
    st.markdown("#### 📅 Training History")

    import re
    ts_pattern = re.compile(r"_(\d{8}_\d{6})\.pt$")

    models_dir = os.path.join(experiment_dir, "models")
    if not os.path.isdir(models_dir):
        st.info("No model files found for timeline.")
        return

    history = []
    for fname in os.listdir(models_dir):
        if not fname.endswith(".pt"):
            continue
        match = ts_pattern.search(fname)
        if match:
            raw_ts = match.group(1)
            try:
                dt = datetime.strptime(raw_ts, "%Y%m%d_%H%M%S")
            except ValueError:
                continue
        else:
            mtime = os.path.getmtime(os.path.join(models_dir, fname))
            dt = datetime.fromtimestamp(mtime)

        try:
            ckpt = torch.load(os.path.join(models_dir, fname), map_location="cpu", weights_only=False)
        except Exception:
            continue

        summary = ckpt.get("summary", {})
        meta = ckpt.get("meta", {})
        task = meta.get("task", "classification")
        metric_label = "Acc" if task == "classification" else "R²"

        history.append({
            "Timestamp": dt.strftime("%Y-%m-%d %H:%M:%S"),
            "Optimizer": summary.get("optimizer", "?"),
            "LR": summary.get("lr", 0),
            "Seed": summary.get("seed", 0),
            f"Best {metric_label}": round(summary.get("best_test_metric", 0), 4),
            "Time (s)": round(summary.get("total_time_s", 0), 1),
            "Epochs": summary.get("epochs", 0),
            "Scheduler": summary.get("scheduler", "None"),
        })

    if not history:
        st.info("No timestamped training runs found.")
        return

    history.sort(key=lambda x: x["Timestamp"], reverse=True)
    st.dataframe(pd.DataFrame(history), use_container_width=True, hide_index=True)
    st.caption(f"Total models: {len(history)}")


def _section_raw_data(results, experiment_dir):
    """Show raw data tables and download options."""
    st.markdown("#### 📋 Raw Data")

    tab1, tab2 = st.tabs(["Results Table", "Run Details"])

    with tab1:
        st.dataframe(results, use_container_width=True, hide_index=True)

        csv_data = results.to_csv(index=False)
        st.download_button(
            "📥 Download results.csv",
            csv_data,
            file_name="results.csv",
            mime="text/csv",
        )

    with tab2:
        # Let user inspect individual run summaries
        run_ids = sorted(results["run_id"].unique())
        selected_run = st.selectbox("Select a run to inspect", run_ids)

        summary = _load_run_summary(experiment_dir, selected_run)
        if summary:
            st.json(summary)

        curves = _load_run_curves(experiment_dir, selected_run)
        if curves is not None:
            st.markdown("**Per-epoch curves:**")
            st.dataframe(curves, use_container_width=True, hide_index=True)


def _fig_to_base64(fig):
    """Render a matplotlib figure to a base64-encoded PNG string."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


def _build_leaderboard_df(results):
    """Build the leaderboard dataframe (shared by UI and report)."""
    grouped = results.groupby(["optimizer", "lr"]).agg(
        metric_mean=("best_test_metric", "mean"),
        metric_std=("best_test_metric", "std"),
        time_mean=("total_time_s", "mean"),
        state_bytes=("optimizer_state_bytes", "first"),
        convergence=("steps_to_threshold", "mean"),
        gpu_mb=("peak_gpu_mb", "mean"),
        num_seeds=("seed", "count"),
    ).reset_index()
    grouped["metric_std"] = grouped["metric_std"].fillna(0)
    grouped["state_KB"] = (grouped["state_bytes"] / 1024).round(0)
    grouped = grouped.sort_values("metric_mean", ascending=False)
    grouped.insert(0, "rank", range(1, len(grouped) + 1))
    grouped["metric"] = grouped.apply(
        lambda r: f"{r['metric_mean']:.4f} ± {r['metric_std']:.4f}", axis=1
    )
    display = grouped[[
        "rank", "optimizer", "lr", "metric", "time_mean",
        "convergence", "state_KB", "gpu_mb", "num_seeds"
    ]].copy()
    display.columns = [
        "Rank", "Optimizer", "LR", "Metric (mean +/- std)", "Avg Time (s)",
        "Convergence (epochs)", "State (KB)", "GPU (MB)", "Seeds"
    ]
    return display


def _build_loss_curves_fig(results, experiment_dir):
    """Build loss curves figure for the report (best LR per optimizer)."""
    all_opts = sorted(results["optimizer"].unique())

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    for opt in all_opts:
        opt_runs = results[results["optimizer"] == opt]
        best_lr_idx = opt_runs.groupby("lr")["best_test_metric"].mean().idxmax()
        runs_to_plot = opt_runs[opt_runs["lr"] == best_lr_idx]

        for lr, lr_group in runs_to_plot.groupby("lr"):
            all_train_loss = []
            all_test_metric = []

            for _, row in lr_group.iterrows():
                curves = _load_run_curves(experiment_dir, row["run_id"])
                if curves is not None:
                    all_train_loss.append(curves["train_loss"].values)
                    all_test_metric.append(curves["test_metric"].values)

            if not all_train_loss:
                continue

            min_len = min(len(x) for x in all_train_loss)
            avg_loss = np.mean([x[:min_len] for x in all_train_loss], axis=0)
            avg_metric = np.mean([x[:min_len] for x in all_test_metric], axis=0)
            epochs = range(1, min_len + 1)

            color = COLORS.get(opt, "#333333")
            label = f"{opt} (lr={lr})"
            ax1.plot(epochs, avg_loss, label=label, color=color, linewidth=2, marker="o", markersize=3)
            ax2.plot(epochs, avg_metric, label=label, color=color, linewidth=2, marker="o", markersize=3)

    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Training Loss")
    ax1.set_title("Training Loss")
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Test Metric")
    ax2.set_title("Test Metric")
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def _build_memory_figs(results):
    """Build memory comparison figures for the report."""
    state_avg = results.groupby("optimizer")["optimizer_state_bytes"].mean().sort_values()
    fig1, ax1 = plt.subplots(figsize=(6, max(3, len(state_avg) * 0.5)))
    colors = [COLORS.get(n, "#333333") for n in state_avg.index]
    bars = ax1.barh(state_avg.index, state_avg.values / 1024, color=colors)
    for bar, val in zip(bars, state_avg.values):
        ax1.text(bar.get_width() + 1, bar.get_y() + bar.get_height() / 2,
                 f"{val / 1024:.0f} KB", va="center", fontsize=9)
    ax1.set_xlabel("State Size (KB)")
    ax1.set_title("Optimizer State Overhead")
    fig1.tight_layout()

    gpu_avg = results.groupby("optimizer")["peak_gpu_mb"].mean().sort_values()
    fig2, ax2 = plt.subplots(figsize=(6, max(3, len(gpu_avg) * 0.5)))
    colors2 = [COLORS.get(n, "#333333") for n in gpu_avg.index]
    bars = ax2.barh(gpu_avg.index, gpu_avg.values, color=colors2)
    for bar, val in zip(bars, gpu_avg.values):
        ax2.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
                 f"{val:.1f} MB", va="center", fontsize=9)
    ax2.set_xlabel("Peak GPU (MB)")
    ax2.set_title("Peak GPU Memory")
    fig2.tight_layout()
    return fig1, fig2


def _build_timing_fig(results):
    """Build training time figure for the report."""
    time_avg = results.groupby("optimizer")["total_time_s"].mean().sort_values()
    fig, ax = plt.subplots(figsize=(8, max(3, len(time_avg) * 0.5)))
    colors = [COLORS.get(n, "#333333") for n in time_avg.index]
    bars = ax.barh(time_avg.index, time_avg.values, color=colors)
    for bar, val in zip(bars, time_avg.values):
        ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
                f"{val:.1f}s", va="center", fontsize=9)
    ax.set_xlabel("Avg Total Time (seconds)")
    ax.set_title("Training Time by Optimizer")
    fig.tight_layout()
    return fig


def _build_convergence_fig(results):
    """Build convergence figure for the report."""
    best_per_opt = (
        results.groupby(["optimizer", "lr"]).agg(
            metric_mean=("best_test_metric", "mean"),
            convergence=("steps_to_threshold", "mean"),
        ).reset_index()
        .sort_values("metric_mean", ascending=False)
        .drop_duplicates("optimizer", keep="first")
    )

    fig, ax = plt.subplots(figsize=(8, max(3, len(best_per_opt) * 0.5)))
    labels = [f"{r['optimizer']} (lr={r['lr']})" for _, r in best_per_opt.iterrows()]
    steps = best_per_opt["convergence"].tolist()
    max_s = max((s for s in steps if s >= 0), default=1)
    if max_s == 0:
        max_s = 1
    display = [s if s > 0 else (0.15 if s == 0 else max_s * 1.3) for s in steps]
    bar_colors = [
        "#cccccc" if s < 0 else COLORS.get(r["optimizer"], "#333333")
        for s, (_, r) in zip(steps, best_per_opt.iterrows())
    ]
    bars = ax.barh(labels, display, color=bar_colors)
    for bar, val in zip(bars, steps):
        if val < 0:
            text = "did not converge"
        elif val == 0:
            text = "epoch 0 (immediate)"
        else:
            text = f"epoch {val:.1f}"
        ax.text(bar.get_width() + 0.05, bar.get_y() + bar.get_height() / 2,
                text, va="center", fontsize=9)
    ax.set_xlabel("Epochs to Threshold")
    ax.set_title("Convergence Speed (best LR per optimizer)")
    ax.set_xlim(0, max_s * 1.6)
    fig.tight_layout()
    return fig


def _df_to_table_fig(df, title=""):
    """Render a DataFrame as a styled matplotlib table figure for PDF output."""
    n_rows = len(df)
    fig_height = max(2.5, 1.2 + n_rows * 0.35)
    fig, ax = plt.subplots(figsize=(11, min(fig_height, 8)))
    ax.axis("off")
    if title:
        ax.set_title(title, fontsize=14, fontweight="bold", pad=15, loc="left")

    col_widths = [max(0.08, min(0.2, len(str(c)) * 0.012)) for c in df.columns]
    total = sum(col_widths)
    col_widths = [w / total for w in col_widths]

    table = ax.table(
        cellText=df.values,
        colLabels=df.columns,
        cellLoc="center",
        loc="center",
        colWidths=col_widths,
    )
    table.auto_set_font_size(False)
    table.set_fontsize(7)
    table.scale(1.0, 1.4)
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_facecolor("#e2e8f0")
            cell.set_text_props(fontweight="bold", fontsize=7)
        else:
            cell.set_facecolor("#ffffff" if row % 2 == 1 else "#f8fafc")
        cell.set_edgecolor("#cbd5e1")
    fig.tight_layout(pad=1.0)
    return fig


def _generate_report_pdf(experiment_label, experiment_name, results, experiment_dir):
    """Generate a multi-page PDF report with all plots, tables, and analysis."""
    from matplotlib.backends.backend_pdf import PdfPages

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    total_runs = len(results)
    n_optimizers = results["optimizer"].nunique()
    best_row = results.loc[results["best_test_metric"].idxmax()]
    worst_row = results.loc[results["best_test_metric"].idxmin()]

    buf = io.BytesIO()
    with PdfPages(buf) as pdf:
        # ── Page 1: Title + Overview ────────────────────────────
        fig = plt.figure(figsize=(11, 8.5))
        fig.text(0.5, 0.78, "Optimizer Benchmark Report", ha="center",
                 fontsize=28, fontweight="bold", color="#1e293b")
        fig.text(0.5, 0.70, experiment_label, ha="center",
                 fontsize=18, color="#334155")
        fig.text(0.5, 0.62, f"Generated: {timestamp}", ha="center",
                 fontsize=11, color="#64748b")

        stats_text = (
            f"Total Runs: {total_runs}    |    Optimizers Tested: {n_optimizers}\n\n"
            f"Best Metric:  {best_row['best_test_metric']:.4f}  "
            f"({best_row['optimizer']}, lr={best_row['lr']})\n"
            f"Worst Metric: {worst_row['best_test_metric']:.4f}  "
            f"({worst_row['optimizer']}, lr={worst_row['lr']})"
        )
        fig.text(0.5, 0.45, stats_text, ha="center", fontsize=13,
                 color="#1a1a1a", family="monospace",
                 bbox=dict(boxstyle="round,pad=1", facecolor="#f1f5f9",
                           edgecolor="#e2e8f0"))
        fig.text(0.5, 0.08, "Generated by Optimizer Benchmark Suite",
                 ha="center", fontsize=9, color="#94a3b8")
        pdf.savefig(fig)
        plt.close(fig)

        # ── Page 2: Leaderboard ─────────────────────────────────
        leaderboard = _build_leaderboard_df(results)
        fig = _df_to_table_fig(leaderboard, "Leaderboard")
        pdf.savefig(fig)
        plt.close(fig)

        # ── Page 3: Loss curves ─────────────────────────────────
        loss_fig = _build_loss_curves_fig(results, experiment_dir)
        pdf.savefig(loss_fig)
        plt.close(loss_fig)

        # ── Page 4: Memory comparison ───────────────────────────
        mem_fig1, mem_fig2 = _build_memory_figs(results)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        buf1 = io.BytesIO()
        mem_fig1.savefig(buf1, format="png", dpi=150, bbox_inches="tight")
        plt.close(mem_fig1)
        buf1.seek(0)
        buf2 = io.BytesIO()
        mem_fig2.savefig(buf2, format="png", dpi=150, bbox_inches="tight")
        plt.close(mem_fig2)
        buf2.seek(0)
        from PIL import Image as PILImage
        ax1.imshow(PILImage.open(buf1))
        ax1.axis("off")
        ax2.imshow(PILImage.open(buf2))
        ax2.axis("off")
        fig.suptitle("Memory Comparison", fontsize=14, fontweight="bold")
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

        # ── Page 5: Training time ───────────────────────────────
        timing_fig = _build_timing_fig(results)
        pdf.savefig(timing_fig)
        plt.close(timing_fig)

        # ── Page 6: Convergence ─────────────────────────────────
        convergence_fig = _build_convergence_fig(results)
        pdf.savefig(convergence_fig)
        plt.close(convergence_fig)

        # ── Page 7: Classification metrics (if applicable) ──────
        is_classification = experiment_name in ("mnist_cnn", "cifar_resnet8")
        is_regression = experiment_name == "housing_mlp"

        if is_classification and "macro_f1" in results.columns and results["macro_f1"].notna().any():
            subset = results[results["macro_f1"].notna() & (results["macro_f1"] != "")].copy()
            for col in ["macro_precision", "macro_recall", "macro_f1", "weighted_f1"]:
                if col in subset.columns:
                    subset[col] = pd.to_numeric(subset[col], errors="coerce")
            if not subset.empty:
                f1_grouped = subset.groupby(["optimizer", "lr"]).agg(
                    macro_precision=("macro_precision", "mean"),
                    macro_recall=("macro_recall", "mean"),
                    macro_f1=("macro_f1", "mean"),
                    weighted_f1=("weighted_f1", "mean"),
                    accuracy=("best_test_metric", "mean"),
                ).reset_index().sort_values("macro_f1", ascending=False)
                for col in ["macro_precision", "macro_recall", "macro_f1", "weighted_f1", "accuracy"]:
                    f1_grouped[col] = f1_grouped[col].apply(lambda x: f"{x:.4f}")
                f1_grouped.columns = ["Optimizer", "LR", "Macro P", "Macro R",
                                      "Macro F1", "Weighted F1", "Accuracy"]
                fig = _df_to_table_fig(f1_grouped, "Precision / Recall / F1")
                pdf.savefig(fig)
                plt.close(fig)

        # ── Page: Regression metrics (if applicable) ────────────
        if is_regression and "mae" in results.columns and results["mae"].notna().any():
            subset = results[results["mae"].notna() & (results["mae"] != "")].copy()
            subset["mae"] = pd.to_numeric(subset["mae"], errors="coerce")
            if not subset.empty:
                reg_display = subset[["optimizer", "lr", "seed", "best_test_metric", "mae"]].copy()
                reg_display.columns = ["Optimizer", "LR", "Seed", "R2", "MAE"]
                fig = _df_to_table_fig(reg_display, "Regression Error Analysis")
                pdf.savefig(fig)
                plt.close(fig)

        # ── Page: Bias-Variance (if multi-seed) ────────────────
        multi_seed = results.groupby(["optimizer", "lr"]).filter(
            lambda g: g["seed"].nunique() > 1
        )
        if not multi_seed.empty:
            bv_rows = []
            for (opt, lr), group in multi_seed.groupby(["optimizer", "lr"]):
                metrics = group["best_test_metric"].values
                bv_rows.append({
                    "Optimizer": opt, "LR": lr,
                    "Mean Metric": f"{np.mean(metrics):.4f}",
                    "Std": f"{np.std(metrics):.4f}",
                    "Variance": f"{np.var(metrics):.6f}",
                    "Seeds": len(group),
                })
            if bv_rows:
                bv_df = pd.DataFrame(bv_rows)
                fig = _df_to_table_fig(bv_df, "Bias-Variance Analysis")
                pdf.savefig(fig)
                plt.close(fig)

        # ── Last page: Raw results ──────────────────────────────
        display_cols = ["optimizer", "lr", "seed", "best_test_metric",
                        "final_test_metric", "total_time_s", "steps_to_threshold"]
        raw_display = results[[c for c in display_cols if c in results.columns]].copy()
        raw_display.columns = [c.replace("_", " ").title() for c in raw_display.columns]
        for col in raw_display.select_dtypes(include="float").columns:
            raw_display[col] = raw_display[col].apply(lambda x: f"{x:.4f}")
        fig = _df_to_table_fig(raw_display, "Raw Results")
        pdf.savefig(fig)
        plt.close(fig)

    buf.seek(0)
    return buf.getvalue()


def render():
    """Main render function for the Compare Results page."""
    st.title("📊 Compare Results")
    st.markdown("Explore and compare results from completed experiments.")

    # ── Discover experiments ─────────────────────────────────────
    experiments = _discover_experiments()

    if not experiments:
        st.warning(
            "No completed experiments found in `runs/`. "
            "Go to **🏋️ Train & Benchmark** to run some experiments first, "
            "or run sweeps from the command line:\n\n"
            "```\npython -m src.run_sweep --config configs/mnist_cnn.yaml\n```"
        )
        return

    # ── Cross-experiment summary (if multiple) ───────────────────
    if len(experiments) > 1:
        st.subheader("🌐 Cross-Experiment Overview")
        _section_cross_experiment(experiments)
        st.markdown("---")

    # ── Experiment selector ──────────────────────────────────────
    st.subheader("Select Experiment to Analyze")
    selected_label = st.selectbox(
        "Choose an experiment",
        list(experiments.keys()),
    )

    info = experiments[selected_label]
    results = pd.read_csv(info["results_path"])

    col_info, col_dl = st.columns([3, 1])
    with col_info:
        st.markdown(f"**Experiment:** `{info['name']}` — **{len(results)} runs**")
    with col_dl:
        if st.button("📄 Download Report", type="primary", use_container_width=True):
            with st.spinner("Generating PDF report..."):
                report_pdf = _generate_report_pdf(
                    selected_label, info["name"], results, info["dir"]
                )
            st.session_state["_report_pdf"] = report_pdf
            st.session_state["_report_name"] = info["name"]

    if "_report_pdf" in st.session_state and st.session_state.get("_report_name") == info["name"]:
        st.download_button(
            "Save report as PDF",
            st.session_state["_report_pdf"],
            file_name=f"{info['name']}_report.pdf",
            mime="application/pdf",
        )

    st.markdown("---")

    # ── Overview metrics ─────────────────────────────────────────
    _section_overview(results, info["name"])
    st.markdown("---")

    # ── All sections ─────────────────────────────────────────────
    _section_leaderboard(results)
    st.markdown("---")

    _section_loss_curves(results, info["dir"])
    st.markdown("---")

    _section_memory(results)
    st.markdown("---")

    _section_timing(results)
    st.markdown("---")

    _section_convergence(results)
    st.markdown("---")

    _section_stability(results)
    st.markdown("---")

    # Detect task type from experiment name to show appropriate metrics
    is_classification = info["name"] in ("mnist_cnn", "cifar_resnet8")
    is_regression = info["name"] == "housing_mlp"

    if is_classification:
        _section_precision_recall_f1(results, info["dir"])
        st.markdown("---")

    if is_regression:
        _section_regression_error(results, info["dir"])
        st.markdown("---")

    _section_bias_variance(results, info["dir"])
    st.markdown("---")

    _section_training_history(results, info["dir"])
    st.markdown("---")

    _section_raw_data(results, info["dir"])