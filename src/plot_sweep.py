"""Comparative plotting for optimizer benchmark sweeps.

Usage:
    python -m src.plot_sweep --experiment test_quick
    python -m src.plot_sweep --experiment mnist_cnn

Reads runs/<experiment>/results.csv and per-run curves.csv files,
produces comparison plots saved to runs/<experiment>/plots/.
"""
import argparse
import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")  # non-interactive backend, works without a display


# ── Color palette for optimizers ─────────────────────────────────
# Consistent colors across all plots so you can visually track an
# optimizer across different charts.
COLORS = {
    "sgd":          "#1f77b4",  # blue
    "sgd_momentum": "#ff7f0e",  # orange
    "nesterov":     "#2ca02c",  # green
    "adagrad":      "#d62728",  # red
    "rmsprop":      "#9467bd",  # purple
    "adam":         "#8c564b",  # brown
    "adamw":        "#e377c2",  # pink
    "novel":        "#7f7f7f",  # gray
}


def load_curves(experiment_dir: str, run_id: str) -> pd.DataFrame:
    """Load the curves.csv for a single run."""
    path = os.path.join(experiment_dir, run_id, "curves.csv")
    if os.path.exists(path):
        return pd.read_csv(path)
    return pd.DataFrame()


def plot_loss_curves(results: pd.DataFrame, experiment_dir: str, plots_dir: str):
    """Overlay training loss curves for each optimizer (best lr per optimizer).

    For each optimizer, we pick the run (across all lrs and seeds) with the
    best final test metric, then plot its loss curve. This shows each
    optimizer at its best.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    # Group by optimizer, pick the run with best final metric
    for opt_name, group in results.groupby("optimizer"):
        best_run = group.loc[group["best_test_metric"].idxmax()]
        run_id = best_run["run_id"]
        curves = load_curves(experiment_dir, run_id)
        if curves.empty:
            continue

        color = COLORS.get(opt_name, "#333333")
        lr = best_run["lr"]
        ax.plot(
            curves["epoch"], curves["train_loss"],
            label=f"{opt_name} (lr={lr})",
            color=color, linewidth=2,
        )

    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Training Loss", fontsize=12)
    ax.set_title("Training Loss by Optimizer (Best LR)", fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(plots_dir, "loss_curves.png"), dpi=150)
    plt.close(fig)
    print("  Saved: loss_curves.png")


def plot_metric_curves(results: pd.DataFrame, experiment_dir: str, plots_dir: str):
    """Overlay test metric curves (accuracy or R²) for best lr per optimizer."""
    fig, ax = plt.subplots(figsize=(10, 6))

    for opt_name, group in results.groupby("optimizer"):
        best_run = group.loc[group["best_test_metric"].idxmax()]
        run_id = best_run["run_id"]
        curves = load_curves(experiment_dir, run_id)
        if curves.empty:
            continue

        color = COLORS.get(opt_name, "#333333")
        lr = best_run["lr"]
        ax.plot(
            curves["epoch"], curves["test_metric"],
            label=f"{opt_name} (lr={lr})",
            color=color, linewidth=2,
        )

    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Test Metric", fontsize=12)
    ax.set_title("Test Metric by Optimizer (Best LR)", fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(plots_dir, "metric_curves.png"), dpi=150)
    plt.close(fig)
    print("  Saved: metric_curves.png")


def plot_optimizer_state_memory(results: pd.DataFrame, plots_dir: str):
    """Bar chart comparing optimizer state size (bytes) across optimizers.

    This is one of the key insights of the project: adaptive optimizers
    like Adam use 2-3x more memory than SGD.
    """
    # Average across seeds and lrs per optimizer
    mem = results.groupby("optimizer")["optimizer_state_bytes"].mean()
    mem = mem.sort_values(ascending=True)

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = [COLORS.get(name, "#333333") for name in mem.index]
    bars = ax.barh(mem.index, mem.values / 1024, color=colors)

    # Add value labels on the bars
    for bar, val in zip(bars, mem.values):
        ax.text(
            bar.get_width() + 1, bar.get_y() + bar.get_height() / 2,
            f"{val / 1024:.0f} KB",
            va="center", fontsize=10,
        )

    ax.set_xlabel("Optimizer State Size (KB)", fontsize=12)
    ax.set_title("Optimizer Memory Overhead", fontsize=14)
    fig.tight_layout()
    fig.savefig(os.path.join(plots_dir, "optimizer_memory.png"), dpi=150)
    plt.close(fig)
    print("  Saved: optimizer_memory.png")


def plot_wall_clock(results: pd.DataFrame, plots_dir: str):
    """Bar chart of average total training time per optimizer."""
    time_avg = results.groupby("optimizer")["total_time_s"].mean()
    time_avg = time_avg.sort_values(ascending=True)

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = [COLORS.get(name, "#333333") for name in time_avg.index]
    bars = ax.barh(time_avg.index, time_avg.values, color=colors)

    for bar, val in zip(bars, time_avg.values):
        ax.text(
            bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
            f"{val:.1f}s",
            va="center", fontsize=10,
        )

    ax.set_xlabel("Total Training Time (seconds)", fontsize=12)
    ax.set_title("Wall-Clock Time by Optimizer", fontsize=14)
    fig.tight_layout()
    fig.savefig(os.path.join(plots_dir, "wall_clock.png"), dpi=150)
    plt.close(fig)
    print("  Saved: wall_clock.png")


def plot_convergence(results: pd.DataFrame, plots_dir: str):
    """Bar chart of steps-to-threshold per optimizer × lr combo.

    Shows average across seeds. -1 means 'did not converge'.
    0 means 'converged in the first epoch'.
    """
    # Average steps across seeds for each (optimizer, lr) combo
    grouped = (
        results.groupby(["optimizer", "lr"])["steps_to_threshold"]
        .mean()
        .reset_index()
    )

    # Create a readable label for each bar
    grouped["label"] = grouped.apply(
        lambda r: f"{r['optimizer']} (lr={r['lr']})", axis=1
    )

    # Sort: converged runs first (ascending steps), then non-converged
    converged = grouped[grouped["steps_to_threshold"] >= 0].sort_values("steps_to_threshold")
    not_converged = grouped[grouped["steps_to_threshold"] < 0]
    ordered = pd.concat([converged, not_converged])

    fig, ax = plt.subplots(figsize=(10, max(6, len(ordered) * 0.4)))

    labels = ordered["label"].tolist()
    steps = ordered["steps_to_threshold"].tolist()

    # For display: -1 gets a placeholder bar, 0 gets a small visible bar
    max_steps = max((s for s in steps if s >= 0), default=1)
    if max_steps == 0:
        max_steps = 1  # avoid zero-range axis

    display_steps = []
    for s in steps:
        if s < 0:
            display_steps.append(max_steps * 1.3)  # placeholder for "did not converge"
        elif s == 0:
            display_steps.append(0.15)  # thin visible bar for "converged immediately"
        else:
            display_steps.append(s)

    bar_colors = []
    for label, s in zip(labels, steps):
        opt_name = label.split(" ")[0]
        if s < 0:
            bar_colors.append("#cccccc")  # gray for did-not-converge
        else:
            bar_colors.append(COLORS.get(opt_name, "#333333"))

    bars = ax.barh(labels, display_steps, color=bar_colors)

    for bar, val in zip(bars, steps):
        if val < 0:
            text = "did not converge"
        elif val == 0:
            text = "epoch 0 (immediate)"
        else:
            text = f"epoch {val:.0f}"
        ax.text(
            bar.get_width() + 0.05, bar.get_y() + bar.get_height() / 2,
            text, va="center", fontsize=9,
        )

    ax.set_xlabel("Epochs to Threshold", fontsize=12)
    ax.set_title("Convergence Speed (lower = faster)", fontsize=14)
    ax.set_xlim(0, max_steps * 1.6)
    fig.tight_layout()
    fig.savefig(os.path.join(plots_dir, "convergence.png"), dpi=150)
    plt.close(fig)
    print("  Saved: convergence.png")


def print_summary_table(results: pd.DataFrame, plots_dir: str):
    """Print and save a seed-averaged summary table.

    Groups by (optimizer, lr), averages the metric across seeds,
    and shows mean ± std. This reveals which optimizers are both
    good AND consistent.
    """
    grouped = results.groupby(["optimizer", "lr"]).agg(
        metric_mean=("best_test_metric", "mean"),
        metric_std=("best_test_metric", "std"),
        time_mean=("total_time_s", "mean"),
        state_kb=("optimizer_state_bytes", "first"),
    ).reset_index()

    grouped["state_kb"] = grouped["state_kb"] / 1024
    grouped["metric_std"] = grouped["metric_std"].fillna(0)
    grouped = grouped.sort_values("metric_mean", ascending=False)

    # Format for display
    grouped["metric"] = grouped.apply(
        lambda r: f"{r['metric_mean']:.4f} ± {r['metric_std']:.4f}", axis=1
    )

    display_cols = ["optimizer", "lr", "metric", "time_mean", "state_kb"]
    table = grouped[display_cols].to_string(index=False)
    print(f"\n{table}")

    # Save to file
    grouped.to_csv(os.path.join(plots_dir, "summary_table.csv"), index=False)
    print("  Saved: summary_table.csv")


def main():
    parser = argparse.ArgumentParser(description="Plot sweep results")
    parser.add_argument(
        "--experiment", required=True,
        help="Experiment name (folder under runs/)"
    )
    parser.add_argument(
        "--runs-dir", default="runs",
        help="Root runs directory (default: runs)"
    )
    args = parser.parse_args()

    experiment_dir = os.path.join(args.runs_dir, args.experiment)
    results_path = os.path.join(experiment_dir, "results.csv")

    if not os.path.exists(results_path):
        print(f"No results found at {results_path}")
        print("Run the sweep first: python -m src.run_sweep --config configs/...")
        return

    results = pd.read_csv(results_path)
    print(f"Loaded {len(results)} runs from {results_path}")

    # Create plots directory
    plots_dir = os.path.join(experiment_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    # Generate all plots
    print("\nGenerating plots...")
    plot_loss_curves(results, experiment_dir, plots_dir)
    plot_metric_curves(results, experiment_dir, plots_dir)
    plot_optimizer_state_memory(results, plots_dir)
    plot_wall_clock(results, plots_dir)
    plot_convergence(results, plots_dir)
    print_summary_table(results, plots_dir)

    print(f"\nAll plots saved to {plots_dir}/")


if __name__ == "__main__":
    main()