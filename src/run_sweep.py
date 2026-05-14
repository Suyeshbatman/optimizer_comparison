"""Sweep runner — the single entry point for all experiments.

Usage:
    python src/run_sweep.py --config configs/mnist_cnn.yaml
    python src/run_sweep.py --config configs/mnist_cnn.yaml --only adam,adamw
    python src/run_sweep.py --config configs/mnist_cnn.yaml --dry-run

Reads the YAML config, expands the cartesian product of
(optimizer × hyperparameter grid × seeds), and trains each combination.
"""
import argparse
import itertools
import yaml
import traceback
import torch

from src.seed import set_seed
from src.device import get_device
from src.datasets import build_dataset
from src.models import build_model
from src.optim.registry import build_optimizer
from src.bench.trainer import train_one_config
from src.bench.logger import RunLogger


def expand_grid(optimizer_cfg: dict, seeds: list) -> list:
    """Expand one optimizer's config into a list of individual run configs.

    Example:
        optimizer_cfg = {"name": "adam", "grid": {"lr": [0.001, 0.003]}}
        seeds = [42, 43]

        Returns 4 runs:
            adam_lr0.001_seed42, adam_lr0.001_seed43,
            adam_lr0.003_seed42, adam_lr0.003_seed43

    The trick is itertools.product — it computes the cartesian product
    of all grid values. If grid has lr=[a,b] and weight_decay=[c,d],
    product gives: (a,c), (a,d), (b,c), (b,d).
    """
    name = optimizer_cfg["name"]
    grid = optimizer_cfg.get("grid", {"lr": [0.001]})

    # Get all hyperparam names and their value lists
    keys = list(grid.keys())
    value_lists = [grid[k] if isinstance(grid[k], list) else [grid[k]] for k in keys]

    runs = []
    for values in itertools.product(*value_lists):
        hparams = dict(zip(keys, values))
        for seed in seeds:
            # Build a human-readable run ID
            hp_str = "_".join(f"{k}{v}" for k, v in hparams.items())
            run_id = f"{name}_{hp_str}_seed{seed}"
            runs.append({
                "optimizer_name": name,
                "hparams": hparams,
                "seed": seed,
                "run_id": run_id,
            })
    return runs


def main():
    # ── Parse command-line arguments ─────────────────────────────
    parser = argparse.ArgumentParser(description="Optimizer Benchmark Sweep")
    parser.add_argument(
        "--config", required=True,
        help="Path to YAML config file (e.g. configs/mnist_cnn.yaml)"
    )
    parser.add_argument(
        "--only", default=None,
        help="Comma-separated optimizer names to run (e.g. adam,adamw). "
             "Skips all others."
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print the run list without actually training."
    )
    args = parser.parse_args()

    # ── Load config ──────────────────────────────────────────────
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    experiment = cfg["experiment"]
    seeds = cfg["seeds"]
    only_filter = set(args.only.split(",")) if args.only else None

    # ── Expand all runs ──────────────────────────────────────────
    all_runs = []
    for opt_cfg in cfg["optimizers"]:
        if only_filter and opt_cfg["name"] not in only_filter:
            continue
        all_runs.extend(expand_grid(opt_cfg, seeds))

    print(f"Experiment: {experiment}")
    print(f"Total runs: {len(all_runs)}")
    print()

    if args.dry_run:
        for i, run in enumerate(all_runs, 1):
            print(f"  [{i:3d}] {run['run_id']}  {run['hparams']}")
        print("\n(dry run — nothing trained)")
        return

    # ── Build dataset (once — shared across all runs) ────────────
    device = get_device()
    print(f"Device: {device}")
    train_loader, test_loader, meta = build_dataset(cfg["dataset"], cfg)

    # ── Run the sweep ────────────────────────────────────────────
    completed = 0
    failed = 0

    for i, run in enumerate(all_runs, 1):
        run_id = run["run_id"]
        print(f"\n[{i}/{len(all_runs)}] {run_id}")
        print(f"  hparams: {run['hparams']}")

        try:
            # Seed everything for this run
            set_seed(run["seed"])

            # Build a fresh model (new random weights from this seed)
            model = build_model(cfg["model"], meta).to(device)

            # Build the optimizer
            optimizer = build_optimizer(
                run["optimizer_name"],
                model.parameters(),
                run["hparams"],
            )

            # Configure the run
            run_cfg = {
                "epochs": cfg["epochs"],
                "device": device,
                "optimizer_name": run["optimizer_name"],
                "lr": run["hparams"]["lr"],
                "seed": run["seed"],
                "threshold": cfg.get("threshold", None),
            }

            # Create logger
            logger = RunLogger(experiment=experiment, run_id=run_id)

            # Train!
            summary = train_one_config(
                model, train_loader, test_loader,
                optimizer, run_cfg, meta, logger,
            )

            # Append to the master results.csv
            cls_m = summary.get("classification_metrics", {})
            reg_m = summary.get("regression_metrics", {})
            logger.append_to_results({
                "run_id": run_id,
                "optimizer": run["optimizer_name"],
                "lr": run["hparams"]["lr"],
                "seed": run["seed"],
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

            # Save detailed metrics JSON
            import json as _json
            detail = {}
            if "classification_metrics" in summary:
                detail["classification_metrics"] = summary["classification_metrics"]
            if "regression_metrics" in summary:
                detail["regression_metrics"] = summary["regression_metrics"]
            if detail:
                detail_path = os.path.join(logger.run_dir, "detailed_metrics.json")
                with open(detail_path, "w") as _f:
                    _json.dump(detail, _f, indent=2)

            logger.close()

            # Save model for prediction page
            model_save_dir = os.path.join("runs", experiment, "models")
            os.makedirs(model_save_dir, exist_ok=True)
            model_path = os.path.join(
                model_save_dir, f"{run['optimizer_name']}_lr{run['hparams']['lr']}_seed{run['seed']}.pt"
            )
            torch.save({
                "model_state_dict": model.state_dict(),
                "model_name": cfg["model"],
                "dataset": cfg["dataset"],
                "optimizer": run["optimizer_name"],
                "lr": run["hparams"]["lr"],
                "seed": run["seed"],
                "meta": meta,
                "summary": summary,
            }, model_path)

            completed += 1

        except Exception as e:
            # One run crashing must not abort the whole sweep.
            print(f"  FAILED: {e}")
            traceback.print_exc()
            failed += 1

    # ── Summary ──────────────────────────────────────────────────
    print(f"\n{'='*50}")
    print(f"Sweep complete: {completed} succeeded, {failed} failed")
    print(f"Results: runs/{experiment}/results.csv")


if __name__ == "__main__":
    main()