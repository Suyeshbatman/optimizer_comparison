"""Logging utilities: writes per-run curves and summaries to disk.

Directory structure created per run:
    runs/<experiment>/<run_id>/curves.csv
    runs/<experiment>/<run_id>/summary.json
    runs/<experiment>/results.csv          (one row appended per run)
"""
import os
import csv
import json


class RunLogger:
    """Handles all file I/O for a single training run.

    Parameters
    ----------
    experiment : str
        Name of the experiment (e.g. "mnist_cnn"). Becomes a folder
        under runs/.
    run_id : str
        Unique identifier for this run (e.g. "adam_lr0.001_seed42").
        Becomes a subfolder under the experiment folder.
    runs_dir : str
        Root directory for all outputs (default "runs").
    """

    def __init__(self, experiment: str, run_id: str, runs_dir: str = "runs"):
        self.run_dir = os.path.join(runs_dir, experiment, run_id)
        self.experiment_dir = os.path.join(runs_dir, experiment)
        os.makedirs(self.run_dir, exist_ok=True)

        # ── Curves CSV ───────────────────────────────────────────
        # We open the file once and keep it open for the whole run.
        # Each call to log_epoch() writes one row.
        self.curves_path = os.path.join(self.run_dir, "curves.csv")
        self.curves_fields = [
            "epoch",
            "train_loss",
            "test_loss",
            "train_metric",
            "test_metric",
            "epoch_time_s",
            "peak_gpu_mb",
            "grad_norm",
            "param_norm",
        ]
        self._curves_file = open(self.curves_path, "w", newline="")
        self._curves_writer = csv.DictWriter(
            self._curves_file, fieldnames=self.curves_fields
        )
        self._curves_writer.writeheader()

    def log_epoch(self, row: dict):
        """Append one row to curves.csv.

        Parameters
        ----------
        row : dict
            Must contain keys matching self.curves_fields.
            Example: {"epoch": 1, "train_loss": 0.45, ...}
        """
        self._curves_writer.writerow(row)
        # Flush immediately so we don't lose data if the run crashes.
        self._curves_file.flush()

    def log_summary(self, summary: dict):
        """Write the final summary.json for this run.

        Called once at the end of training with aggregated results
        like best_metric, total_time, optimizer_state_bytes, etc.
        """
        path = os.path.join(self.run_dir, "summary.json")
        with open(path, "w") as f:
            json.dump(summary, f, indent=2)

    def append_to_results(self, row: dict):
        """Append one row to the experiment-level results.csv.

        This is the master table — one row per completed run.
        plot_sweep.py reads this file to make comparative charts.

        We use append mode ("a") so multiple runs accumulate in the
        same file. The header is written only if the file doesn't
        exist yet.
        """
        results_path = os.path.join(self.experiment_dir, "results.csv")
        file_exists = os.path.exists(results_path)

        with open(results_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=row.keys())
            if not file_exists:
                writer.writeheader()
            writer.writerow(row)

    def close(self):
        """Close the curves CSV file handle."""
        self._curves_file.close()