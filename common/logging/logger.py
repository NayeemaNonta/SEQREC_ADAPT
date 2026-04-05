"""
common/logging/logger.py

Experiment logger — writes results to CSV and JSON in a consistent format.
"""

import csv
import json
from pathlib import Path


class ExperimentLogger:
    """
    Accumulates per-epoch training rows and final eval results.
    Writes to <outdir>/train_history.json and <outdir>/summary.json.
    """

    def __init__(self, outdir: str | Path):
        self.outdir = Path(outdir)
        self.outdir.mkdir(parents=True, exist_ok=True)
        self.history: list[dict] = []

    def log_epoch(self, row: dict):
        self.history.append(row)

    def save_history(self):
        path = self.outdir / "train_history.json"
        with open(path, "w") as f:
            json.dump(self.history, f, indent=2)
        return path

    def save_summary(self, summary: dict):
        path = self.outdir / "summary.json"
        with open(path, "w") as f:
            json.dump(summary, f, indent=2)
        return path


class SweepLogger:
    """
    Accumulates sweep results and writes a running CSV after each completed run.
    """

    def __init__(self, outdir: str | Path, columns: list[str]):
        self.outdir  = Path(outdir)
        self.outdir.mkdir(parents=True, exist_ok=True)
        self.columns = columns
        self.rows:   list[dict] = []
        self.path    = self.outdir / "sweep_results.csv"

    def log_run(self, row: dict):
        self.rows.append(row)
        self._write()

    def _write(self):
        with open(self.path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.columns, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(self.rows)

    def best_run(self, key: str = "ndcg@10_finetuned") -> dict | None:
        if not self.rows:
            return None
        return max(self.rows, key=lambda r: r.get(key, 0.0))
