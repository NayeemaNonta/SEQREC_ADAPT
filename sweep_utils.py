"""
sweep_utils.py

Shared helpers for all adaptation sweep scripts.
"""

import json
import subprocess
import sys
import time
from pathlib import Path

from common.metrics.ranking import METRIC_NAMES


def run_cmd(cmd: list, label: str) -> bool:
    print(f"[sweep] >>> {label}: " + " ".join(str(c) for c in cmd[-6:]))
    t0 = time.time()
    result = subprocess.run(cmd)
    elapsed = time.time() - t0
    if result.returncode != 0:
        print(f"[sweep] FAILED (exit {result.returncode}) after {elapsed:.1f}s")
        return False
    print(f"[sweep] OK in {elapsed:.1f}s")
    return True


def load_summary(path: Path) -> dict | None:
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def build_sweep_row(run_id: str, cfg: dict, summary: dict, key_adapted: str = "adapted_metrics") -> dict:
    """
    Build a flat CSV row from a summary.json dict.
    key_adapted: the key under summary["eval"] that holds the adapted metrics
                 (e.g. "finetuned_metrics" for last_block, "adapted_metrics" for context_gate).
    """
    ev   = summary["eval"]
    base = ev["baseline_metrics"]
    adpt = ev[key_adapted]

    row = {
        "run_id":          run_id,
        "best_eval_loss":  adpt.get("best_eval_loss", ""),
        **{k: v for k, v in cfg.items()},
    }

    n_improved = 0
    for m in METRIC_NAMES:
        b, a  = float(base[m]), float(adpt[m])
        delta = a - b
        pct   = (delta / b * 100) if b != 0 else 0.0
        row[f"{m}_baseline"]   = round(b,     6)
        row[f"{m}_finetuned"]  = round(a,     6)   # generic name; always present
        row[f"{m}_delta"]      = round(delta, 6)
        row[f"{m}_pct_change"] = round(pct,   4)
        if delta > 1e-9:
            n_improved += 1

    row["n_improved"] = n_improved
    row["status"]     = (
        "ALL_IMPROVED" if n_improved == len(METRIC_NAMES)
        else f"{n_improved}/{len(METRIC_NAMES)}_improved"
    )
    return row


def print_best(best: dict | None, metrics: list[str]):
    if best is None:
        return
    print(f"\n[sweep] Best run: {best['run_id']}")
    for m in metrics:
        delta = best.get(f"{m}_delta", 0)
        pct   = best.get(f"{m}_pct_change", 0)
        base  = best.get(f"{m}_baseline", 0)
        fine  = best.get(f"{m}_finetuned", base)
        print(f"  {m:8s}  {base:.6f} → {fine:.6f}  delta={delta:+.6f}  ({pct:+.2f}%)")
