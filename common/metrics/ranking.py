"""
common/metrics/ranking.py

Ranking metric computation. All evaluation scripts use these functions.
Metrics are computed over a candidate set of 1 target + N negatives.
"""

import numpy as np
import pandas as pd


METRIC_NAMES = ["ndcg@10", "hr@10", "ndcg@20", "hr@20", "mrr@10"]


def metrics_from_rank(rank: int) -> dict:
    """
    Compute all metrics for a single prediction given its 0-indexed rank
    in a candidate list sorted by score descending (rank=0 means top-1).
    """
    return {
        "hit@10":  int(rank < 10),
        "hit@20":  int(rank < 20),
        "ndcg@10": float(1.0 / np.log2(rank + 2)) if rank < 10 else 0.0,
        "ndcg@20": float(1.0 / np.log2(rank + 2)) if rank < 20 else 0.0,
        "mrr@10":  float(1.0 / (rank + 1))        if rank < 10 else 0.0,
    }


def summarize(df: pd.DataFrame) -> dict:
    """Aggregate per-user metric rows into mean metrics dict."""
    if len(df) == 0:
        return {"n_users": 0, "ndcg@10": 0.0, "hr@10": 0.0,
                "ndcg@20": 0.0, "hr@20": 0.0, "mrr@10": 0.0}
    return {
        "n_users":  int(len(df)),
        "ndcg@10":  float(df["ndcg@10"].mean()),
        "hr@10":    float(df["hit@10"].mean()),
        "ndcg@20":  float(df["ndcg@20"].mean()),
        "hr@20":    float(df["hit@20"].mean()),
        "mrr@10":   float(df["mrr@10"].mean()),
    }


def compute_deltas(baseline: dict, adapted: dict) -> dict:
    """
    Compute per-metric deltas and percentage changes.
    Returns dict keyed by metric name with baseline/adapted/delta/pct_change/status.
    """
    out = {}
    for m in METRIC_NAMES:
        b, a = float(baseline[m]), float(adapted[m])
        delta = a - b
        pct = (delta / b * 100) if b != 0 else 0.0
        out[m] = {
            "baseline":   b,
            "adapted":    a,
            "delta":      delta,
            "pct_change": pct,
            "status":     "IMPROVED" if delta > 1e-6 else ("WORSE" if delta < -1e-6 else "NO CHANGE"),
        }
    return out


def print_delta_report(title: str, deltas: dict):
    print(f"\n=== {title} ===")
    for m in METRIC_NAMES:
        d = deltas[m]
        print(f"{m:8s}  baseline={d['baseline']:.6f}  adapted={d['adapted']:.6f}"
              f"  delta={d['delta']:+.6f}  ({d['pct_change']:+.2f}%)  {d['status']}")
