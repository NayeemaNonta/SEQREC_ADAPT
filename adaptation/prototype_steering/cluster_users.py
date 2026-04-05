#!/usr/bin/env python3
"""
adaptation/prototype_steering/cluster_users.py

Cluster users for prototype-steering adaptation using drift-score features
produced by the preprocessing pipeline (user_drift_scores_final_subset.csv).

Features used (from the preprocessing pipeline):
  - item_profile_cosine_distance
  - last_hidden_cosine_distance
  - topk_item_jaccard_drift
  - hist_len, future_len  (if present)

Output:
  <outdir>/user_clusters_K<K>.csv   — columns: user_idx, user_id, cluster_id
  <outdir>/cluster_summary_K<K>.json

Usage (standalone):
python adaptation/prototype_steering/cluster_users.py \
  --drift_scores_csv data/processed/split_10M_contiguous/user_drift_scores_final_subset.csv \
  --num_clusters 5 \
  --outdir       data/processed/split_10M_contiguous \
  --seed 42
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

FEATURE_COLS = [
    "item_profile_cosine_distance",
    "last_hidden_cosine_distance",
    "topk_item_jaccard_drift",
]
OPTIONAL_COLS = ["hist_len", "future_len"]


def cluster_users(drift_scores_csv: str, num_clusters: int, outdir: str, seed: int = 42) -> str:
    """
    Run KMeans clustering on drift-score features. Returns the path to the
    saved cluster CSV.

    Can be called programmatically from train.py.
    """
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(drift_scores_csv)

    required = {"user_idx", "user_id"} | set(FEATURE_COLS)
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"user_drift_scores CSV is missing required columns: {missing}\n"
            f"  found: {list(df.columns)}\n"
            f"  expected source: data/processed/<split>/user_drift_scores_final_subset.csv"
        )

    feature_cols = FEATURE_COLS.copy()
    for c in OPTIONAL_COLS:
        if c in df.columns:
            feature_cols.append(c)

    X = df[feature_cols].astype(float).values
    Xs = StandardScaler().fit_transform(X)

    kmeans = KMeans(n_clusters=num_clusters, random_state=seed, n_init=20)
    cluster_ids = kmeans.fit_predict(Xs)

    out_df = df[["user_idx", "user_id"]].copy()
    out_df["cluster_id"] = cluster_ids
    out_df = out_df.sort_values(["cluster_id", "user_idx"]).reset_index(drop=True)

    out_csv = outdir / f"user_clusters_K{num_clusters}.csv"
    out_df.to_csv(out_csv, index=False)

    cluster_sizes = out_df["cluster_id"].value_counts().sort_index().to_dict()
    summary = {
        "drift_scores_csv": str(drift_scores_csv),
        "num_users_clustered": int(len(out_df)),
        "num_clusters": int(num_clusters),
        "feature_columns": feature_cols,
        "cluster_sizes": {str(k): int(v) for k, v in cluster_sizes.items()},
        "seed": seed,
    }
    summary_path = outdir / f"cluster_summary_K{num_clusters}.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"[cluster] {len(out_df)} users → {num_clusters} clusters")
    for k, v in cluster_sizes.items():
        print(f"  cluster {k}: {v} users")
    print(f"[cluster] saved: {out_csv}")

    return str(out_csv)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--drift_scores_csv", required=True,
                   help="user_drift_scores_final_subset.csv from the preprocessing pipeline")
    p.add_argument("--num_clusters", type=int, default=5)
    p.add_argument("--outdir",       required=True,
                   help="Directory to write user_clusters_K<K>.csv")
    p.add_argument("--seed",         type=int, default=42)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    cluster_users(
        drift_scores_csv=args.drift_scores_csv,
        num_clusters=args.num_clusters,
        outdir=args.outdir,
        seed=args.seed,
    )
