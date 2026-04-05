#!/usr/bin/env python3
"""
build_final_drift_scores.py

Filter drift scores from the full overlap subset down to only the users
that survived the final high-drift k-core filtering, and attach their
user_idx from the high-drift backbone checkpoint.

Inputs:
  --drift_scores_csv   user_drift_scores_overlap.csv  (from detect_high_drift_users_overlap.py)
  --hist_high_drift    hist_high_drift_kcore.csv       (from filter_to_selected_users_kcore.py)
  --checkpoint         sasrec_t1_best.pt               (high-drift backbone, for le_user)
  --outdir

Output:
  outdir/user_drift_scores_final_subset.csv

Usage:
python data_preprocessing/build_final_drift_scores.py \
  --drift_scores_csv results/overlap_item_subset_kcore/user_drift_overlap/user_drift_scores_overlap.csv \
  --hist_high_drift   results/overlap_item_subset_kcore/high_drift_overlap_subset_kcore/hist_high_drift_kcore.csv \
  --checkpoint        results/overlap_item_subset_kcore/high_drift_overlap_subset_kcore/t1_backbone_high_drift/sasrec_t1_best.pt \
  --outdir            results/overlap_item_subset_kcore/high_drift_overlap_subset_kcore
"""

import argparse
import sys
from pathlib import Path

import pandas as pd
import torch
from sklearn.preprocessing import LabelEncoder

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--drift_scores_csv", type=str, required=True)
    parser.add_argument("--hist_high_drift",  type=str, required=True)
    parser.add_argument("--checkpoint",        type=str, required=True)
    parser.add_argument("--outdir",            type=str, required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    drift = pd.read_csv(args.drift_scores_csv)
    final_hist = pd.read_csv(args.hist_high_drift)
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)

    final_users = set(final_hist["user_id"].astype(str).unique())
    drift["user_id"] = drift["user_id"].astype(str)

    le_user = LabelEncoder()
    le_user.classes_ = pd.Index(ckpt["le_user_classes"]).to_numpy(dtype=object)

    final_drift = drift[
        drift["user_id"].isin(final_users) &
        drift["user_id"].isin(set(le_user.classes_.tolist()))
    ].copy()

    final_drift["user_idx"] = le_user.transform(final_drift["user_id"])
    final_drift = final_drift.sort_values("combined_drift_score", ascending=False).reset_index(drop=True)

    out = outdir / "user_drift_scores_final_subset.csv"
    final_drift.to_csv(out, index=False)

    print(f"[done] saved: {out}")
    print(f"[done] rows: {len(final_drift)}")
    print(f"[done] unique users: {final_drift['user_id'].nunique()}")
    print(f"[done] unique user_idx: {final_drift['user_idx'].nunique()}")


if __name__ == "__main__":
    main()
