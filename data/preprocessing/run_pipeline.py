#!/usr/bin/env python3
"""
data/preprocessing/run_pipeline.py

Runner for the drift-based user filtering pipeline.
Selects high-drift users and filters data to overlap items.

Steps:
  1. filter_to_overlap_items_kcore.py  — restrict to items present in both hist and future
  2. filter_to_selected_users_kcore.py — apply k-core on the overlap subset
  3. detect_high_drift_users_overlap.py — score users by preference drift
  4. build_final_drift_scores.py        — attach user indices, sort by drift

Usage:
python data/preprocessing/run_pipeline.py \
  --hist_data    data/processed/hist_kcore.csv \
  --future_data  data/processed/future.csv \
  --backbone_ckpt results/backbone/sasrec_backbone_best.pt \
  --output_dir   data/processed/high_drift \
  --drift_top_k  0.3
"""

import argparse
import subprocess
import sys
from pathlib import Path


def run(cmd: list, label: str, skip_if: Path = None):
    if skip_if and skip_if.exists():
        print(f"[pipeline] SKIP {label} (output exists: {skip_if})")
        return True
    print(f"\n[pipeline] >>> {label}")
    r = subprocess.run(cmd)
    if r.returncode != 0:
        print(f"[pipeline] FAILED: {label}")
        return False
    return True


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--hist_data",      required=True)
    p.add_argument("--future_data",    required=True)
    p.add_argument("--backbone_ckpt",  required=True)
    p.add_argument("--output_dir",     required=True)
    p.add_argument("--drift_top_k",    type=float, default=0.3,
                   help="Top fraction of users by drift score to retain (default: top 30%%)")
    p.add_argument("--min_interactions", type=int, default=5)
    return p.parse_args()


def main():
    args   = parse_args()
    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    base   = Path(__file__).parent

    steps = [
        (
            [sys.executable, str(base / "filter_to_overlap_items_kcore.py"),
             "--hist_data",    args.hist_data,
             "--future_data",  args.future_data,
             "--output_dir",   str(outdir),
             "--min_count",    str(args.min_interactions)],
            "Step 1: filter to overlap items",
            outdir / "hist_overlap_kcore.csv",
        ),
        (
            [sys.executable, str(base / "filter_to_selected_users_kcore.py"),
             "--input_dir",  str(outdir),
             "--output_dir", str(outdir),
             "--min_count",  str(args.min_interactions)],
            "Step 2: k-core on overlap subset",
            outdir / "hist_overlap_users_kcore.csv",
        ),
        (
            [sys.executable, str(base / "detect_high_drift_users_overlap.py"),
             "--hist_data",     str(outdir / "hist_overlap_users_kcore.csv"),
             "--future_data",   args.future_data,
             "--backbone_ckpt", args.backbone_ckpt,
             "--output_dir",    str(outdir)],
            "Step 3: detect high-drift users",
            outdir / "user_drift_scores_overlap.csv",
        ),
        (
            [sys.executable, str(base / "build_final_drift_scores.py"),
             "--drift_scores",  str(outdir / "user_drift_scores_overlap.csv"),
             "--backbone_ckpt", args.backbone_ckpt,
             "--output_dir",    str(outdir),
             "--top_k",         str(args.drift_top_k)],
            "Step 4: build final drift scores + high-drift subset",
            outdir / "final_drift_scores.csv",
        ),
    ]

    for cmd, label, skip_file in steps:
        if not run(cmd, label, skip_if=skip_file):
            sys.exit(1)

    print("\n[pipeline] Preprocessing pipeline complete.")


if __name__ == "__main__":
    main()
