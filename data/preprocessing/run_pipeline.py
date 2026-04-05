#!/usr/bin/env python3
"""
data/preprocessing/run_pipeline.py

Runner for the drift-based user filtering pipeline.

Steps (in dependency order):
  1. filter_to_overlap_items_kcore.py   — k-core hist; filter future files to surviving items
  2. backbone/train_backbone.py         — train SASRec backbone on overlap-filtered hist
  3. detect_high_drift_users_overlap.py — score users by preference drift (needs backbone + overlap data)
  4. filter_to_selected_users_kcore.py  — filter to high-drift users + re-apply k-core
  5. build_final_drift_scores.py        — attach user indices to final drift scores

Usage:
python data/preprocessing/run_pipeline.py \
  --hist_data         data/data_csv/splits/split_10M_contiguous/interactions_hist.csv \
  --future_adapt_data data/data_csv/splits/split_10M_contiguous/interactions_future_adapt.csv \
  --future_test_data  data/data_csv/splits/split_10M_contiguous/interactions_future_test.csv \
  --backbone_ckpt     results/backbone/sasrec_backbone_best.pt \
  --output_dir        data/processed/split_10M_contiguous \
  --device cuda
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
    p.add_argument("--hist_data",         required=True)
    p.add_argument("--future_adapt_data", required=True)
    p.add_argument("--future_test_data",  required=True)
    p.add_argument("--backbone_ckpt",     required=True)
    p.add_argument("--output_dir",        required=True)
    p.add_argument("--drift_top_pct",     type=float, default=0.1,
                   help="Top fraction of users by drift score to retain (default: top 10%%)")
    p.add_argument("--min_interactions",  type=int, default=5,
                   help="Min interactions per user/item for k-core (default: 5)")
    p.add_argument("--device",            default="cuda")
    return p.parse_args()


def main():
    args        = parse_args()
    outdir      = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    base        = Path(__file__).parent
    repo_root   = base.parent.parent
    backbone_ckpt = Path(args.backbone_ckpt)

    min_int = str(args.min_interactions)

    steps = [
        (
            [sys.executable, str(base / "filter_to_overlap_items_kcore.py"),
             "--hist_data",          args.hist_data,
             "--future_adapt_data",  args.future_adapt_data,
             "--future_test_data",   args.future_test_data,
             "--outdir",             str(outdir),
             "--user_min_len",       min_int,
             "--item_min_count",     min_int],
            "Step 1: filter to overlap items + k-core hist",
            outdir / "hist_overlap_items_kcore.csv",
        ),
        (
            [sys.executable, str(repo_root / "backbone" / "train_backbone.py"),
             "--hist_data",   str(outdir / "hist_overlap_items_kcore.csv"),
             "--val_data",    str(outdir / "future_adapt_overlap_items_kcore.csv"),
             "--output_dir",  str(backbone_ckpt.parent),
             "--device",      args.device],
            "Step 2: train backbone on overlap-filtered hist",
            backbone_ckpt,
        ),
        (
            [sys.executable, str(base / "detect_high_drift_users_overlap.py"),
             "--checkpoint",  args.backbone_ckpt,
             "--hist_data",   str(outdir / "hist_overlap_items_kcore.csv"),
             "--future_data", str(outdir / "future_adapt_overlap_items_kcore.csv"),
             "--outdir",      str(outdir),
             "--top_pct",     str(args.drift_top_pct),
             "--device",      args.device],
            "Step 3: detect high-drift users",
            outdir / "user_drift_scores_overlap.csv",
        ),
        (
            [sys.executable, str(base / "filter_to_selected_users_kcore.py"),
             "--selected_users_csv", str(outdir / "high_drift_users_top_overlap.csv"),
             "--hist_data",          str(outdir / "hist_overlap_items_kcore.csv"),
             "--future_adapt_data",  str(outdir / "future_adapt_overlap_items_kcore.csv"),
             "--future_test_data",   str(outdir / "future_test_overlap_items_kcore.csv"),
             "--outdir",             str(outdir),
             "--user_min_len",       min_int,
             "--item_min_count",     min_int],
            "Step 4: filter to high-drift users + re-apply k-core",
            outdir / "hist_high_drift_kcore.csv",
        ),
        (
            [sys.executable, str(base / "build_final_drift_scores.py"),
             "--drift_scores_csv", str(outdir / "user_drift_scores_overlap.csv"),
             "--hist_high_drift",  str(outdir / "hist_high_drift_kcore.csv"),
             "--checkpoint",       args.backbone_ckpt,
             "--outdir",           str(outdir)],
            "Step 5: build final drift scores",
            outdir / "user_drift_scores_final_subset.csv",
        ),
    ]

    for cmd, label, skip_file in steps:
        if not run(cmd, label, skip_if=skip_file):
            sys.exit(1)

    print("\n[pipeline] Preprocessing pipeline complete.")


if __name__ == "__main__":
    main()
