#!/usr/bin/env python3
"""
data/create_dataset/build_dataset.py

Runner for the full dataset creation pipeline.
Calls the individual steps in order and respects skip-if-exists logic.

Steps:
  1. prepare_30music_csv.py   — convert raw source to standardised CSV
  2. create_subsets.py        — apply k-core filtering, train/val/test time split
  3. create_adaptation_split.py — split future data into future_adapt / future_test

Usage:
python data/create_dataset/build_dataset.py \
  --raw_data   data/raw/30music.tsv \
  --output_dir data/processed \
  --train_end  2014-01-01 \
  --val_end    2014-04-01
"""

import argparse
import subprocess
import sys
from pathlib import Path


def run(cmd: list, label: str, skip_if: Path = None):
    if skip_if and skip_if.exists():
        print(f"[build] SKIP {label} (output exists: {skip_if})")
        return True
    print(f"\n[build] >>> {label}")
    r = subprocess.run(cmd)
    if r.returncode != 0:
        print(f"[build] FAILED: {label}")
        return False
    return True


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--raw_data",   required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--train_end",  default="2014-01-01",
                   help="Cutoff timestamp for historical / future split")
    p.add_argument("--val_end",    default="2014-04-01",
                   help="Cutoff timestamp for adapt / test split within future")
    p.add_argument("--min_interactions", type=int, default=5,
                   help="k-core threshold (min interactions per user and item)")
    return p.parse_args()


def main():
    args   = parse_args()
    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    base   = Path(__file__).parent

    steps = [
        (
            [sys.executable, str(base / "prepare_30music_csv.py"),
             "--input", args.raw_data, "--output", str(outdir / "interactions_raw.csv")],
            "Step 1: prepare raw CSV",
            outdir / "interactions_raw.csv",
        ),
        (
            [sys.executable, str(base / "create_subsets.py"),
             "--input",     str(outdir / "interactions_raw.csv"),
             "--output_dir",str(outdir),
             "--train_end", args.train_end,
             "--min_count", str(args.min_interactions)],
            "Step 2: k-core + time split",
            outdir / "hist_kcore.csv",
        ),
        (
            [sys.executable, str(base / "create_adaptation_split.py"),
             "--future_data", str(outdir / "future.csv"),
             "--output_dir",  str(outdir),
             "--val_end",     args.val_end],
            "Step 3: adapt / test split",
            outdir / "future_adapt.csv",
        ),
    ]

    for cmd, label, skip_file in steps:
        if not run(cmd, label, skip_if=skip_file):
            sys.exit(1)

    print("\n[build] Dataset pipeline complete.")


if __name__ == "__main__":
    main()
