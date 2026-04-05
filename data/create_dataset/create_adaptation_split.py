#!/usr/bin/env python3
"""
create_adaptation_split.py

Produces two fixed temporal splits of a 30Music interaction CSV.

  TOTAL      = 10_000_000  (override with --total, e.g. --total 5000000)
  hist_size  = TOTAL // 2
  future_size = TOTAL - hist_size   (= hist_size when TOTAL is even)

Both splits share the same hist block (the first hist_size rows).
They differ only in where the future block comes from:

  split_contiguous/   future = rows [hist_size : hist_size + future_size]
                       (immediately after hist, no temporal gap)

  split_tail/         future = rows [-future_size:]
                       (the actual chronological tail of the full dataset)

Each future block is split 70 / 30 into future_adapt / future_test.

Outputs per directory
---------------------
  interactions_hist.csv
  interactions_future.csv
  interactions_future_adapt.csv
  interactions_future_test.csv
  split_metadata.json

Usage
-----
  python create_adaptation_split.py --src data/data_csv/30M.csv
  python create_adaptation_split.py --src data/data_csv/30M.csv --total 5000000
"""

import argparse
import json
import os

import pandas as pd


HIST_FRAC        = 0.5
FUTURE_ADAPT_FRAC = 0.7


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def overlap_stats(df_a: pd.DataFrame, df_b: pd.DataFrame, label_a: str, label_b: str) -> dict:
    users_a = set(df_a["user"].unique())
    users_b = set(df_b["user"].unique())
    items_a = set(df_a["item"].unique())
    items_b = set(df_b["item"].unique())

    user_overlap  = users_a & users_b
    item_overlap  = items_a & items_b
    new_users_b   = users_b - users_a
    new_items_b   = items_b - items_a

    return {
        f"{label_a}_users":          len(users_a),
        f"{label_b}_users":          len(users_b),
        "user_overlap":              len(user_overlap),
        "user_overlap_pct_of_b":     round(100 * len(user_overlap) / max(len(users_b), 1), 2),
        "new_users_in_b":            len(new_users_b),
        "new_users_pct_of_b":        round(100 * len(new_users_b) / max(len(users_b), 1), 2),
        f"{label_a}_items":          len(items_a),
        f"{label_b}_items":          len(items_b),
        "item_overlap":              len(item_overlap),
        "item_overlap_pct_of_b":     round(100 * len(item_overlap) / max(len(items_b), 1), 2),
        "new_items_in_b":            len(new_items_b),
        "new_items_pct_of_b":        round(100 * len(new_items_b) / max(len(items_b), 1), 2),
    }


def save_split(
    outdir: str,
    hist: pd.DataFrame,
    future: pd.DataFrame,
    future_adapt: pd.DataFrame,
    future_test: pd.DataFrame,
    split_label: str,
    src: str,
    total: int,
) -> None:
    os.makedirs(outdir, exist_ok=True)

    hist_path   = os.path.join(outdir, "interactions_hist.csv")
    future_path = os.path.join(outdir, "interactions_future.csv")
    adapt_path  = os.path.join(outdir, "interactions_future_adapt.csv")
    test_path   = os.path.join(outdir, "interactions_future_test.csv")

    hist.to_csv(hist_path,   index=False)
    future.to_csv(future_path, index=False)
    future_adapt.to_csv(adapt_path, index=False)
    future_test.to_csv(test_path,  index=False)

    hist_end_ts  = hist["timestamp"].max()
    adapt_end_ts = future_adapt["timestamp"].max()

    # Console summary
    hist_users   = set(hist["user"].unique())
    future_users = set(future["user"].unique())
    hist_items   = set(hist["item"].unique())
    future_items = set(future["item"].unique())
    adapt_items  = set(future_adapt["item"].unique())
    test_items   = set(future_test["item"].unique())

    new_users_future  = future_users - hist_users
    new_items_future  = future_items - hist_items
    new_items_adapt   = adapt_items  - hist_items
    new_items_test    = test_items   - hist_items
    test_only_items   = test_items   - adapt_items - hist_items

    sizes_mb = {p: os.path.getsize(p) / 1e6 for p in [hist_path, future_path, adapt_path, test_path]}

    print(f"\n  [{split_label}] → {outdir}/")
    print(f"    hist         : {len(hist):>10,} rows  ({sizes_mb[hist_path]:.1f} MB)")
    print(f"    future       : {len(future):>10,} rows  ({sizes_mb[future_path]:.1f} MB)")
    print(f"    future_adapt : {len(future_adapt):>10,} rows  ({sizes_mb[adapt_path]:.1f} MB)")
    print(f"    future_test  : {len(future_test):>10,} rows  ({sizes_mb[test_path]:.1f} MB)")
    print(f"    hist ts   <= {hist_end_ts:.0f}")
    print(f"    adapt ts  <= {adapt_end_ts:.0f}")
    print(f"    users  hist={len(hist_users):,}  future={len(future_users):,}  "
          f"new_in_future={len(new_users_future):,}  "
          f"({100 * len(new_users_future) / max(len(future_users), 1):.1f}%)")
    print(f"    items  hist={len(hist_items):,}  future={len(future_items):,}  "
          f"new_in_future={len(new_items_future):,}  "
          f"({100 * len(new_items_future) / max(len(future_items), 1):.1f}%)")
    print(f"           adapt new_vs_hist={len(new_items_adapt):,}  "
          f"({100 * len(new_items_adapt) / max(len(adapt_items), 1):.1f}%)")
    print(f"           test  new_vs_hist={len(new_items_test):,}  "
          f"({100 * len(new_items_test) / max(len(test_items), 1):.1f}%)")
    print(f"           test_only (not in hist or adapt)={len(test_only_items):,}")

    meta = {
        "generated_by":    "create_adaptation_split.py",
        "source_csv":      src,
        "split":           split_label,
        "total":           total,
        "hist_frac":       HIST_FRAC,
        "future_adapt_frac": FUTURE_ADAPT_FRAC,
        "row_counts": {
            "hist":         int(len(hist)),
            "future":       int(len(future)),
            "future_adapt": int(len(future_adapt)),
            "future_test":  int(len(future_test)),
        },
        "user_counts": {
            "hist":         int(hist["user"].nunique()),
            "future":       int(future["user"].nunique()),
            "future_adapt": int(future_adapt["user"].nunique()),
            "future_test":  int(future_test["user"].nunique()),
        },
        "item_counts": {
            "hist":         int(hist["item"].nunique()),
            "future":       int(future["item"].nunique()),
            "future_adapt": int(future_adapt["item"].nunique()),
            "future_test":  int(future_test["item"].nunique()),
        },
        "timestamp_boundaries": {
            "hist_min":               int(hist["timestamp"].min()),
            "hist_max":               int(hist["timestamp"].max()),
            "future_min":             int(future["timestamp"].min()),
            "future_max":             int(future["timestamp"].max()),
            "future_adapt_min":       int(future_adapt["timestamp"].min()) if len(future_adapt) else None,
            "future_adapt_max":       int(future_adapt["timestamp"].max()) if len(future_adapt) else None,
            "future_test_min":        int(future_test["timestamp"].min())  if len(future_test)  else None,
            "future_test_max":        int(future_test["timestamp"].max())  if len(future_test)  else None,
            "hist_future_boundary_ts": float(hist_end_ts),
            "adapt_test_boundary_ts":  float(adapt_end_ts),
        },
        "overlap": {
            "hist_future":              overlap_stats(hist, future,       "hist", "future"),
            "hist_future_adapt":        overlap_stats(hist, future_adapt, "hist", "future_adapt"),
            "hist_future_test":         overlap_stats(hist, future_test,  "hist", "future_test"),
            "future_adapt_future_test": overlap_stats(future_adapt, future_test, "future_adapt", "future_test"),
        },
    }

    meta_path = os.path.join(outdir, "split_metadata.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"    metadata → {meta_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--src",   default="data/data_csv/30M.csv",
                        help="Source CSV with columns: user, item, timestamp")
    parser.add_argument("--total", type=int, default=10_000_000,
                        help="Total interactions to use across hist + future (default 10M)")
    parser.add_argument("--outbase", default="data/data_csv/splits",
                        help="Parent directory for output split folders")
    args = parser.parse_args()

    print(f"Reading {args.src} ...")
    df = pd.read_csv(args.src)
    if not {"user", "item", "timestamp"}.issubset(df.columns):
        raise ValueError("CSV must contain columns: user, item, timestamp")

    print(f"  Loaded {len(df):,} rows")
    print("Sorting by timestamp ...")
    df = df.sort_values("timestamp").reset_index(drop=True)
    print(f"  Timestamp range: {df['timestamp'].min():.0f} – {df['timestamp'].max():.0f}")

    if args.total > len(df):
        raise ValueError(
            f"--total {args.total:,} exceeds dataset size {len(df):,}"
        )

    total       = args.total
    hist_size   = int(round(HIST_FRAC * total))
    future_size = total - hist_size
    adapt_size  = int(round(FUTURE_ADAPT_FRAC * future_size))

    print(f"\n  total={total:,}  hist={hist_size:,}  future={future_size:,}  "
          f"adapt={adapt_size:,}  test={future_size - adapt_size:,}")

    hist = df.iloc[:hist_size].copy()

    N = total // 1_000_000
    outbase = args.outbase

    # ---- Split A: contiguous (future immediately after hist) ---------------
    future_a       = df.iloc[hist_size:hist_size + future_size].copy()
    future_adapt_a = future_a.iloc[:adapt_size].copy()
    future_test_a  = future_a.iloc[adapt_size:].copy()
    save_split(
        outdir        = os.path.join(outbase, f"split_{N}M_contiguous"),
        hist          = hist,
        future        = future_a,
        future_adapt  = future_adapt_a,
        future_test   = future_test_a,
        split_label   = "contiguous",
        src           = args.src,
        total         = total,
    )

    # ---- Split B: tail (future = last future_size rows of full dataset) ----
    future_b       = df.iloc[-future_size:].copy()
    future_adapt_b = future_b.iloc[:adapt_size].copy()
    future_test_b  = future_b.iloc[adapt_size:].copy()

    gap = future_b["timestamp"].min() - hist["timestamp"].max()
    if gap < 0:
        print("\n  WARNING: hist and future_tail overlap in time. "
              "Reduce --total or increase the source dataset size.")
    else:
        print(f"\n  Temporal gap (hist → tail future): {gap:.0f} seconds")

    save_split(
        outdir        = os.path.join(outbase, f"split_{N}M_tail"),
        hist          = hist,
        future        = future_b,
        future_adapt  = future_adapt_b,
        future_test   = future_test_b,
        split_label   = "tail",
        src           = args.src,
        total         = total,
    )

    print("\nDone.")


if __name__ == "__main__":
    main()
