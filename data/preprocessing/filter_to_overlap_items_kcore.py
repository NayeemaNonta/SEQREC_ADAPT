#!/usr/bin/env python3
"""
filter_to_overlap_items_kcore.py

Create an overlap-item regime using the POST-K-CORE hist item set, so the overlap
catalog matches the actual training catalog.

This script:
  1. loads raw hist
  2. applies the same iterative k-core filtering used in training
  3. defines overlap items as the surviving hist items after k-core
  4. filters future_adapt and future_test to those overlap items
  5. writes filtered CSVs

Outputs:
  outdir/hist_overlap_items_kcore.csv
  outdir/future_adapt_overlap_items_kcore.csv
  outdir/future_test_overlap_items_kcore.csv
  outdir/overlap_filter_kcore_summary.json

Usage:
python filter_to_overlap_items_kcore.py \
  --hist_data data2/interactions_hist.csv \
  --future_adapt_data data2/interactions_future_adapt.csv \
  --future_test_data data2/interactions_future_test.csv \
  --outdir results/overlap_item_subset_kcore
"""

import argparse
import json
from pathlib import Path

import pandas as pd


def read_interactions(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    rename_map = {}
    if "user" in df.columns:
        rename_map["user"] = "user_id"
    if "item" in df.columns:
        rename_map["item"] = "item_id"
    df = df.rename(columns=rename_map)

    required = {"user_id", "item_id", "timestamp"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path} missing required columns: {missing}")

    return df[list(required)].dropna().copy()


def iterative_kcore(df: pd.DataFrame, user_min_len: int = 5, item_min_count: int = 5):
    df = df.copy()
    step = 1
    history = []

    while True:
        u_counts = df["user_id"].value_counts()
        i_counts = df["item_id"].value_counts()

        if len(u_counts) == 0 or len(i_counts) == 0:
            break

        min_u = int(u_counts.min())
        min_i = int(i_counts.min())

        history.append({
            "step": step,
            "rows": int(len(df)),
            "users": int(df["user_id"].nunique()),
            "items": int(df["item_id"].nunique()),
            "min_user_count": min_u,
            "min_item_count": min_i,
        })

        if min_u >= user_min_len and min_i >= item_min_count:
            break

        df = df[df["user_id"].isin(u_counts[u_counts >= user_min_len].index)]
        df = df[df["item_id"].isin(i_counts[i_counts >= item_min_count].index)]
        step += 1

    return df, history


def summarize(df: pd.DataFrame, name: str):
    return {
        f"{name}_rows": int(len(df)),
        f"{name}_users": int(df["user_id"].nunique()) if len(df) else 0,
        f"{name}_items": int(df["item_id"].nunique()) if len(df) else 0,
        f"{name}_start_ts": float(df["timestamp"].min()) if len(df) else None,
        f"{name}_end_ts": float(df["timestamp"].max()) if len(df) else None,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hist_data", type=str, required=True)
    parser.add_argument("--future_adapt_data", type=str, required=True)
    parser.add_argument("--future_test_data", type=str, required=True)
    parser.add_argument("--outdir", type=str, required=True)
    parser.add_argument("--user_min_len", type=int, default=5)
    parser.add_argument("--item_min_count", type=int, default=5)
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    print(f"[data] reading hist: {args.hist_data}")
    hist_raw = read_interactions(args.hist_data)
    print(f"[data] reading future_adapt: {args.future_adapt_data}")
    fut_adapt_raw = read_interactions(args.future_adapt_data)
    print(f"[data] reading future_test: {args.future_test_data}")
    fut_test_raw = read_interactions(args.future_test_data)

    print("[kcore] applying iterative k-core to hist...")
    hist_kcore, kcore_history = iterative_kcore(
        hist_raw,
        user_min_len=args.user_min_len,
        item_min_count=args.item_min_count,
    )

    overlap_items = set(hist_kcore["item_id"].astype(str).tolist())

    hist_f = hist_kcore.sort_values(["user_id", "timestamp"]).reset_index(drop=True)
    fut_adapt_f = fut_adapt_raw[fut_adapt_raw["item_id"].astype(str).isin(overlap_items)].copy()
    fut_test_f = fut_test_raw[fut_test_raw["item_id"].astype(str).isin(overlap_items)].copy()

    fut_adapt_f = fut_adapt_f.sort_values(["user_id", "timestamp"]).reset_index(drop=True)
    fut_test_f = fut_test_f.sort_values(["user_id", "timestamp"]).reset_index(drop=True)

    hist_out = outdir / "hist_overlap_items_kcore.csv"
    fut_adapt_out = outdir / "future_adapt_overlap_items_kcore.csv"
    fut_test_out = outdir / "future_test_overlap_items_kcore.csv"

    hist_f.to_csv(hist_out, index=False)
    fut_adapt_f.to_csv(fut_adapt_out, index=False)
    fut_test_f.to_csv(fut_test_out, index=False)

    summary = {
        "hist_data": args.hist_data,
        "future_adapt_data": args.future_adapt_data,
        "future_test_data": args.future_test_data,
        "user_min_len": int(args.user_min_len),
        "item_min_count": int(args.item_min_count),
        "raw_hist_rows": int(len(hist_raw)),
        "raw_hist_users": int(hist_raw["user_id"].nunique()),
        "raw_hist_items": int(hist_raw["item_id"].nunique()),
        "post_kcore_hist_rows": int(len(hist_kcore)),
        "post_kcore_hist_users": int(hist_kcore["user_id"].nunique()),
        "post_kcore_hist_items": int(hist_kcore["item_id"].nunique()),
        "n_overlap_items_defined_from_post_kcore_hist": int(len(overlap_items)),
        **summarize(hist_f, "hist_overlap_items_kcore"),
        **summarize(fut_adapt_f, "future_adapt_overlap_items_kcore"),
        **summarize(fut_test_f, "future_test_overlap_items_kcore"),
        "kcore_history": kcore_history,
    }

    with open(outdir / "overlap_filter_kcore_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))
    print("Saved:")
    print(f"  {hist_out}")
    print(f"  {fut_adapt_out}")
    print(f"  {fut_test_out}")
    print(f"  {outdir / 'overlap_filter_kcore_summary.json'}")


if __name__ == "__main__":
    main()