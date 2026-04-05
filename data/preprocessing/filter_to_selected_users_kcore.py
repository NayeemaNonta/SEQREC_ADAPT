#!/usr/bin/env python3
"""
filter_to_selected_users_kcore.py

Filter interaction CSV files to a selected user subset, then apply hist k-core,
and finally restrict future files to the post-k-core hist item set.

This is the clean version for the high-drift overlap-user regime.

Pipeline:
  1. filter hist/future files to selected users
  2. apply iterative k-core to the filtered hist file
  3. define final overlap items from post-k-core filtered hist
  4. filter future_adapt / future_test to:
       - selected users
       - post-k-core hist items

Outputs:
  hist_high_drift_kcore.csv
  future_adapt_high_drift_kcore.csv
  future_test_high_drift_kcore.csv
  filter_kcore_summary.json

Usage:
python filter_to_selected_users_kcore.py \
  --selected_users_csv results/overlap_item_subset_kcore/user_drift_overlap/high_drift_users_top_overlap.csv \
  --hist_data results/overlap_item_subset_kcore/hist_overlap_items_kcore.csv \
  --future_adapt_data results/overlap_item_subset_kcore/future_adapt_overlap_items_kcore.csv \
  --future_test_data results/overlap_item_subset_kcore/future_test_overlap_items_kcore.csv \
  --outdir results/overlap_item_subset_kcore/high_drift_overlap_subset_kcore
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


def filter_users(df: pd.DataFrame, selected_users: set[str]) -> pd.DataFrame:
    out = df[df["user_id"].astype(str).isin(selected_users)].copy()
    return out.sort_values(["user_id", "timestamp"]).reset_index(drop=True)


def iterative_kcore(df: pd.DataFrame, user_min_len: int = 5, item_min_count: int = 5):
    df = df.copy()
    step = 1
    history = []

    while True:
        if len(df) == 0:
            break

        u_counts = df["user_id"].value_counts()
        i_counts = df["item_id"].value_counts()

        min_u = int(u_counts.min()) if len(u_counts) else 0
        min_i = int(i_counts.min()) if len(i_counts) else 0

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

    return df.sort_values(["user_id", "timestamp"]).reset_index(drop=True), history


def filter_users_and_items(df: pd.DataFrame, selected_users: set[str], allowed_items: set[str]) -> pd.DataFrame:
    out = df[
        df["user_id"].astype(str).isin(selected_users)
        & df["item_id"].astype(str).isin(allowed_items)
    ].copy()
    return out.sort_values(["user_id", "timestamp"]).reset_index(drop=True)


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
    parser.add_argument("--selected_users_csv", type=str, required=True)
    parser.add_argument("--hist_data", type=str, required=True)
    parser.add_argument("--future_adapt_data", type=str, required=True)
    parser.add_argument("--future_test_data", type=str, required=True)
    parser.add_argument("--outdir", type=str, required=True)
    parser.add_argument("--user_min_len", type=int, default=5)
    parser.add_argument("--item_min_count", type=int, default=5)
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    selected_df = pd.read_csv(args.selected_users_csv)
    if "user_id" not in selected_df.columns:
        raise ValueError(
            f"{args.selected_users_csv} must contain a user_id column. "
            f"Found columns: {selected_df.columns.tolist()}"
        )

    selected_users = set(selected_df["user_id"].astype(str).tolist())
    if len(selected_users) == 0:
        raise ValueError("No selected users found in selected_users_csv.")

    hist = read_interactions(args.hist_data)
    fut_adapt = read_interactions(args.future_adapt_data)
    fut_test = read_interactions(args.future_test_data)

    # Step 1: filter all files to selected users
    hist_sel = filter_users(hist, selected_users)
    fut_adapt_sel = filter_users(fut_adapt, selected_users)
    fut_test_sel = filter_users(fut_test, selected_users)

    # Step 2: apply k-core to filtered hist
    hist_kcore, kcore_history = iterative_kcore(
        hist_sel,
        user_min_len=args.user_min_len,
        item_min_count=args.item_min_count,
    )

    # After hist k-core, only keep the surviving selected users and items
    final_users = set(hist_kcore["user_id"].astype(str).tolist())
    final_items = set(hist_kcore["item_id"].astype(str).tolist())

    # Step 3: filter future files to final users + post-k-core hist items
    fut_adapt_final = filter_users_and_items(fut_adapt_sel, final_users, final_items)
    fut_test_final = filter_users_and_items(fut_test_sel, final_users, final_items)

    hist_out = outdir / "hist_high_drift_kcore.csv"
    fut_adapt_out = outdir / "future_adapt_high_drift_kcore.csv"
    fut_test_out = outdir / "future_test_high_drift_kcore.csv"

    hist_kcore.to_csv(hist_out, index=False)
    fut_adapt_final.to_csv(fut_adapt_out, index=False)
    fut_test_final.to_csv(fut_test_out, index=False)

    summary = {
        "selected_users_csv": args.selected_users_csv,
        "n_selected_users_input": int(len(selected_users)),
        "user_min_len": int(args.user_min_len),
        "item_min_count": int(args.item_min_count),

        **summarize(hist_sel, "hist_selected_users_raw"),
        **summarize(hist_kcore, "hist_high_drift_kcore"),
        **summarize(fut_adapt_sel, "future_adapt_selected_users_raw"),
        **summarize(fut_adapt_final, "future_adapt_high_drift_kcore"),
        **summarize(fut_test_sel, "future_test_selected_users_raw"),
        **summarize(fut_test_final, "future_test_high_drift_kcore"),

        "n_final_users_after_hist_kcore": int(len(final_users)),
        "n_final_items_after_hist_kcore": int(len(final_items)),
        "kcore_history": kcore_history,
    }

    with open(outdir / "filter_kcore_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))
    print("Saved:")
    print(f"  {hist_out}")
    print(f"  {fut_adapt_out}")
    print(f"  {fut_test_out}")
    print(f"  {outdir / 'filter_kcore_summary.json'}")


if __name__ == "__main__":
    main()