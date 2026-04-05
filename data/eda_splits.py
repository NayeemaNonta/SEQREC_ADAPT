#!/usr/bin/env python3
"""
data/eda_splits.py

EDA for the temporal adaptation splits produced by create_adaptation_split.py
and (optionally) the preprocessed high-drift subset from run_pipeline.py.

Expected files in --split_dir:
  interactions_hist.csv
  interactions_future.csv
  interactions_future_adapt.csv
  interactions_future_test.csv

Optional --processed_dir (output of run_pipeline.py) adds:
  user_drift_scores_final_subset.csv   — richer drift metrics from the backbone

Outputs in --outdir:
  temporal_ranges.png
  temporal_density.png
  top_items_share.png
  distribution_shift_summary.png
  new_item_activation_over_time.png
  seq_length_distribution.png
  user_drift_histogram.png              (multi-metric if --processed_dir given)
  rank_frequency_curves.png
  split_summary.json
  user_drift_scores.csv                 (only if drift recomputed from raw splits)

Usage:
python data/eda_splits.py \
  --split_dir     data/data_csv/splits/split_10M_contiguous \
  --outdir        results/eda/split_10M_contiguous

# With preprocessed drift scores and full-data background:
python data/eda_splits.py \
  --split_dir     data/data_csv/splits/split_10M_contiguous \
  --processed_dir data/processed/split_10M_contiguous \
  --full_data     data/data_csv/30M.csv \
  --outdir        results/eda/split_10M_contiguous
"""

import argparse
import json
import os
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# ---------------------------------------------------------------------------
# Theme
# ---------------------------------------------------------------------------

sns.set_theme(style="whitegrid", font_scale=1.5)
PALETTE   = sns.color_palette("viridis", 5)
# "Full" uses a neutral grey so it reads as a background reference, not a split
COLOR_MAP = {
    "Full":         "#aaaaaa",
    "Historical":   PALETTE[0],
    "Future":       PALETTE[1],
    "Future Adapt": PALETTE[2],
    "Future Test":  PALETTE[3],
}

DPI = 300


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def load_interactions(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if not {"user", "item", "timestamp"}.issubset(df.columns):
        raise ValueError(f"{path} must contain columns: user, item, timestamp")
    df = df.sort_values("timestamp").reset_index(drop=True)
    df["dt"] = pd.to_datetime(df["timestamp"], unit="s", utc=True).dt.tz_localize(None)
    return df


def period_counts(dt_series: pd.Series, freq: str) -> pd.Series:
    return dt_series.dt.to_period(freq).value_counts().sort_index()



def overlap_stats(a: pd.DataFrame, b: pd.DataFrame, name_a: str, name_b: str) -> dict:
    users_a, users_b = set(a["user"].unique()), set(b["user"].unique())
    items_a, items_b = set(a["item"].unique()), set(b["item"].unique())
    return {
        f"{name_a}_rows":          int(len(a)),
        f"{name_b}_rows":          int(len(b)),
        f"{name_a}_users":         int(len(users_a)),
        f"{name_b}_users":         int(len(users_b)),
        f"{name_a}_items":         int(len(items_a)),
        f"{name_b}_items":         int(len(items_b)),
        "user_overlap":            int(len(users_a & users_b)),
        "item_overlap":            int(len(items_a & items_b)),
        "new_users_in_b":          int(len(users_b - users_a)),
        "new_items_in_b":          int(len(items_b - items_a)),
        "new_users_in_b_pct":      float(100 * len(users_b - users_a) / max(len(users_b), 1)),
        "new_items_in_b_pct":      float(100 * len(items_b - items_a) / max(len(items_b), 1)),
    }


def top_k_share(reference: pd.DataFrame, target: pd.DataFrame, k: int = 1000) -> float:
    top_items = reference["item"].value_counts().head(k).index
    return float(target["item"].isin(top_items).mean())



def new_item_activation_df(hist: pd.DataFrame, future: pd.DataFrame, freq: str = "W") -> pd.DataFrame:
    hist_items = set(hist["item"].unique())
    tmp = future[["item", "dt"]].copy()
    tmp["period"]     = tmp["dt"].dt.to_period(freq).dt.start_time
    tmp["is_new_item"] = ~tmp["item"].isin(hist_items)
    out = (
        tmp.groupby("period")
        .agg(
            total_interactions   = ("item",        "size"),
            new_item_interactions = ("is_new_item", "sum"),
            active_items         = ("item",        "nunique"),
            active_new_items     = ("item",        lambda x: x[~x.isin(hist_items)].nunique()),
        )
        .reset_index()
    )
    out["new_item_interaction_share"] = (
        out["new_item_interactions"] / out["total_interactions"].clip(lower=1)
    )
    out["new_item_active_share"] = (
        out["active_new_items"] / out["active_items"].clip(lower=1)
    )
    return out


def compute_user_drift_scores(hist: pd.DataFrame, future: pd.DataFrame) -> pd.DataFrame:
    """Recompute item-profile cosine distance and Jaccard distance from raw splits."""
    hist_counts   = hist.groupby(["user", "item"]).size().rename("hist_count").reset_index()
    future_counts = future.groupby(["user", "item"]).size().rename("future_count").reset_index()
    merged = pd.merge(hist_counts, future_counts, on=["user", "item"], how="outer").fillna(0)

    rows = []
    for user, grp in merged.groupby("user"):
        x = grp["hist_count"].to_numpy(dtype=float)
        y = grp["future_count"].to_numpy(dtype=float)
        xn, yn = np.linalg.norm(x), np.linalg.norm(y)
        cosine_dist = float(1.0 - np.dot(x, y) / (xn * yn)) if xn > 0 and yn > 0 else np.nan
        hist_items   = set(grp.loc[grp["hist_count"]   > 0, "item"])
        future_items = set(grp.loc[grp["future_count"] > 0, "item"])
        union = len(hist_items | future_items)
        jaccard_dist = float(1.0 - len(hist_items & future_items) / union) if union > 0 else np.nan
        rows.append({
            "user":                user,
            "hist_interactions":   int(x.sum()),
            "future_interactions": int(y.sum()),
            "hist_unique_items":   int((x > 0).sum()),
            "future_unique_items": int((y > 0).sum()),
            "cosine_distance":     cosine_dist,
            "jaccard_distance":    jaccard_dist,
        })
    return pd.DataFrame(rows)


def rank_frequency_df(df: pd.DataFrame, label: str) -> pd.DataFrame:
    counts = df["item"].value_counts().reset_index()
    counts.columns = ["item", "count"]
    counts["rank"]  = np.arange(1, len(counts) + 1)
    counts["label"] = label
    counts["share"] = counts["count"] / counts["count"].sum()
    return counts


def seq_lengths(df: pd.DataFrame) -> pd.Series:
    return df.groupby("user")["item"].count()


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------

def _boundary_lines(ax, hist_end, adapt_end, label: bool = False):
    kw = dict(color="#333333", linewidth=1.5, alpha=0.85)
    ax.axvline(hist_end,  linestyle="--", **kw,
               label=f"hist/future boundary ({hist_end.date()})" if label else None)
    ax.axvline(adapt_end, linestyle=":",  **kw,
               label=f"adapt/test boundary ({adapt_end.date()})" if label else None)


def _fmt_xdate(ax, interval: int = 3):
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=interval))
    plt.setp(ax.get_xticklabels(), rotation=35, ha="right")


# ---------------------------------------------------------------------------
# Individual plots
# ---------------------------------------------------------------------------

def plot_temporal_ranges(data: dict, hist_end, adapt_end, outdir: Path):
    labels = list(data.keys())
    fig, ax = plt.subplots(figsize=(13, 4.5))
    for i, label in enumerate(reversed(labels)):
        ts    = data[label]["dt"]
        start = mdates.date2num(ts.min())
        end   = mdates.date2num(ts.max())
        ax.barh(i, end - start, left=start,
                color=COLOR_MAP.get(label, PALETTE[0]),
                alpha=0.85, edgecolor="white", linewidth=0.8, height=0.55)
        ax.text(end + 4, i, f"{len(ts)/1e6:.2f}M", va="center", fontsize=11)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(list(reversed(labels)))
    ax.xaxis_date()
    _fmt_xdate(ax, interval=3)
    _boundary_lines(ax, hist_end, adapt_end)
    ax.set_title("Temporal coverage of data blocks", fontweight="bold")
    ax.set_xlabel("Date")
    fig.tight_layout()
    fig.savefig(outdir / "temporal_ranges.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def plot_temporal_density(data: dict, hist_end, adapt_end, outdir: Path):
    fig, ax = plt.subplots(figsize=(14, 5.5))
    for label, df in data.items():
        counts  = period_counts(df["dt"], "D")
        periods = [p.to_timestamp() for p in counts.index]
        y       = counts.values / 1e3
        alpha   = 0.3 if label == "Full" else 0.6
        ax.fill_between(periods, y, color=COLOR_MAP.get(label, PALETTE[0]),
                        alpha=alpha, label=label)
    _boundary_lines(ax, hist_end, adapt_end, label=True)
    _fmt_xdate(ax, interval=3)
    ax.set_title("Daily interaction density across splits", fontweight="bold")
    ax.set_ylabel("Interactions (thousands)")
    ax.legend(fontsize=11, ncol=3, loc="upper left")
    fig.tight_layout()
    fig.savefig(outdir / "temporal_density.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)



def plot_top_items_share(hist, future_adapt, future_test, topk: int, outdir: Path):
    labels = ["Historical", "Future Adapt", "Future Test"]
    shares = [
        top_k_share(hist, hist,          topk),
        top_k_share(hist, future_adapt,  topk),
        top_k_share(hist, future_test,   topk),
    ]
    colors = [COLOR_MAP["Historical"], COLOR_MAP["Future Adapt"], COLOR_MAP["Future Test"]]
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(labels, shares, color=colors, width=0.5)
    for b in bars:
        h = b.get_height()
        ax.text(b.get_x() + b.get_width() / 2, h + 0.008, f"{100*h:.1f}%",
                ha="center", va="bottom", fontsize=12, fontweight="bold")
    ax.set_ylim(0, min(1.0, max(shares) * 1.22))
    ax.set_ylabel(f"Share of interactions\nfrom top-{topk} historical items")
    ax.set_title("Popularity concentration shift", fontweight="bold")
    fig.tight_layout()
    fig.savefig(outdir / "top_items_share.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def plot_distribution_shift(hist, future_adapt, future_test, outdir: Path):
    hist_users  = set(hist["user"].unique())
    hist_items  = set(hist["item"].unique())
    adp_users   = set(future_adapt["user"].unique())
    adp_items   = set(future_adapt["item"].unique())
    tst_users   = set(future_test["user"].unique())
    tst_items   = set(future_test["item"].unique())

    stats = {
        "new users in adapt (%)":    100 * len(adp_users - hist_users) / max(len(adp_users), 1),
        "new users in test (%)":     100 * len(tst_users - hist_users) / max(len(tst_users), 1),
        "new items in adapt (%)":    100 * len(adp_items - hist_items) / max(len(adp_items), 1),
        "new items in test (%)":     100 * len(tst_items - hist_items) / max(len(tst_items), 1),
        "test-only items (%)":       100 * len(tst_items - hist_items - adp_items) / max(len(tst_items), 1),
    }
    names = list(stats.keys())
    vals  = list(stats.values())

    fig, ax = plt.subplots(figsize=(10, 5.5))
    bars = ax.barh(names, vals, color=PALETTE[2], height=0.5)
    for b in bars:
        w = b.get_width()
        ax.text(w + 0.4, b.get_y() + b.get_height() / 2, f"{w:.1f}%",
                va="center", fontsize=11, fontweight="bold")
    ax.set_xlabel("Percentage (%)")
    ax.set_title("Distribution shift relative to historical block", fontweight="bold")
    ax.set_xlim(0, max(vals) * 1.20 if vals else 10)
    fig.tight_layout()
    fig.savefig(outdir / "distribution_shift_summary.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)



def plot_new_item_activation(new_item_activation: pd.DataFrame, adapt_end, outdir: Path):
    fig, axes = plt.subplots(2, 1, figsize=(13, 8), sharex=True)
    c = PALETTE[3]
    axes[0].plot(new_item_activation["period"],
                 100 * new_item_activation["new_item_interaction_share"],
                 linewidth=2.2, color=c)
    axes[1].plot(new_item_activation["period"],
                 100 * new_item_activation["new_item_active_share"],
                 linewidth=2.2, color=c)
    for ax in axes:
        ax.axvline(adapt_end, color="#333333", linestyle=":", linewidth=1.5)
        ax.set_ylabel("Percentage (%)")
    axes[0].set_title("New-item activation over time", fontweight="bold")
    axes[0].set_ylabel("% interactions on new items")
    axes[1].set_ylabel("% active items that are new")
    _fmt_xdate(axes[1], interval=2)
    fig.tight_layout()
    fig.savefig(outdir / "new_item_activation_over_time.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def plot_seq_length_distribution(data: dict, outdir: Path):
    """Box plot of per-user sequence lengths across each split."""
    records = []
    for label, df in data.items():
        if label == "Full":
            continue
        lengths = seq_lengths(df)
        records.append(
            pd.DataFrame({"split": label, "seq_length": lengths.values})
        )
    combined = pd.concat(records, ignore_index=True)

    order  = [k for k in ["Historical", "Future", "Future Adapt", "Future Test"] if k in combined["split"].unique()]
    colors = [COLOR_MAP[k] for k in order]

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.boxplot(data=combined, x="split", y="seq_length", order=order,
                palette=colors, width=0.5, fliersize=2, ax=ax)
    ax.set_yscale("log")
    ax.set_xlabel("")
    ax.set_ylabel("Interactions per user (log scale)")
    ax.set_title("Per-user sequence length distribution", fontweight="bold")
    fig.tight_layout()
    fig.savefig(outdir / "seq_length_distribution.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def plot_user_drift_histogram_raw(user_drift_scores: pd.DataFrame, outdir: Path):
    """Two-metric drift histogram computed from the raw split CSVs."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    for ax, col, label in [
        (axes[0], "cosine_distance",  "Cosine distance\n(item profiles, hist vs future)"),
        (axes[1], "jaccard_distance", "Jaccard distance\n(item sets, hist vs future)"),
    ]:
        vals = user_drift_scores[col].dropna()
        sns.histplot(vals, bins=40, color=PALETTE[2], alpha=0.85, ax=ax, kde=True)
        if len(vals):
            med = vals.median()
            ax.axvline(med, linestyle="--", linewidth=1.8, color="#333333",
                       label=f"median = {med:.3f}")
            ax.legend(fontsize=11)
        ax.set_xlabel(label)
        ax.set_ylabel("# users")

    axes[0].set_title("User drift — cosine distance", fontweight="bold")
    axes[1].set_title("User drift — Jaccard distance", fontweight="bold")
    fig.suptitle("User preference drift (item profiles: hist vs future)", fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(outdir / "user_drift_histogram.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def plot_user_drift_histogram_processed(drift_csv: str, outdir: Path):
    """Three-metric drift histogram from the preprocessing pipeline."""
    df = pd.read_csv(drift_csv)
    metrics = [
        ("item_profile_cosine_distance", "Item-profile\ncosine distance"),
        ("last_hidden_cosine_distance",  "Hidden-state\ncosine distance"),
        ("topk_item_jaccard_drift",      "Top-k item\nJaccard drift"),
    ]
    # keep only metrics present in the file
    metrics = [(c, l) for c, l in metrics if c in df.columns]
    if not metrics:
        return

    n = len(metrics)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5.5))
    if n == 1:
        axes = [axes]

    for ax, (col, xlabel) in zip(axes, metrics):
        vals = df[col].dropna()
        sns.histplot(vals, bins=40, color=PALETTE[2], alpha=0.85, ax=ax, kde=True)
        if len(vals):
            med = vals.median()
            ax.axvline(med, linestyle="--", linewidth=1.8, color="#333333",
                       label=f"median = {med:.3f}")
            ax.legend(fontsize=11)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("# users")

    axes[0].set_title("Item-profile cosine distance", fontweight="bold")
    if len(axes) > 1:
        axes[1].set_title("Hidden-state cosine distance", fontweight="bold")
    if len(axes) > 2:
        axes[2].set_title("Top-k Jaccard drift", fontweight="bold")

    fig.suptitle("User preference drift scores (from preprocessing pipeline)",
                 fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(outdir / "user_drift_histogram.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)

    # also plot combined_drift_score if present
    if "combined_drift_score" in df.columns:
        fig, ax = plt.subplots(figsize=(8, 5.5))
        vals = df["combined_drift_score"].dropna()
        sns.histplot(vals, bins=40, color=PALETTE[3], alpha=0.85, ax=ax, kde=True)
        if len(vals):
            med = vals.median()
            ax.axvline(med, linestyle="--", linewidth=1.8, color="#333333",
                       label=f"median = {med:.3f}")
            ax.legend(fontsize=12)
        ax.set_xlabel("Combined drift score (mean z-score)")
        ax.set_ylabel("# users")
        ax.set_title("Combined drift score distribution\n(high-drift subset after k-core filtering)",
                     fontweight="bold")
        fig.tight_layout()
        fig.savefig(outdir / "combined_drift_score_histogram.png", dpi=DPI, bbox_inches="tight")
        plt.close(fig)


def plot_rank_frequency(rank_freq_curves: pd.DataFrame, outdir: Path):
    fig, ax = plt.subplots(figsize=(9, 6.5))
    for label, color in [
        ("Historical",   COLOR_MAP["Historical"]),
        ("Future Adapt", COLOR_MAP["Future Adapt"]),
        ("Future Test",  COLOR_MAP["Future Test"]),
    ]:
        tmp = rank_freq_curves[rank_freq_curves["label"] == label]
        ax.plot(tmp["rank"], tmp["count"], linewidth=2.2, label=label, color=color)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_title("Rank-frequency curves (Zipf plot)", fontweight="bold")
    ax.set_xlabel("Item popularity rank (log scale)")
    ax.set_ylabel("Interaction count (log scale)")
    ax.legend(fontsize=12)
    fig.tight_layout()
    fig.savefig(outdir / "rank_frequency_curves.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split_dir",     required=True,
                        help="Directory produced by create_adaptation_split.py")
    parser.add_argument("--processed_dir", default=None,
                        help="Optional: output dir of run_pipeline.py — uses precomputed "
                             "drift scores instead of recomputing them")
    parser.add_argument("--full_data",     default=None,
                        help="Optional full dataset CSV (e.g. data/data_csv/30M.csv)")
    parser.add_argument("--outdir",        required=True,
                        help="Output directory for plots and summary JSON")
    parser.add_argument("--topk",          type=int, default=1000,
                        help="Top-k items for popularity concentration plot (default: 1000)")
    args = parser.parse_args()

    split_dir = Path(args.split_dir)
    outdir    = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # ---- load split files ----
    required = {
        "hist":         split_dir / "interactions_hist.csv",
        "future":       split_dir / "interactions_future.csv",
        "future_adapt": split_dir / "interactions_future_adapt.csv",
        "future_test":  split_dir / "interactions_future_test.csv",
    }
    for name, path in required.items():
        if not path.exists():
            raise FileNotFoundError(f"Missing expected file: {path}")

    print("Loading split files...")
    hist         = load_interactions(str(required["hist"]))
    future       = load_interactions(str(required["future"]))
    future_adapt = load_interactions(str(required["future_adapt"]))
    future_test  = load_interactions(str(required["future_test"]))

    data = {
        "Historical":   hist,
        "Future":       future,
        "Future Adapt": future_adapt,
        "Future Test":  future_test,
    }

    # Resolve full-data path: explicit flag > auto-detect 30M.csv two levels up from split_dir
    full_data_path = None
    if args.full_data:
        full_data_path = args.full_data
    else:
        candidate = split_dir.parent.parent / "30M.csv"
        if candidate.exists():
            full_data_path = str(candidate)

    if full_data_path and os.path.exists(full_data_path):
        print(f"Loading full dataset (30M reference): {full_data_path}")
        data = {"Full": load_interactions(full_data_path), **data}
    else:
        print("WARNING: full dataset (30M.csv) not found — temporal plots will show splits only. "
              "Pass --full_data to include it.")

    # ---- derived dataframes ----
    print("Computing derived statistics...")
    new_item_activation  = new_item_activation_df(hist, future, freq="W")
    rank_freq_curves     = pd.concat([
        rank_frequency_df(hist,         "Historical"),
        rank_frequency_df(future_adapt, "Future Adapt"),
        rank_frequency_df(future_test,  "Future Test"),
    ], ignore_index=True)

    # drift scores — use precomputed if available
    drift_csv_path = None
    user_drift_scores = None
    if args.processed_dir:
        candidate = Path(args.processed_dir) / "user_drift_scores_final_subset.csv"
        if candidate.exists():
            drift_csv_path = str(candidate)
            print(f"Using precomputed drift scores: {candidate}")
    if drift_csv_path is None:
        print("Recomputing user drift scores from raw splits (this may take a moment)...")
        user_drift_scores = compute_user_drift_scores(hist, future)
        user_drift_scores.to_csv(outdir / "user_drift_scores.csv", index=False)

    # ---- temporal boundaries ----
    hist_end  = hist["dt"].max()
    adapt_end = future_adapt["dt"].max()

    # ---- summary JSON ----
    summary = {
        "row_counts":  {k: int(len(v)) for k, v in data.items()},
        "user_counts": {k: int(v["user"].nunique()) for k, v in data.items()},
        "item_counts": {k: int(v["item"].nunique()) for k, v in data.items()},
        "time_ranges": {
            k: {"start": str(v["dt"].min()), "end": str(v["dt"].max())}
            for k, v in data.items()
        },
        "overlap_hist_future": overlap_stats(hist, future,       "hist", "future"),
        "overlap_hist_adapt":  overlap_stats(hist, future_adapt, "hist", "future_adapt"),
        "overlap_hist_test":   overlap_stats(hist, future_test,  "hist", "future_test"),
        "overlap_adapt_test":  overlap_stats(future_adapt, future_test, "future_adapt", "future_test"),
        "topk_share": {
            f"top_{args.topk}_hist_in_hist":         top_k_share(hist, hist,         args.topk),
            f"top_{args.topk}_hist_in_future_adapt": top_k_share(hist, future_adapt, args.topk),
            f"top_{args.topk}_hist_in_future_test":  top_k_share(hist, future_test,  args.topk),
        },
        "new_item_activation_summary": {
            "mean_new_item_interaction_share": float(new_item_activation["new_item_interaction_share"].mean()),
            "max_new_item_interaction_share":  float(new_item_activation["new_item_interaction_share"].max()),
            "mean_new_item_active_share":      float(new_item_activation["new_item_active_share"].mean()),
        } if len(new_item_activation) else {},
    }
    with open(outdir / "split_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # save derived CSVs
    new_item_activation.to_csv(outdir / "new_item_activation_over_time.csv", index=False)

    # ---- plots ----
    print("Generating plots...")

    plot_temporal_ranges(data, hist_end, adapt_end, outdir)
    print("  temporal_ranges.png")

    plot_temporal_density(data, hist_end, adapt_end, outdir)
    print("  temporal_density.png")

    plot_top_items_share(hist, future_adapt, future_test, args.topk, outdir)
    print("  top_items_share.png")

    plot_distribution_shift(hist, future_adapt, future_test, outdir)
    print("  distribution_shift_summary.png")

    plot_new_item_activation(new_item_activation, adapt_end, outdir)
    print("  new_item_activation_over_time.png")

    plot_seq_length_distribution(data, outdir)
    print("  seq_length_distribution.png")

    if drift_csv_path:
        plot_user_drift_histogram_processed(drift_csv_path, outdir)
        print("  user_drift_histogram.png  (+ combined_drift_score_histogram.png)")
    elif user_drift_scores is not None:
        plot_user_drift_histogram_raw(user_drift_scores, outdir)
        print("  user_drift_histogram.png")

    plot_rank_frequency(rank_freq_curves, outdir)
    print("  rank_frequency_curves.png")

    print(f"\nAll outputs saved to: {outdir}/")


if __name__ == "__main__":
    main()
