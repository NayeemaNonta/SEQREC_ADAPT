#!/usr/bin/env python3
"""
detect_high_drift_users_overlap.py

Detect high-drift users within the overlap-item regime.

Assumes:
  - input files already contain only overlap items
  - checkpoint was produced by train_sasrec_t1.py

Drift signals:
  1. item_profile_cosine_distance
  2. last_hidden_cosine_distance
  3. topk_item_jaccard_drift

Outputs:
  outdir/user_drift_scores_overlap.csv
  outdir/high_drift_users_top_overlap.csv
  outdir/drift_summary_overlap.json

Usage:
python detect_high_drift_users_overlap.py \
  --checkpoint results/overlap_item_subset_kcore/t1_backbone_overlap/sasrec_t1_best.pt \
  --hist_data results/overlap_item_subset_kcore/hist_overlap_items_kcore.csv \
  --future_data results/overlap_item_subset_kcore/future_adapt_overlap_items_kcore.csv \
  --outdir results/overlap_item_subset_kcore/user_drift_overlap \
  --top_pct 0.1 \
  --device cuda
"""

import argparse
import json
import random
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
from sklearn.metrics.pairwise import cosine_distances
from sklearn.preprocessing import LabelEncoder, StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from backbone.model import SASRec


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


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
        raise ValueError(f"{path} missing columns: {missing}")

    return df[list(required)].dropna().copy()


def load_checkpoint(path: str, device: str):
    return torch.load(path, map_location=device, weights_only=False)


def build_args_from_ckpt(ckpt, device):
    cfg = ckpt["config"]
    return SimpleNamespace(
        hidden_units=cfg["hidden_units"],
        maxlen=cfg["maxlen"],
        dropout_rate=cfg["dropout_rate"],
        num_blocks=cfg["num_blocks"],
        num_heads=cfg["num_heads"],
        device=device,
    )


def load_model_from_checkpoint(ckpt, device: str):
    args = build_args_from_ckpt(ckpt, device)
    model = SASRec(
        item_num=ckpt["itemnum"],
        maxlen=args.maxlen,
        hidden_units=args.hidden_units,
        num_blocks=args.num_blocks,
        num_heads=args.num_heads,
        dropout_rate=args.dropout_rate,
        add_head=False,
        pos_enc=True,
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, args


def make_encoders_from_ckpt(ckpt):
    le_item = LabelEncoder()
    le_item.classes_ = np.array(ckpt["le_item_classes"], dtype=object)

    le_user = LabelEncoder()
    le_user.classes_ = np.array(ckpt["le_user_classes"], dtype=object)

    return le_user, le_item


def encode_overlap(df: pd.DataFrame, le_user: LabelEncoder, le_item: LabelEncoder) -> pd.DataFrame:
    known_users = set(le_user.classes_.tolist())
    known_items = set(le_item.classes_.tolist())

    out = df.copy()
    out["user_id"] = out["user_id"].astype(str)
    out["item_id"] = out["item_id"].astype(str)

    out = out[out["user_id"].isin(known_users) & out["item_id"].isin(known_items)].copy()
    out["user_idx"] = le_user.transform(out["user_id"])          # 0-based
    out["item_idx"] = le_item.transform(out["item_id"]) + 1      # 1-based for model

    return out.sort_values(["user_idx", "timestamp"]).reset_index(drop=True)


def build_input_tensor(seq, maxlen, device):
    seq = seq[-maxlen:]
    pad = maxlen - len(seq)
    return torch.tensor([[0] * pad + seq], dtype=torch.long, device=device)


@torch.no_grad()
def get_last_hidden(model, input_ids: torch.Tensor):
    return model.get_last_hidden(input_ids)


def user_item_count_dict(seq):
    counts = {}
    for x in seq:
        counts[x] = counts.get(x, 0) + 1
    return counts


def cosine_distance_from_count_dicts(a_counts, b_counts):
    keys = sorted(set(a_counts.keys()) | set(b_counts.keys()))
    if len(keys) == 0:
        return 0.0

    a = np.array([a_counts.get(k, 0) for k in keys], dtype=np.float64).reshape(1, -1)
    b = np.array([b_counts.get(k, 0) for k in keys], dtype=np.float64).reshape(1, -1)

    if np.all(a == 0) or np.all(b == 0):
        return 0.0

    return float(cosine_distances(a, b)[0, 0])


def topk_jaccard_drift(a_counts, b_counts, k=10):
    a_top = set([x for x, _ in sorted(a_counts.items(), key=lambda z: (-z[1], z[0]))[:k]])
    b_top = set([x for x, _ in sorted(b_counts.items(), key=lambda z: (-z[1], z[0]))[:k]])

    union = a_top | b_top
    inter = a_top & b_top

    if len(union) == 0:
        return 0.0

    jacc = len(inter) / len(union)
    return float(1.0 - jacc)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--hist_data", type=str, required=True)
    parser.add_argument("--future_data", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--top_pct", type=float, default=0.1)
    parser.add_argument("--topk_items", type=int, default=10)
    parser.add_argument("--min_hist_len", type=int, default=3)
    parser.add_argument("--min_future_len", type=int, default=3)
    parser.add_argument("--outdir", type=str, required=True)
    args = parser.parse_args()

    set_seed(args.seed)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    print(f"[setup] device={args.device}")
    print(f"[setup] loading checkpoint: {args.checkpoint}")
    ckpt = load_checkpoint(args.checkpoint, args.device)
    model, model_args = load_model_from_checkpoint(ckpt, args.device)
    le_user, le_item = make_encoders_from_ckpt(ckpt)

    print(f"[setup] model loaded (itemnum={ckpt['itemnum']}, hidden={model_args.hidden_units})")

    print(f"[data] reading hist: {args.hist_data}")
    hist = read_interactions(args.hist_data)
    print(f"[data] reading future: {args.future_data}")
    future = read_interactions(args.future_data)

    hist_enc = encode_overlap(hist, le_user, le_item)
    future_enc = encode_overlap(future, le_user, le_item)

    hist_seqs = hist_enc.groupby("user_idx")["item_idx"].agg(list).to_dict()
    future_seqs = future_enc.groupby("user_idx")["item_idx"].agg(list).to_dict()

    overlap_users = sorted(set(hist_seqs.keys()) & set(future_seqs.keys()))
    print(f"[data] {len(overlap_users)} overlap users (hist={len(hist_seqs)}, future={len(future_seqs)})")
    print(f"[drift] computing overlap-item drift scores (min_hist={args.min_hist_len}, min_future={args.min_future_len})")

    rows = []
    filtered_out = 0

    for u in overlap_users:
        hist_seq = hist_seqs[u]
        fut_seq = future_seqs[u]

        if len(hist_seq) < args.min_hist_len or len(fut_seq) < args.min_future_len:
            filtered_out += 1
            continue

        hist_counts = user_item_count_dict(hist_seq)
        fut_counts = user_item_count_dict(fut_seq)

        item_profile_drift = cosine_distance_from_count_dicts(hist_counts, fut_counts)
        topk_drift = topk_jaccard_drift(hist_counts, fut_counts, k=args.topk_items)

        hist_input = build_input_tensor(hist_seq, model_args.maxlen, args.device)
        fut_input = build_input_tensor(fut_seq, model_args.maxlen, args.device)

        with torch.no_grad():
            h_hist = get_last_hidden(model, hist_input)[0].detach().cpu().numpy().reshape(1, -1)
            h_fut = get_last_hidden(model, fut_input)[0].detach().cpu().numpy().reshape(1, -1)

        hidden_drift = float(cosine_distances(h_hist, h_fut)[0, 0])

        rows.append(
            {
                "user_idx": int(u),
                "user_id": le_user.classes_[u],
                "hist_len": int(len(hist_seq)),
                "future_len": int(len(fut_seq)),
                "item_profile_cosine_distance": float(item_profile_drift),
                "last_hidden_cosine_distance": float(hidden_drift),
                "topk_item_jaccard_drift": float(topk_drift),
            }
        )

    drift_df = pd.DataFrame(rows)
    if len(drift_df) == 0:
        raise ValueError("No overlap users survived the minimum length filters.")

    print(f"[drift] {len(drift_df)} users scored ({filtered_out} filtered by min length)")

    score_cols = [
        "item_profile_cosine_distance",
        "last_hidden_cosine_distance",
        "topk_item_jaccard_drift",
    ]

    scaler = StandardScaler()
    Z = scaler.fit_transform(drift_df[score_cols].values)
    drift_df["combined_drift_score"] = Z.mean(axis=1)

    drift_df = drift_df.sort_values("combined_drift_score", ascending=False).reset_index(drop=True)
    n_keep = max(1, int(len(drift_df) * args.top_pct))
    high_drift_df = drift_df.head(n_keep).copy()

    drift_df.to_csv(outdir / "user_drift_scores_overlap.csv", index=False)
    high_drift_df.to_csv(outdir / "high_drift_users_top_overlap.csv", index=False)

    summary = {
        "checkpoint": args.checkpoint,
        "hist_data": args.hist_data,
        "future_data": args.future_data,
        "n_overlap_users_total": int(len(overlap_users)),
        "n_users_scored": int(len(drift_df)),
        "top_pct": float(args.top_pct),
        "n_users_selected": int(len(high_drift_df)),
        "min_hist_len": int(args.min_hist_len),
        "min_future_len": int(args.min_future_len),
        "score_columns": score_cols,
        "combined_score": "mean z-score across item-profile drift, hidden-state drift, and top-k item drift within overlap-item regime",
        "top10_preview_user_ids": high_drift_df["user_id"].head(10).astype(str).tolist(),
    }

    with open(outdir / "drift_summary_overlap.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))
    print("Saved:")
    print(f"  {outdir / 'user_drift_scores_overlap.csv'}")
    print(f"  {outdir / 'high_drift_users_top_overlap.csv'}")
    print(f"  {outdir / 'drift_summary_overlap.json'}")


if __name__ == "__main__":
    main()