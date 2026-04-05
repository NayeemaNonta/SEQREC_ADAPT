#!/usr/bin/env python3
"""
adaptation/prototype_steering/eval.py

Evaluate prototype-steering adapter vs frozen baseline.
Identical candidate sets per user (apples-to-apples).

Usage:
python adaptation/prototype_steering/eval.py \
  --checkpoint       results/backbone/sasrec_backbone_best.pt \
  --adapt_checkpoint results/prototype_steering/prototype_best.pt \
  --cluster_csv      data/processed/user_clusters.csv \
  --test_data        data/processed/future_test.csv \
  --outdir           results/prototype_steering/eval
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from backbone.model import SASRec
from adaptation.prototype_steering.train import PrototypeAdapter
from common.data_utils import (
    read_interactions, load_checkpoint, make_encoders_from_ckpt,
    encode_df, build_sequences_by_user, leave_one_out,
    build_input_tensor, sample_negatives,
)
from common.metrics.ranking import metrics_from_rank, summarize, compute_deltas, print_delta_report
from common.logging.logger import ExperimentLogger

import random


def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint",       required=True)
    p.add_argument("--adapt_checkpoint", required=True)
    p.add_argument("--cluster_csv",      required=True)
    p.add_argument("--test_data",        required=True)
    p.add_argument("--outdir",           required=True)
    p.add_argument("--num_neg_eval", type=int, default=100)
    p.add_argument("--seed",         type=int, default=42)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    logger = ExperimentLogger(args.outdir)

    base_ckpt  = load_checkpoint(args.checkpoint,       args.device)
    adapt_ckpt = load_checkpoint(args.adapt_checkpoint, args.device)
    le_user, le_item = make_encoders_from_ckpt(base_ckpt)
    cfg = base_ckpt["config"]

    backbone = SASRec(
        item_num=base_ckpt["itemnum"], maxlen=cfg["maxlen"],
        hidden_units=cfg["hidden_units"], num_blocks=cfg["num_blocks"],
        num_heads=cfg["num_heads"], dropout_rate=cfg["dropout_rate"],
        add_head=False, pos_enc=True,
    ).to(args.device)
    backbone.load_state_dict(base_ckpt["model_state_dict"])
    backbone.eval()

    adapter = PrototypeAdapter(
        adapt_ckpt["num_clusters"], cfg["hidden_units"], adapt_ckpt["bottleneck_dim"]
    ).to(args.device)
    adapter.load_state_dict(adapt_ckpt["adapter_state_dict"])
    adapter.eval()

    # cluster map
    cdf = pd.read_csv(args.cluster_csv)
    cdf["user_id"] = cdf["user_id"].astype(str)
    known = set(le_user.classes_.tolist())
    cdf   = cdf[cdf["user_id"].isin(known)]
    cluster_map = dict(zip(le_user.transform(cdf["user_id"]), cdf["cluster_id"].astype(int)))

    test_seqs = build_sequences_by_user(
        encode_df(read_interactions(args.test_data), le_user, le_item)
    )
    ctx, tgt, skipped = leave_one_out(test_seqs)
    users = sorted(set(ctx) & set(tgt) & set(cluster_map))
    print(f"[eval] users (with cluster): {len(users):,}  skipped (seq<2): {skipped}")

    base_rows, adpt_rows = [], []
    maxlen  = cfg["maxlen"]
    itemnum = base_ckpt["itemnum"]

    with torch.no_grad():
        for user in users:
            context    = ctx[user]
            target     = tgt[user]
            cluster_id = cluster_map[user]
            seen       = set(context) | {0}
            negs       = sample_negatives(itemnum, seen, target, n_neg=args.num_neg_eval)
            cands      = [target] + negs

            input_ids     = build_input_tensor(context, maxlen, args.device)
            candidate_ids = torch.tensor([cands], dtype=torch.long, device=args.device)
            cid_t         = torch.tensor([cluster_id], dtype=torch.long, device=args.device)

            h_base = backbone.get_last_hidden(input_ids)
            base_logits = backbone.score_from_hidden(h_base, candidate_ids)[0]
            base_rank   = int((base_logits > base_logits[0]).sum().item())

            h_adpt = adapter(h_base, cid_t)
            adpt_logits = backbone.score_from_hidden(h_adpt, candidate_ids)[0]
            adpt_rank   = int((adpt_logits > adpt_logits[0]).sum().item())

            for rank, rows in [(base_rank, base_rows), (adpt_rank, adpt_rows)]:
                rows.append({"user_idx": int(user), "cluster_id": cluster_id,
                              "rank": rank, **metrics_from_rank(rank)})

    base_df = pd.DataFrame(base_rows)
    adpt_df = pd.DataFrame(adpt_rows)
    base_m  = summarize(base_df)
    adpt_m  = summarize(adpt_df)
    deltas  = compute_deltas(base_m, adpt_m)

    outdir = Path(args.outdir)
    base_df.to_csv(outdir / "eval_rows_baseline.csv", index=False)
    adpt_df.to_csv(outdir / "eval_rows_adapted.csv",  index=False)
    merged = base_df.merge(adpt_df, on=["user_idx", "cluster_id"], suffixes=("_baseline", "_adapted"))
    merged["rank_delta"] = merged["rank_adapted"] - merged["rank_baseline"]
    merged.to_csv(outdir / "eval_rows_merged.csv", index=False)

    logger.save_summary({
        "checkpoint": args.checkpoint, "adapt_checkpoint": args.adapt_checkpoint,
        "eval": {"n_users": len(users), "skipped_lt2": skipped,
                 "baseline_metrics": base_m, "adapted_metrics": adpt_m, "deltas": deltas},
    })
    print_delta_report("PROTOTYPE STEERING — FUTURE TEST", deltas)
    print(f"\nSaved to {args.outdir}/")


if __name__ == "__main__":
    main()
