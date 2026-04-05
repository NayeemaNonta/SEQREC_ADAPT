#!/usr/bin/env python3
"""
adaptation/last_block/eval.py

Evaluate last-block fine-tuned model vs frozen baseline.
Identical candidate sets per user (apples-to-apples).

Usage:
python adaptation/last_block/eval.py \
  --checkpoint    results/backbone/sasrec_backbone_best.pt \
  --ft_checkpoint results/last_block/last_block_best.pt \
  --test_data     data/processed/future_test.csv \
  --outdir        results/last_block/eval
"""

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from backbone.model import SASRec
from common.data_utils import (
    read_interactions, load_checkpoint, make_encoders_from_ckpt,
    encode_df, build_sequences_by_user, leave_one_out,
)
from common.evaluation.evaluator import build_eval_examples, evaluate_paired
from common.metrics.ranking import compute_deltas, print_delta_report
from common.logging.logger import ExperimentLogger

import random
import numpy as np


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_model(ckpt: dict, device) -> SASRec:
    cfg = ckpt["config"]
    model = SASRec(
        item_num=ckpt["itemnum"], maxlen=cfg["maxlen"],
        hidden_units=cfg["hidden_units"], num_blocks=cfg["num_blocks"],
        num_heads=cfg["num_heads"], dropout_rate=cfg["dropout_rate"],
        add_head=False, pos_enc=True,
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint",    required=True)
    p.add_argument("--ft_checkpoint", required=True)
    p.add_argument("--test_data",     required=True)
    p.add_argument("--outdir",        required=True)
    p.add_argument("--num_neg_eval",  type=int, default=100)
    p.add_argument("--seed",          type=int, default=42)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    logger = ExperimentLogger(args.outdir)

    base_ckpt = load_checkpoint(args.checkpoint,    args.device)
    ft_ckpt   = load_checkpoint(args.ft_checkpoint, args.device)
    le_user, le_item = make_encoders_from_ckpt(base_ckpt)

    baseline  = load_model(base_ckpt, args.device)
    finetuned = load_model(ft_ckpt,   args.device)

    test_df   = read_interactions(args.test_data)
    test_enc  = encode_df(test_df, le_user, le_item)
    test_seqs = build_sequences_by_user(test_enc)
    ctx, tgt, skipped = leave_one_out(test_seqs)

    print(f"[eval] users: {len(ctx):,}  skipped (seq<2): {skipped}")

    examples = build_eval_examples(ctx, tgt, base_ckpt["itemnum"], n_neg=args.num_neg_eval)

    base_df, base_m, ft_df, ft_m = evaluate_paired(
        baseline_scorer=baseline.get_last_hidden,
        adapted_scorer=finetuned.get_last_hidden,
        backbone=baseline,
        examples=examples,
        maxlen=base_ckpt["config"]["maxlen"],
        device=args.device,
    )

    deltas = compute_deltas(base_m, ft_m)

    merged = base_df.merge(ft_df, on="user_idx", suffixes=("_baseline", "_finetuned"))
    merged["rank_delta"] = merged["rank_finetuned"] - merged["rank_baseline"]

    outdir = Path(args.outdir)
    base_df.to_csv(outdir / "eval_rows_baseline.csv",  index=False)
    ft_df.to_csv(outdir   / "eval_rows_finetuned.csv", index=False)
    merged.to_csv(outdir  / "eval_rows_merged.csv",    index=False)

    summary = {
        "checkpoint": args.checkpoint, "ft_checkpoint": args.ft_checkpoint,
        "test_data":  args.test_data,
        "eval": {
            "n_users": len(examples), "skipped_lt2": skipped,
            "baseline_metrics": base_m, "finetuned_metrics": ft_m, "deltas": deltas,
        },
    }
    logger.save_summary(summary)
    print_delta_report("LAST BLOCK FT — FUTURE TEST", deltas)
    print(f"\nSaved to {args.outdir}/")


if __name__ == "__main__":
    main()
