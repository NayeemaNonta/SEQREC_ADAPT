#!/usr/bin/env python3
"""
adaptation/context_steering/eval.py

Evaluate context-gate adapter vs frozen baseline.
Identical candidate sets per user (apples-to-apples).

Usage:
python adaptation/context_steering/eval.py \
  --checkpoint      results/backbone/sasrec_backbone_best.pt \
  --adapt_checkpoint results/context_gate/context_gate_best.pt \
  --test_data       data/processed/future_test.csv \
  --outdir          results/context_gate/eval
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from backbone.model import SASRec, SASRecContextGateModel, ContextGateAdapter
from common.data_utils import (
    read_interactions, load_checkpoint, make_encoders_from_ckpt,
    encode_df, build_sequences_by_user, leave_one_out,
)
from common.evaluation.evaluator import build_eval_examples, evaluate_paired
from common.metrics.ranking import compute_deltas, print_delta_report
from common.logging.logger import ExperimentLogger

import random


def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)


def load_baseline(ckpt, device):
    cfg = ckpt["config"]
    m = SASRec(
        item_num=ckpt["itemnum"], maxlen=cfg["maxlen"],
        hidden_units=cfg["hidden_units"], num_blocks=cfg["num_blocks"],
        num_heads=cfg["num_heads"], dropout_rate=cfg["dropout_rate"],
        add_head=False, pos_enc=True,
    ).to(device)
    m.load_state_dict(ckpt["model_state_dict"])
    m.eval()
    return m


def load_adapted(backbone_ckpt, adapt_ckpt, device):
    backbone = load_baseline(backbone_ckpt, device)
    ack = adapt_ckpt
    adapted = SASRecContextGateModel(
        backbone=backbone,
        bottleneck_dim=ack["bottleneck_dim"],
        adapter_dropout=ack.get("adapter_dropout", 0.0),
        adapter_activation=ack.get("adapter_activation", "gelu"),
        freeze_backbone=True,
    ).to(device)
    adapted.adapter.load_state_dict(ack["adapter_state_dict"])
    adapted.eval()
    return adapted


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint",       required=True)
    p.add_argument("--adapt_checkpoint", required=True)
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

    baseline = load_baseline(base_ckpt, args.device)
    adapted  = load_adapted(base_ckpt, adapt_ckpt, args.device)

    test_seqs = build_sequences_by_user(
        encode_df(read_interactions(args.test_data), le_user, le_item)
    )
    ctx, tgt, skipped = leave_one_out(test_seqs)
    print(f"[eval] users: {len(ctx):,}  skipped (seq<2): {skipped}")

    examples = build_eval_examples(ctx, tgt, base_ckpt["itemnum"], n_neg=args.num_neg_eval)

    base_df, base_m, adpt_df, adpt_m = evaluate_paired(
        baseline_scorer=baseline.get_last_hidden,
        adapted_scorer=adapted.get_adapted_hidden,
        backbone=baseline,
        examples=examples,
        maxlen=base_ckpt["config"]["maxlen"],
        device=args.device,
    )

    deltas = compute_deltas(base_m, adpt_m)
    outdir = Path(args.outdir)
    base_df.to_csv(outdir / "eval_rows_baseline.csv", index=False)
    adpt_df.to_csv(outdir / "eval_rows_adapted.csv",  index=False)
    merged = base_df.merge(adpt_df, on="user_idx", suffixes=("_baseline", "_adapted"))
    merged["rank_delta"] = merged["rank_adapted"] - merged["rank_baseline"]
    merged.to_csv(outdir / "eval_rows_merged.csv", index=False)

    logger.save_summary({
        "checkpoint": args.checkpoint, "adapt_checkpoint": args.adapt_checkpoint,
        "eval": {"n_users": len(examples), "skipped_lt2": skipped,
                 "baseline_metrics": base_m, "adapted_metrics": adpt_m, "deltas": deltas},
    })
    print_delta_report("CONTEXT GATE — FUTURE TEST", deltas)
    print(f"\nSaved to {args.outdir}/")


if __name__ == "__main__":
    main()
