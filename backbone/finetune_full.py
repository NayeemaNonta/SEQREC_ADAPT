#!/usr/bin/env python3
"""
backbone/finetune_full.py

Full backbone fine-tuning on future_adapt data — upper-bound baseline.
All backbone parameters are updated (no freezing).
Uses the same full-vocab cross-entropy loss as context_steering for fair comparison.

Outputs:
  <output_dir>/full_ft_best.pt       — fine-tuned backbone checkpoint
  <output_dir>/summary.json          — eval results in standard format (baseline vs finetuned)
  <output_dir>/training_log.json     — per-epoch loss curve

Usage:
python backbone/finetune_full.py \
  --checkpoint  results/backbone/sasrec_backbone_best.pt \
  --adapt_data  data/processed/split_10M_contiguous/future_adapt_high_drift_kcore.csv \
  --test_data   data/processed/split_10M_contiguous/future_test_high_drift_kcore.csv \
  --output_dir  results/full_ft_contiguous --device cuda
"""

import argparse
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backbone.model import SASRec
from common.data_utils import (
    read_interactions, load_checkpoint, make_encoders_from_ckpt,
    encode_df, build_sequences_by_user, pad_sequence,
    leave_one_out, build_input_tensor, sample_negatives,
)
from common.metrics.ranking import metrics_from_rank, summarize, compute_deltas, print_delta_report
from common.memory_profiler.profiler import MemoryProfiler, Timer
from common.logging.logger import ExperimentLogger


def set_seed(seed: int):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Dataset — same as context_steering (full-vocab CE, no negative sampling)
# ---------------------------------------------------------------------------

class AdaptDataset(Dataset):
    def __init__(self, user_seqs: dict, maxlen: int):
        self.maxlen = maxlen
        self.examples = []
        for seq in user_seqs.values():
            for t in range(1, len(seq)):
                self.examples.append((seq[:t], seq[t] - 1))   # target 0-indexed

    def __len__(self): return len(self.examples)

    def __getitem__(self, idx):
        prefix, target = self.examples[idx]
        return {
            "input_ids": torch.tensor(pad_sequence(prefix, self.maxlen), dtype=torch.long),
            "target":    torch.tensor(target, dtype=torch.long),
        }


def ce_loss(model, input_ids, target):
    logits = model.score_from_hidden(model.get_last_hidden(input_ids))[:, 1:]  # drop pad
    return F.cross_entropy(logits, target)


@torch.no_grad()
def eval_ce(model, loader, device):
    model.eval()
    total, steps = 0.0, 0
    for b in loader:
        total += ce_loss(model, b["input_ids"].to(device), b["target"].to(device)).item()
        steps += 1
    model.train()
    return total / max(steps, 1)


# ---------------------------------------------------------------------------
# Ranking eval — identical protocol to all other methods
# ---------------------------------------------------------------------------

@torch.no_grad()
def ranking_eval(model, test_seqs, le_user, le_item, cfg, itemnum, device, num_neg):
    model.eval()
    ctx, tgt, skipped = leave_one_out(test_seqs)
    users = sorted(set(ctx) & set(tgt))
    maxlen = cfg["maxlen"]
    rows = []
    for user in users:
        seen   = set(ctx[user]) | {0}
        negs   = sample_negatives(itemnum, seen, tgt[user], n_neg=num_neg)
        cands  = [tgt[user]] + negs
        ids    = build_input_tensor(ctx[user], maxlen, device)
        cand_t = torch.tensor([cands], dtype=torch.long, device=device)
        logits = model.score_from_hidden(model.get_last_hidden(ids), cand_t)[0]
        rank   = int((logits > logits[0]).sum().item())
        rows.append({"user_idx": int(user), "rank": rank, **metrics_from_rank(rank)})
    print(f"[full_ft] eval: {len(users):,} users  skipped (seq<2): {skipped}")
    return summarize(rows), rows


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint",    required=True,
                   help="Pre-trained backbone checkpoint (starting point)")
    p.add_argument("--adapt_data",    required=True)
    p.add_argument("--test_data",     required=True)
    p.add_argument("--output_dir",    required=True)
    p.add_argument("--device",        default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--lr",            type=float, default=1e-4,
                   help="Lower than adapter LRs — all params, risk of forgetting")
    p.add_argument("--weight_decay",  type=float, default=1e-4)
    p.add_argument("--epochs",        type=int,   default=20)
    p.add_argument("--batch_size",    type=int,   default=256)
    p.add_argument("--eval_batch_size", type=int, default=512)
    p.add_argument("--num_neg_eval",  type=int,   default=100)
    p.add_argument("--seed",          type=int,   default=42)
    return p.parse_args()


def main():
    args   = parse_args()
    set_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    logger = ExperimentLogger(args.output_dir)

    ckpt             = load_checkpoint(args.checkpoint, args.device)
    le_user, le_item = make_encoders_from_ckpt(ckpt)
    cfg              = ckpt["config"]
    itemnum          = ckpt["itemnum"]

    # ---- build model — ALL parameters trainable ----
    model = SASRec(
        item_num=itemnum, maxlen=cfg["maxlen"],
        hidden_units=cfg["hidden_units"], num_blocks=cfg["num_blocks"],
        num_heads=cfg["num_heads"], dropout_rate=cfg["dropout_rate"],
        add_head=False, pos_enc=True,
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])

    total_params    = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[full_ft] trainable params: {trainable_params:,} / {total_params:,} (all unfrozen)")

    # ---- data ----
    adapt_enc  = encode_df(read_interactions(args.adapt_data), le_user, le_item)
    adapt_seqs = build_sequences_by_user(adapt_enc)
    maxlen     = cfg["maxlen"]

    train_ds    = AdaptDataset(adapt_seqs, maxlen)
    loader      = DataLoader(train_ds, batch_size=args.batch_size,      shuffle=True,  num_workers=0, pin_memory=True)
    eval_loader = DataLoader(train_ds, batch_size=args.eval_batch_size, shuffle=False, num_workers=0, pin_memory=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # ---- baseline ranking eval (frozen backbone) ----
    print("[full_ft] computing baseline metrics (frozen backbone) ...")
    test_enc  = encode_df(read_interactions(args.test_data), le_user, le_item)
    test_seqs = build_sequences_by_user(test_enc)
    baseline_metrics, _ = ranking_eval(model, test_seqs, le_user, le_item,
                                       cfg, itemnum, device, args.num_neg_eval)

    # ---- training ----
    mem = MemoryProfiler(); mem.reset()
    timer = Timer(); timer.start()

    best_loss, best_state = float("inf"), None
    ckpt_path = Path(args.output_dir) / "full_ft_best.pt"

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()
        running, steps = 0.0, 0
        pbar = tqdm(loader, desc=f"Epoch {epoch:03d}", leave=False, dynamic_ncols=True)
        for batch in pbar:
            optimizer.zero_grad()
            loss = ce_loss(model, batch["input_ids"].to(device), batch["target"].to(device))
            loss.backward(); optimizer.step()
            running += loss.item(); steps += 1
            pbar.set_postfix(loss=f"{running/steps:.4f}")

        avg_train = running / max(steps, 1)
        avg_eval  = eval_ce(model, eval_loader, device)
        logger.log_epoch({"epoch": epoch, "train_loss": float(avg_train),
                          "eval_loss": float(avg_eval), "epoch_time_s": round(time.time()-t0, 2)})
        print(f"Epoch {epoch:03d} | train_loss={avg_train:.4f} | eval_loss={avg_eval:.4f} | time={time.time()-t0:.1f}s")

        if avg_eval < best_loss:
            best_loss  = avg_eval
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            print(f"[full_ft] new best at epoch {epoch} (eval_loss={best_loss:.4f})")

    wall_time = timer.elapsed()
    peak_mem  = mem.peak_mb()
    print(f"[full_ft] total wall time: {wall_time:.1f}s  peak GPU mem: {peak_mem:.1f} MB")
    logger.save_history()

    # ---- save best checkpoint ----
    model.load_state_dict(best_state)
    torch.save({
        "model_state_dict": best_state,
        "config":           cfg,
        "itemnum":          int(itemnum),
        "le_user_classes":  ckpt["le_user_classes"],
        "le_item_classes":  ckpt["le_item_classes"],
        "best_eval_loss":   float(best_loss),
        "trainable_params": int(trainable_params),
    }, ckpt_path)
    print(f"[full_ft] saved to {ckpt_path}")

    # ---- finetuned ranking eval ----
    print("[full_ft] computing finetuned metrics ...")
    finetuned_metrics, _ = ranking_eval(model, test_seqs, le_user, le_item,
                                        cfg, itemnum, device, args.num_neg_eval)
    deltas = compute_deltas(baseline_metrics, finetuned_metrics)
    print_delta_report("FULL BACKBONE FT — FUTURE TEST", deltas)

    logger.save_summary({
        "checkpoint":       args.checkpoint,
        "output_dir":       args.output_dir,
        "trainable_params": int(trainable_params),
        "total_params":     int(total_params),
        "wall_time_s":      round(wall_time, 2),
        "peak_gpu_mb":      round(peak_mem, 1),
        "eval": {
            "n_users":            len(test_seqs),
            "baseline_metrics":   baseline_metrics,
            "finetuned_metrics":  finetuned_metrics,
            "deltas":             deltas,
        },
    })
    print(f"\nSaved to {args.output_dir}/")


if __name__ == "__main__":
    main()
