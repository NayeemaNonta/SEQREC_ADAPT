#!/usr/bin/env python3
"""
adaptation/context_steering/train.py

Train context-conditioned scalar gate adapter on future_adapt.
Backbone is fully frozen; only the adapter parameters are updated.

  h̃ = h + σ(w_gate · h + b_gate) · f_ϕ(h)

Loss: cross-entropy over full item vocabulary.

Usage:
python adaptation/context_steering/train.py \
  --checkpoint  results/backbone/sasrec_backbone_best.pt \
  --adapt_data  data/processed/future_adapt.csv \
  --output_dir  results/context_gate \
  --device cuda --bottleneck_dim 8 --epochs 15 --lr 5e-4
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from backbone.model import SASRec, SASRecContextGateModel
from common.data_utils import (
    read_interactions, load_checkpoint, make_encoders_from_ckpt,
    encode_df, build_sequences_by_user, pad_sequence,
)
from common.memory_profiler.profiler import MemoryProfiler, Timer
from common.logging.logger import ExperimentLogger

import random  # kept for set_seed


def set_seed(seed: int):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


class AdaptDataset(Dataset):
    """One (input_ids, target) pair per (user, position) — no negative sampling."""

    def __init__(self, user_seqs: dict, maxlen: int):
        self.maxlen = maxlen
        self.examples = []
        for seq in user_seqs.values():
            for t in range(1, len(seq)):
                self.examples.append((seq[:t], seq[t] - 1))  # target 0-indexed

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        prefix, target = self.examples[idx]
        return {
            "input_ids": torch.tensor(pad_sequence(prefix, self.maxlen), dtype=torch.long),
            "target":    torch.tensor(target, dtype=torch.long),
        }


def ce_loss(model, input_ids, target):
    h = model.get_adapted_hidden(input_ids)
    logits = model.backbone.score_from_hidden(h)[:, 1:]  # drop padding idx → (B, itemnum)
    return F.cross_entropy(logits, target)


@torch.no_grad()
def eval_ce(model, loader, device):
    model.eval()
    total, steps = 0.0, 0
    for batch in loader:
        total += ce_loss(
            model, batch["input_ids"].to(device), batch["target"].to(device),
        ).item()
        steps += 1
    return total / max(steps, 1)


def load_backbone(ckpt, device):
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
    p.add_argument("--checkpoint",          required=True)
    p.add_argument("--adapt_data",          required=True)
    p.add_argument("--output_dir",          required=True)
    p.add_argument("--device",              default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--bottleneck_dim",      type=int,   default=8)
    p.add_argument("--adapter_dropout",     type=float, default=0.0)
    p.add_argument("--adapter_activation",  default="gelu")
    p.add_argument("--lr",                  type=float, default=5e-4)
    p.add_argument("--weight_decay",        type=float, default=0.0)
    p.add_argument("--epochs",              type=int,   default=15)
    p.add_argument("--batch_size",          type=int,   default=256)
    p.add_argument("--eval_batch_size",     type=int,   default=512)
    p.add_argument("--seed",                type=int,   default=42)
    return p.parse_args()


def main():
    args   = parse_args()
    set_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    logger = ExperimentLogger(args.output_dir)

    ckpt           = load_checkpoint(args.checkpoint, args.device)
    le_user, le_item = make_encoders_from_ckpt(ckpt)
    backbone       = load_backbone(ckpt, device)

    adapt_df  = read_interactions(args.adapt_data)
    adapt_enc = encode_df(adapt_df, le_user, le_item)
    seqs      = build_sequences_by_user(adapt_enc)
    maxlen    = ckpt["config"]["maxlen"]
    itemnum   = ckpt["itemnum"]

    train_ds = AdaptDataset(seqs, maxlen)
    eval_ds  = AdaptDataset(seqs, maxlen)
    loader      = DataLoader(train_ds, batch_size=args.batch_size,      shuffle=True,  num_workers=0, pin_memory=True)
    eval_loader = DataLoader(eval_ds,  batch_size=args.eval_batch_size, shuffle=False, num_workers=0, pin_memory=True)

    model = SASRecContextGateModel(
        backbone=backbone, bottleneck_dim=args.bottleneck_dim,
        adapter_dropout=args.adapter_dropout, adapter_activation=args.adapter_activation,
        freeze_backbone=True,
    ).to(device)

    optimizer  = torch.optim.Adam([p for p in model.parameters() if p.requires_grad],
                                  lr=args.lr, weight_decay=args.weight_decay)
    trainable  = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[context_gate] train examples={len(train_ds)}  trainable params={trainable:,}")

    mem = MemoryProfiler(); mem.reset()
    timer = Timer(); timer.start()

    best_loss, best_state = float("inf"), None
    ckpt_path = Path(args.output_dir) / "context_gate_best.pt"

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
        epoch_time = time.time() - t0

        logger.log_epoch({"epoch": epoch, "train_loss": float(avg_train),
                          "eval_loss": float(avg_eval), "epoch_time_s": round(epoch_time, 2)})
        print(f"Epoch {epoch:03d} | train_loss={avg_train:.4f} | eval_loss={avg_eval:.4f} | time={epoch_time:.1f}s")

        if avg_eval < best_loss:
            best_loss  = avg_eval
            best_state = {k: v.detach().cpu().clone() for k, v in model.adapter.state_dict().items()}
            print(f"[context_gate] new best at epoch {epoch} (eval_loss={best_loss:.4f})")

    print(f"[context_gate] total wall time: {timer.elapsed():.1f}s  {mem.report()}")
    logger.save_history()

    torch.save({
        "adapter_state_dict":  best_state,
        "bottleneck_dim":      int(args.bottleneck_dim),
        "adapter_dropout":     float(args.adapter_dropout),
        "adapter_activation":  args.adapter_activation,
        "backbone_config":     ckpt["config"],
        "itemnum":             int(itemnum),
        "source_checkpoint":   args.checkpoint,
        "best_eval_loss":      float(best_loss),
    }, ckpt_path)
    print(f"[context_gate] saved to {ckpt_path}")


if __name__ == "__main__":
    main()
