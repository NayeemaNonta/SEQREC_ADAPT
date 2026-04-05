#!/usr/bin/env python3
"""
adaptation/last_block/train.py

Fine-tune only the last SASRec transformer block on future_adapt data.
All earlier layers are frozen. Loss: cross-entropy over full item vocabulary.

Trainable modules:
  attention_layernorms[-1], attention_layers[-1],
  forward_layernorms[-1],   forward_layers[-1],
  last_layernorm (optional, --include_last_layernorm)

Usage:
python adaptation/last_block/train.py \
  --checkpoint  results/backbone/sasrec_backbone_best.pt \
  --adapt_data  data/processed/future_adapt.csv \
  --output_dir  results/last_block \
  --device cuda --lr 1e-4 --epochs 20
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

from backbone.model import SASRec
from common.data_utils import (
    read_interactions, load_checkpoint, make_encoders_from_ckpt,
    encode_df, build_sequences_by_user, pad_sequence,
)
from common.memory_profiler.profiler import MemoryProfiler, Timer
from common.logging.logger import ExperimentLogger


def set_seed(seed: int):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

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
        padded = pad_sequence(prefix, self.maxlen)
        return {
            "input_ids": torch.tensor(padded, dtype=torch.long),
            "target":    torch.tensor(target, dtype=torch.long),
        }


# ---------------------------------------------------------------------------
# Model setup
# ---------------------------------------------------------------------------

def load_backbone(ckpt: dict, device) -> SASRec:
    cfg = ckpt["config"]
    model = SASRec(
        item_num=ckpt["itemnum"], maxlen=cfg["maxlen"],
        hidden_units=cfg["hidden_units"], num_blocks=cfg["num_blocks"],
        num_heads=cfg["num_heads"], dropout_rate=cfg["dropout_rate"],
        add_head=False, pos_enc=True,
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    return model


def freeze_all(model: SASRec):
    for p in model.parameters():
        p.requires_grad_(False)


def unfreeze_last_block(model: SASRec, include_last_layernorm: bool = False):
    last = model.num_blocks - 1
    trainable = [
        model.attention_layernorms[last],
        model.attention_layers[last],
        model.forward_layernorms[last],
        model.forward_layers[last],
    ]
    if include_last_layernorm:
        trainable.append(model.last_layernorm)
    for m in trainable:
        for p in m.parameters():
            p.requires_grad_(True)


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

def ce_loss(model: SASRec, input_ids: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    h      = model.get_last_hidden(input_ids)
    logits = model.score_from_hidden(h)[:, 1:]   # drop padding idx → (B, itemnum)
    return F.cross_entropy(logits, target)


@torch.no_grad()
def eval_loss(model: SASRec, loader, device) -> float:
    model.eval()
    total, steps = 0.0, 0
    for batch in loader:
        total += ce_loss(
            model,
            batch["input_ids"].to(device),
            batch["target"].to(device),
        ).item()
        steps += 1
    return total / max(steps, 1)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint",             required=True)
    p.add_argument("--adapt_data",             required=True)
    p.add_argument("--output_dir",             required=True)
    p.add_argument("--device",                 default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--lr",          type=float, default=1e-4)
    p.add_argument("--weight_decay",type=float, default=0.0)
    p.add_argument("--epochs",      type=int,   default=20)
    p.add_argument("--batch_size",  type=int,   default=256)
    p.add_argument("--eval_batch_size", type=int, default=512)
    p.add_argument("--include_last_layernorm", action="store_true")
    p.add_argument("--seed",        type=int,   default=42)
    return p.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    logger = ExperimentLogger(args.output_dir)

    ckpt       = load_checkpoint(args.checkpoint, args.device)
    le_user, le_item = make_encoders_from_ckpt(ckpt)
    model      = load_backbone(ckpt, device)
    freeze_all(model)
    unfreeze_last_block(model, args.include_last_layernorm)

    adapt_df   = read_interactions(args.adapt_data)
    adapt_enc  = encode_df(adapt_df, le_user, le_item)
    adapt_seqs = build_sequences_by_user(adapt_enc)
    maxlen     = ckpt["config"]["maxlen"]

    train_ds = AdaptDataset(adapt_seqs, maxlen)
    eval_ds  = AdaptDataset(adapt_seqs, maxlen)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size,   shuffle=True,  num_workers=0, pin_memory=True)
    eval_loader  = DataLoader(eval_ds,  batch_size=args.eval_batch_size, shuffle=False, num_workers=0, pin_memory=True)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    optimizer  = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr, weight_decay=args.weight_decay,
    )

    print(f"[last_block] device={device}  itemnum={ckpt['itemnum']}")
    print(f"[last_block] train examples={len(train_ds)}  trainable params={trainable:,}")

    mem = MemoryProfiler(); mem.reset()
    timer = Timer(); timer.start()

    best_loss  = float("inf")
    best_state = None
    ckpt_path  = Path(args.output_dir) / "last_block_best.pt"

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()
        running, steps = 0.0, 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch:03d}", leave=False, dynamic_ncols=True)
        for batch in pbar:
            optimizer.zero_grad()
            loss = ce_loss(model, batch["input_ids"].to(device), batch["target"].to(device))
            loss.backward()
            optimizer.step()
            running += loss.item(); steps += 1
            pbar.set_postfix(loss=f"{running/steps:.4f}")

        avg_train = running / max(steps, 1)
        avg_eval  = eval_loss(model, eval_loader, device)
        epoch_time = time.time() - t0

        row = {"epoch": epoch, "train_loss": float(avg_train),
               "eval_loss": float(avg_eval), "epoch_time_s": round(epoch_time, 2)}
        logger.log_epoch(row)
        print(f"Epoch {epoch:03d} | train_loss={avg_train:.4f} | eval_loss={avg_eval:.4f} | time={epoch_time:.1f}s")

        if avg_eval < best_loss:
            best_loss  = avg_eval
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            print(f"[last_block] new best at epoch {epoch} (eval_loss={best_loss:.4f})")

    print(f"[last_block] total wall time: {timer.elapsed():.1f}s  {mem.report()}")
    logger.save_history()

    torch.save({
        "model_state_dict":       best_state,
        "config":                 ckpt["config"],
        "itemnum":                int(ckpt["itemnum"]),
        "le_item_classes":        ckpt["le_item_classes"],
        "le_user_classes":        ckpt["le_user_classes"],
        "source_checkpoint":      args.checkpoint,
        "best_eval_loss":         float(best_loss),
        "include_last_layernorm": bool(args.include_last_layernorm),
    }, ckpt_path)
    print(f"[last_block] saved to {ckpt_path}")


if __name__ == "__main__":
    main()
