#!/usr/bin/env python3
"""
backbone/train_backbone.py

Train SASRec backbone on historical data (pre-deployment).
Uses packed-sequence cross-entropy over full item vocabulary.
The saved checkpoint is frozen for all downstream adaptation experiments.

Usage:
python backbone/train_backbone.py \
  --hist_data   data/processed/hist_kcore.csv \
  --val_data    data/processed/future_val.csv \
  --output_dir  results/backbone \
  --device cuda
"""

import argparse
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backbone.model import SASRec
from common.data_utils import (
    read_interactions, encode_df, build_sequences_by_user,
    leave_one_out, build_input_tensor, sample_negatives,
)
from common.metrics.ranking import summarize, METRIC_NAMES
from common.memory_profiler.profiler import MemoryProfiler, Timer


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Dataset — packed sequences, one sample per user per position
# ---------------------------------------------------------------------------

class SeqDataset(Dataset):
    """
    One (input_ids, target) pair per (user, position).
    target is 0-indexed for F.cross_entropy (item_idx - 1).
    """

    def __init__(self, user_seqs: dict, maxlen: int):
        self.maxlen = maxlen
        self.examples = []
        for seq in user_seqs.values():
            for t in range(1, len(seq)):
                prefix = seq[:t]
                target = seq[t] - 1  # 0-indexed
                self.examples.append((prefix, target))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        prefix, target = self.examples[idx]
        prefix = prefix[-self.maxlen:]
        padded = [0] * (self.maxlen - len(prefix)) + prefix
        return {
            "input_ids": torch.tensor(padded, dtype=torch.long),
            "target":    torch.tensor(target, dtype=torch.long),
        }


# ---------------------------------------------------------------------------
# Training / evaluation helpers
# ---------------------------------------------------------------------------

def train_epoch(model, loader, optimizer, device, epoch):
    model.train()
    running, steps = 0.0, 0
    pbar = tqdm(loader, desc=f"Epoch {epoch:03d}", leave=False, dynamic_ncols=True)
    for batch in pbar:
        input_ids = batch["input_ids"].to(device)
        target    = batch["target"].to(device)
        optimizer.zero_grad()
        h      = model.get_last_hidden(input_ids)
        logits = model.score_from_hidden(h)[:, 1:]   # drop padding idx
        loss   = F.cross_entropy(logits, target)
        loss.backward()
        optimizer.step()
        running += loss.item()
        steps   += 1
        pbar.set_postfix(loss=f"{running/steps:.4f}")
    return running / max(steps, 1)


@torch.no_grad()
def evaluate_ndcg(model, context_map, target_map, itemnum, maxlen, device, n_neg=100):
    model.eval()
    rows = []
    for user in sorted(set(context_map) & set(target_map)):
        context = context_map[user]
        target  = target_map[user]
        if not context:
            continue
        seen  = set(context) | {0}
        negs  = sample_negatives(itemnum, seen, target, n_neg=n_neg)
        cands = [target] + negs

        input_ids     = build_input_tensor(context, maxlen, device)
        candidate_ids = torch.tensor([cands], dtype=torch.long, device=device)
        h      = model.get_last_hidden(input_ids)
        logits = model.score_from_hidden(h, candidate_ids)[0]
        rank   = int((logits > logits[0]).sum().item())
        rows.append({
            "hit@10":  int(rank < 10),
            "hit@20":  int(rank < 20),
            "ndcg@10": 1.0 / np.log2(rank + 2) if rank < 10 else 0.0,
            "ndcg@20": 1.0 / np.log2(rank + 2) if rank < 20 else 0.0,
            "mrr@10":  1.0 / (rank + 1)         if rank < 10 else 0.0,
        })
    return summarize(pd.DataFrame(rows))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--hist_data",   required=True)
    p.add_argument("--val_data",    required=True)
    p.add_argument("--output_dir",  required=True)
    p.add_argument("--hidden_units",type=int,   default=64)
    p.add_argument("--num_blocks",  type=int,   default=2)
    p.add_argument("--num_heads",   type=int,   default=2)
    p.add_argument("--dropout_rate",type=float, default=0.2)
    p.add_argument("--maxlen",      type=int,   default=128)
    p.add_argument("--epochs",      type=int,   default=200)
    p.add_argument("--batch_size",  type=int,   default=128)
    p.add_argument("--lr",          type=float, default=1e-3)
    p.add_argument("--weight_decay",type=float, default=0.0)
    p.add_argument("--patience",    type=int,   default=20)
    p.add_argument("--eval_every",  type=int,   default=5)
    p.add_argument("--num_neg_eval",type=int,   default=100)
    p.add_argument("--seed",        type=int,   default=42)
    p.add_argument("--device",      type=str,   default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # --- data ---
    hist_df = read_interactions(args.hist_data)
    val_df  = read_interactions(args.val_data)

    le_item = LabelEncoder().fit(hist_df["item_id"].astype(str))
    le_user = LabelEncoder().fit(hist_df["user_id"].astype(str))
    itemnum = len(le_item.classes_)

    hist_enc = encode_df(hist_df, le_user, le_item)
    val_enc  = encode_df(val_df,  le_user, le_item)

    hist_seqs = build_sequences_by_user(hist_enc)
    val_seqs  = build_sequences_by_user(val_enc)
    val_ctx, val_tgt, _ = leave_one_out(val_seqs)

    dataset = SeqDataset(hist_seqs, maxlen=args.maxlen)
    loader  = DataLoader(dataset, batch_size=args.batch_size,
                         shuffle=True, num_workers=0, pin_memory=True)

    # --- model ---
    model = SASRec(
        item_num=itemnum, maxlen=args.maxlen,
        hidden_units=args.hidden_units, num_blocks=args.num_blocks,
        num_heads=args.num_heads, dropout_rate=args.dropout_rate,
        add_head=False, pos_enc=True,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,
                                 weight_decay=args.weight_decay)

    print(f"[backbone] device={device}  itemnum={itemnum}  "
          f"train examples={len(dataset)}  params={sum(p.numel() for p in model.parameters()):,}")

    mem = MemoryProfiler()
    mem.reset()
    timer = Timer()
    timer.start()

    best_ndcg = 0.0
    best_state = None
    no_improve = 0
    ckpt_path  = outdir / "sasrec_backbone_best.pt"

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss = train_epoch(model, loader, optimizer, device, epoch)

        if epoch % args.eval_every == 0:
            val_metrics = evaluate_ndcg(
                model, val_ctx, val_tgt, itemnum,
                args.maxlen, device, n_neg=args.num_neg_eval,
            )
            ndcg = val_metrics["ndcg@10"]
            print(f"Epoch {epoch:03d} | train_loss={train_loss:.4f} | "
                  f"val_ndcg@10={ndcg:.6f} | time={time.time()-t0:.1f}s")

            if ndcg > best_ndcg:
                best_ndcg = ndcg
                no_improve = 0
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                print(f"[backbone] new best ndcg@10={best_ndcg:.6f} at epoch {epoch}")
            else:
                no_improve += args.eval_every
                if no_improve >= args.patience:
                    print(f"[backbone] early stopping at epoch {epoch} (no improvement for {args.patience} epochs)")
                    break
        else:
            print(f"Epoch {epoch:03d} | train_loss={train_loss:.4f} | time={time.time()-t0:.1f}s")

    total_wall = timer.elapsed()
    print(f"[backbone] total wall time: {total_wall:.1f}s  {mem.report()}")

    torch.save({
        "model_state_dict": best_state,
        "config": {
            "hidden_units": args.hidden_units, "num_blocks": args.num_blocks,
            "num_heads": args.num_heads, "dropout_rate": args.dropout_rate,
            "maxlen": args.maxlen,
        },
        "itemnum":         itemnum,
        "le_item_classes": le_item.classes_.tolist(),
        "le_user_classes": le_user.classes_.tolist(),
        "best_val_ndcg10": float(best_ndcg),
    }, ckpt_path)
    print(f"[backbone] saved to {ckpt_path}")


if __name__ == "__main__":
    main()
