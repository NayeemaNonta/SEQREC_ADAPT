#!/usr/bin/env python3
"""
adaptation/prototype_steering/train.py

Train a cluster-specific MLP adapter (one per prototype cluster).
Backbone is fully frozen; each cluster learns an independent residual:

  h̃_{z_u} = h + f_{ϕ_{z_u}}(h)    z_u ∈ {0, …, K-1}

Cluster assignments are read from --cluster_csv (columns: user_id, cluster_id).
If --cluster_csv is omitted or the file does not exist, clustering is run
automatically using user_drift_scores_final_subset.csv from the same directory
as --adapt_data, and the result is cached at:
  <adapt_data_dir>/user_clusters_K<num_clusters>.csv

Usage:
python adaptation/prototype_steering/train.py \
  --checkpoint   results/backbone/sasrec_backbone_best.pt \
  --adapt_data   data/processed/split_10M_contiguous/future_adapt_high_drift_kcore.csv \
  --output_dir   results/prototype_steering \
  --num_clusters 5 --bottleneck_dim 32 --device cuda
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
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

import pandas as pd
import random

from adaptation.prototype_steering.cluster_users import cluster_users


def set_seed(seed):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class PrototypeAdapter(nn.Module):
    """K independent bottleneck MLPs, one per cluster."""

    def __init__(self, num_clusters: int, hidden_dim: int, bottleneck_dim: int):
        super().__init__()
        self.adapters = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, bottleneck_dim),
                nn.GELU(),
                nn.Linear(bottleneck_dim, hidden_dim),
            )
            for _ in range(num_clusters)
        ])
        # zero-init output projection so residual starts at 0
        for adapter in self.adapters:
            nn.init.zeros_(adapter[2].weight)
            nn.init.zeros_(adapter[2].bias)

    def forward(self, h: torch.Tensor, cluster_ids: torch.Tensor) -> torch.Tensor:
        out = torch.zeros_like(h)
        for c_id, adapter in enumerate(self.adapters):
            mask = (cluster_ids == c_id)
            if mask.any():
                out[mask] = h[mask] + adapter(h[mask])
        return out


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class AdaptDataset(Dataset):
    def __init__(self, user_seqs: dict, cluster_map: dict, itemnum: int,
                 maxlen: int, num_neg: int = 10):
        self.itemnum = itemnum
        self.maxlen  = maxlen
        self.num_neg = num_neg
        self.examples = []
        for u, seq in user_seqs.items():
            if u not in cluster_map:
                continue
            c = cluster_map[u]
            for t in range(1, len(seq)):
                self.examples.append((seq[:t], seq[t], set(seq), c))

    def __len__(self): return len(self.examples)

    def _neg(self, seen, pos):
        negs = []
        while len(negs) < self.num_neg:
            n = random.randint(1, self.itemnum)
            if n != pos and n not in seen: negs.append(n)
        return negs

    def __getitem__(self, idx):
        prefix, pos, seen, c = self.examples[idx]
        return {
            "input_ids":  torch.tensor(pad_sequence(prefix, self.maxlen), dtype=torch.long),
            "pos_item":   torch.tensor(pos, dtype=torch.long),
            "neg_items":  torch.tensor(self._neg(seen, pos), dtype=torch.long),
            "cluster_id": torch.tensor(c, dtype=torch.long),
        }


# ---------------------------------------------------------------------------
# Loss / eval
# ---------------------------------------------------------------------------

def bce_loss(backbone, adapter, input_ids, pos_item, neg_items, cluster_ids):
    with torch.no_grad():
        h = backbone.get_last_hidden(input_ids)
    h_tilde = adapter(h, cluster_ids)
    pos_logits = backbone.score_from_hidden(h_tilde, pos_item.unsqueeze(1)).squeeze(1)
    neg_logits = backbone.score_from_hidden(h_tilde, neg_items)
    return (-torch.log(torch.sigmoid(pos_logits) + 1e-24).mean()
            - torch.log(1.0 - torch.sigmoid(neg_logits) + 1e-24).mean())


@torch.no_grad()
def eval_bce(backbone, adapter, loader, device):
    adapter.eval()
    total, steps = 0.0, 0
    for b in loader:
        total += bce_loss(
            backbone, adapter,
            b["input_ids"].to(device), b["pos_item"].to(device),
            b["neg_items"].to(device), b["cluster_id"].to(device),
        ).item()
        steps += 1
    adapter.train()
    return total / max(steps, 1)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def resolve_cluster_csv(cluster_csv: str | None, adapt_data: str, num_clusters: int, seed: int) -> str:
    """
    Return path to a valid cluster CSV. If cluster_csv is not provided or the
    file does not exist, auto-generate it from user_drift_scores_final_subset.csv
    located in the same directory as adapt_data.
    """
    if cluster_csv and Path(cluster_csv).exists():
        return cluster_csv

    adapt_dir = Path(adapt_data).parent
    auto_path = adapt_dir / f"user_clusters_K{num_clusters}.csv"

    if auto_path.exists():
        print(f"[cluster] using cached cluster CSV: {auto_path}")
        return str(auto_path)

    drift_scores = adapt_dir / "user_drift_scores_final_subset.csv"
    if not drift_scores.exists():
        raise FileNotFoundError(
            f"Cannot auto-generate cluster CSV: drift scores not found at {drift_scores}\n"
            f"Either provide --cluster_csv explicitly or run the preprocessing pipeline first."
        )

    print(f"[cluster] cluster CSV not found — running clustering (K={num_clusters}) ...")
    return cluster_users(
        drift_scores_csv=str(drift_scores),
        num_clusters=num_clusters,
        outdir=str(adapt_dir),
        seed=seed,
    )


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint",    required=True)
    p.add_argument("--adapt_data",    required=True)
    p.add_argument("--cluster_csv",   default=None,
                   help="CSV with columns [user_id, cluster_id]. "
                        "If omitted, auto-generated from user_drift_scores_final_subset.csv "
                        "in the adapt_data directory and cached as user_clusters_K<K>.csv.")
    p.add_argument("--output_dir",    required=True)
    p.add_argument("--device",        default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--num_clusters",  type=int,   default=5)
    p.add_argument("--bottleneck_dim",type=int,   default=32)
    p.add_argument("--lr",            type=float, default=1e-3)
    p.add_argument("--weight_decay",  type=float, default=0.0)
    p.add_argument("--epochs",        type=int,   default=20)
    p.add_argument("--batch_size",    type=int,   default=256)
    p.add_argument("--num_neg_train", type=int,   default=10)
    p.add_argument("--seed",          type=int,   default=42)
    return p.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    logger = ExperimentLogger(args.output_dir)

    ckpt = load_checkpoint(args.checkpoint, args.device)
    le_user, le_item = make_encoders_from_ckpt(ckpt)

    # backbone — fully frozen
    cfg = ckpt["config"]
    backbone = SASRec(
        item_num=ckpt["itemnum"], maxlen=cfg["maxlen"],
        hidden_units=cfg["hidden_units"], num_blocks=cfg["num_blocks"],
        num_heads=cfg["num_heads"], dropout_rate=cfg["dropout_rate"],
        add_head=False, pos_enc=True,
    ).to(device)
    backbone.load_state_dict(ckpt["model_state_dict"])
    for p in backbone.parameters(): p.requires_grad_(False)
    backbone.eval()

    # cluster map: user_id (str) → cluster_id (int)
    cluster_csv_path = resolve_cluster_csv(
        args.cluster_csv, args.adapt_data, args.num_clusters, args.seed
    )
    cdf = pd.read_csv(cluster_csv_path)
    cdf["user_id"] = cdf["user_id"].astype(str)
    known_users = set(le_user.classes_.tolist())
    cdf = cdf[cdf["user_id"].isin(known_users)]
    user_to_cluster = dict(zip(le_user.transform(cdf["user_id"]), cdf["cluster_id"].astype(int)))

    adapt_enc  = encode_df(read_interactions(args.adapt_data), le_user, le_item)
    adapt_seqs = build_sequences_by_user(adapt_enc)
    maxlen     = cfg["maxlen"]

    ds      = AdaptDataset(adapt_seqs, user_to_cluster, ckpt["itemnum"], maxlen, args.num_neg_train)
    loader  = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=True)

    adapter   = PrototypeAdapter(args.num_clusters, cfg["hidden_units"], args.bottleneck_dim).to(device)
    optimizer = torch.optim.Adam(adapter.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    trainable = sum(p.numel() for p in adapter.parameters())

    print(f"[prototype] train examples={len(ds)}  clusters={args.num_clusters}"
          f"  trainable params={trainable:,}")

    mem = MemoryProfiler(); mem.reset()
    timer = Timer(); timer.start()

    best_loss, best_state = float("inf"), None
    ckpt_path = Path(args.output_dir) / "prototype_best.pt"

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        adapter.train()
        running, steps = 0.0, 0
        pbar = tqdm(loader, desc=f"Epoch {epoch:03d}", leave=False, dynamic_ncols=True)
        for batch in pbar:
            optimizer.zero_grad()
            loss = bce_loss(backbone, adapter,
                            batch["input_ids"].to(device), batch["pos_item"].to(device),
                            batch["neg_items"].to(device), batch["cluster_id"].to(device))
            loss.backward(); optimizer.step()
            running += loss.item(); steps += 1
            pbar.set_postfix(loss=f"{running/steps:.4f}")

        avg = running / max(steps, 1)
        logger.log_epoch({"epoch": epoch, "train_loss": float(avg),
                          "epoch_time_s": round(time.time()-t0, 2)})
        print(f"Epoch {epoch:03d} | train_loss={avg:.4f} | time={time.time()-t0:.1f}s")

        if avg < best_loss:
            best_loss  = avg
            best_state = {k: v.detach().cpu().clone() for k, v in adapter.state_dict().items()}
            print(f"[prototype] new best at epoch {epoch}")

    print(f"[prototype] total wall time: {timer.elapsed():.1f}s  {mem.report()}")
    logger.save_history()

    torch.save({
        "adapter_state_dict": best_state,
        "num_clusters":       args.num_clusters,
        "bottleneck_dim":     args.bottleneck_dim,
        "backbone_config":    cfg,
        "itemnum":            int(ckpt["itemnum"]),
        "source_checkpoint":  args.checkpoint,
        "cluster_csv":        args.cluster_csv,
        "best_train_loss":    float(best_loss),
    }, ckpt_path)
    print(f"[prototype] saved to {ckpt_path}")


if __name__ == "__main__":
    main()
