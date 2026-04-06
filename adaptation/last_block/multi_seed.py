#!/usr/bin/env python3
"""
adaptation/last_block/multi_seed.py

Run the best last-block config across N seeds for statistical significance testing.
Seed controls: DataLoader shuffle order.
(Backbone weights are a deterministic init — variance comes from training data order only.)

Best config (from sweep): lr=5e-4, epochs=50, include_last_layernorm=True, weight_decay=1e-4

Usage:
python adaptation/last_block/multi_seed.py \
  --checkpoint  results/backbone/sasrec_backbone_best.pt \
  --adapt_data  data/processed/split_10M_contiguous/future_adapt_high_drift_kcore.csv \
  --test_data   data/processed/split_10M_contiguous/future_test_high_drift_kcore.csv \
  --base_outdir results/multi_seed_last_block_contiguous --device cuda
"""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from common.metrics.ranking import METRIC_NAMES
from common.logging.logger import SweepLogger
from sweep_utils import run_cmd, load_summary, build_sweep_row

# ---------------------------------------------------------------------------
# Best config from HP sweep
# ---------------------------------------------------------------------------

BEST = {
    "lr":                    5e-4,
    "epochs":                50,
    "include_last_layernorm": True,   # passed as flag to train.py
    "weight_decay":          1e-4,
}

SEEDS = list(range(10))   # seeds 0–9

CSV_COLUMNS = (
    ["run_id", "seed", "best_eval_loss"]
    + [f"{m}_{s}" for m in METRIC_NAMES for s in ("baseline", "finetuned", "delta", "pct_change")]
    + ["n_improved", "status"]
)


# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint",   required=True)
    p.add_argument("--adapt_data",   required=True)
    p.add_argument("--test_data",    required=True)
    p.add_argument("--base_outdir",  required=True)
    p.add_argument("--device",       default="cuda")
    p.add_argument("--num_neg_eval",          type=int,   default=100)
    p.add_argument("--seeds",                 type=int, nargs="+", default=SEEDS,
                   help="Seeds to run (default: 0–9)")
    # HP overrides — default to BEST dict above
    p.add_argument("--lr",                    type=float, default=BEST["lr"])
    p.add_argument("--epochs",                type=int,   default=BEST["epochs"])
    p.add_argument("--weight_decay",          type=float, default=BEST["weight_decay"])
    p.add_argument("--include_last_layernorm",action="store_true",
                   default=BEST["include_last_layernorm"])
    return p.parse_args()


def print_stats(rows: list[dict]):
    print(f"\n{'='*65}")
    print(f"  MULTI-SEED SUMMARY  ({len(rows)} seeds)")
    print(f"{'='*65}")
    print(f"  {'Metric':<12}  {'Mean %Δ':>9}  {'Std':>7}  {'Min %Δ':>9}  {'Max %Δ':>9}")
    print(f"  {'-'*55}")
    for m in METRIC_NAMES:
        pcts = np.array([r[f"{m}_pct_change"] for r in rows])
        print(f"  {m:<12}  {pcts.mean():>+8.3f}%  {pcts.std():>6.3f}%  "
              f"{pcts.min():>+8.3f}%  {pcts.max():>+8.3f}%")
    print()


def main():
    args     = parse_args()
    base_out = Path(args.base_outdir)
    base_out.mkdir(parents=True, exist_ok=True)
    sweep_log = SweepLogger(base_out, CSV_COLUMNS)

    cfg = {
        "lr": args.lr, "epochs": args.epochs,
        "weight_decay": args.weight_decay,
        "include_last_layernorm": args.include_last_layernorm,
    }
    seeds = args.seeds
    print(f"[multi_seed] last_block | {len(seeds)} seeds | config: {cfg}")
    print(f"[multi_seed] output → {sweep_log.path}\n")

    for seed in seeds:
        name    = f"seed{seed:02d}"
        run_dir = base_out / name
        ev_dir  = base_out / f"{name}_eval"

        print(f"\n{'='*60}\n[multi_seed] seed={seed}\n{'='*60}")

        train_cmd = [
            sys.executable, "adaptation/last_block/train.py",
            "--checkpoint",  args.checkpoint,
            "--adapt_data",  args.adapt_data,
            "--output_dir",  str(run_dir),
            "--device",      args.device,
            "--lr",          str(cfg["lr"]),
            "--epochs",      str(cfg["epochs"]),
            "--weight_decay",str(cfg["weight_decay"]),
            "--seed",        str(seed),
        ]
        if cfg["include_last_layernorm"]:
            train_cmd.append("--include_last_layernorm")

        if not run_cmd(train_cmd, "TRAIN"):
            continue

        ft_ckpt = run_dir / "last_block_best.pt"
        if not ft_ckpt.exists():
            continue

        eval_cmd = [
            sys.executable, "adaptation/last_block/eval.py",
            "--checkpoint",    args.checkpoint,
            "--ft_checkpoint", str(ft_ckpt),
            "--test_data",     args.test_data,
            "--outdir",        str(ev_dir),
            "--device",        args.device,
            "--num_neg_eval",  str(args.num_neg_eval),
            "--seed",          str(seed),
        ]
        if not run_cmd(eval_cmd, "EVAL"):
            continue

        summary = load_summary(ev_dir / "summary.json")
        if summary is None:
            continue

        row = build_sweep_row(name, {"seed": seed}, summary, key_adapted="finetuned_metrics")
        sweep_log.log_run(row)
        print(f"[multi_seed] seed={seed}  NDCG@10 delta={row['ndcg@10_delta']:+.6f}  status={row['status']}")

    print(f"\n[multi_seed] DONE — {len(sweep_log.rows)}/{len(seeds)} seeds completed")
    if sweep_log.rows:
        print_stats(sweep_log.rows)


if __name__ == "__main__":
    main()
