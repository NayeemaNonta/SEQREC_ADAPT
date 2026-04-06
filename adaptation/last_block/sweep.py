#!/usr/bin/env python3
"""
adaptation/last_block/sweep.py

Grid search over last-block fine-tuning hyperparameters.
Runs train + eval for each config and writes results to sweep_results.csv.

Usage:
python adaptation/last_block/sweep.py \
  --checkpoint     results/backbone/sasrec_backbone_best.pt \
  --adapt_data     data/processed/future_adapt.csv \
  --test_data      data/processed/future_test.csv \
  --base_outdir    results/sweep_last_block \
  --device cuda
"""

import argparse
import itertools
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from common.metrics.ranking import METRIC_NAMES
from common.logging.logger import SweepLogger
from sweep_utils import run_cmd, load_summary, build_sweep_row, print_best


# ---------------------------------------------------------------------------
# Grid — edit to change search space
# ---------------------------------------------------------------------------
GRID = {
    "lr":                     [1e-3, 5e-4, 1e-4, 5e-5],
    "epochs":                 [20, 50],
    "include_last_layernorm": [False, True],
    "weight_decay":           [0.0, 1e-4],
}

CSV_COLUMNS = (
    ["run_id", "lr", "epochs", "include_last_layernorm", "weight_decay", "best_eval_loss"]
    + [f"{m}_{s}" for m in METRIC_NAMES for s in ("baseline", "finetuned", "delta", "pct_change")]
    + ["n_improved", "status"]
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint",      required=True)
    p.add_argument("--adapt_data",      required=True)
    p.add_argument("--test_data",       required=True)
    p.add_argument("--base_outdir",     required=True)
    p.add_argument("--device",          default="cuda")
    p.add_argument("--seed",  type=int, default=42)
    p.add_argument("--num_neg_eval", type=int, default=100)
    p.add_argument("--skip_existing", action="store_true",
                   help="Skip any run whose eval summary.json already exists.")
    return p.parse_args()


def main():
    args      = parse_args()
    base_out  = Path(args.base_outdir)
    base_out.mkdir(parents=True, exist_ok=True)
    sweep_log = SweepLogger(base_out, CSV_COLUMNS)

    keys    = list(GRID.keys())
    configs = [dict(zip(keys, vals)) for vals in itertools.product(*GRID.values())]
    total   = len(configs)
    print(f"[sweep] {total} configs  →  {sweep_log.path}")

    for run_id, cfg in enumerate(configs, 1):
        ln      = cfg["include_last_layernorm"]
        name    = f"run{run_id:03d}_lr{cfg['lr']}_ep{cfg['epochs']}_ln{int(ln)}_wd{cfg['weight_decay']}"
        run_dir = base_out / name
        ev_dir  = base_out / f"{name}_eval"

        if args.skip_existing and (ev_dir / "summary.json").exists():
            print(f"[sweep] SKIP (already done): {name}")
            continue

        print(f"\n{'='*60}\n[sweep] {run_id}/{total}: {name}\n{'='*60}")

        train_cmd = [
            sys.executable, "adaptation/last_block/train.py",
            "--checkpoint",  args.checkpoint,
            "--adapt_data",  args.adapt_data,
            "--output_dir",  str(run_dir),
            "--device",      args.device,
            "--lr",          str(cfg["lr"]),
            "--epochs",      str(cfg["epochs"]),
            "--weight_decay",str(cfg["weight_decay"]),
            "--seed",        str(args.seed),
        ]
        if ln:
            train_cmd.append("--include_last_layernorm")

        if not run_cmd(train_cmd, "TRAIN"):
            continue

        ft_ckpt = run_dir / "last_block_best.pt"
        if not ft_ckpt.exists():
            print("[sweep] checkpoint not found, skipping")
            continue

        eval_cmd = [
            sys.executable, "adaptation/last_block/eval.py",
            "--checkpoint",    args.checkpoint,
            "--ft_checkpoint", str(ft_ckpt),
            "--test_data",     args.test_data,
            "--outdir",        str(ev_dir),
            "--device",        args.device,
            "--num_neg_eval",  str(args.num_neg_eval),
            "--seed",          str(args.seed),
        ]
        if not run_cmd(eval_cmd, "EVAL"):
            continue

        summary = load_summary(ev_dir / "summary.json")
        if summary is None:
            continue

        row = build_sweep_row(name, cfg, summary, key_adapted="finetuned_metrics")
        sweep_log.log_run(row)
        print(f"[sweep] NDCG@10 delta={row['ndcg@10_delta']:+.6f}  "
              f"HR@10 delta={row['hr@10_delta']:+.6f}  status={row['status']}")

    print(f"\n[sweep] DONE — {len(sweep_log.rows)}/{total} runs")
    print_best(sweep_log.best_run(), METRIC_NAMES)


if __name__ == "__main__":
    main()
