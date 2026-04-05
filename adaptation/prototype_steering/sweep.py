#!/usr/bin/env python3
"""
adaptation/prototype_steering/sweep.py

Grid search over prototype-steering hyperparameters.

Usage:
python adaptation/prototype_steering/sweep.py \
  --checkpoint   results/backbone/sasrec_backbone_best.pt \
  --adapt_data   data/processed/split_10M_contiguous/future_adapt_high_drift_kcore.csv \
  --test_data    data/processed/split_10M_contiguous/future_test_high_drift_kcore.csv \
  --base_outdir  results/sweep_prototype_steering_contiguous \
  --device cuda

--cluster_csv is optional. If omitted, per-K cluster CSVs are auto-generated from
user_drift_scores_final_subset.csv in the adapt_data directory and cached as
user_clusters_K<K>.csv so each unique K is only clustered once.
"""

import argparse
import itertools
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from common.metrics.ranking import METRIC_NAMES
from common.logging.logger import SweepLogger
from sweep_utils import run_cmd, load_summary, build_sweep_row, print_best


GRID = {
    "num_clusters":   [3, 5, 10],
    "bottleneck_dim": [16, 32, 64],
    "lr":             [1e-3, 5e-4],
    "epochs":         [20],
}

CSV_COLUMNS = (
    ["run_id", "num_clusters", "bottleneck_dim", "lr", "epochs"]
    + [f"{m}_{s}" for m in METRIC_NAMES for s in ("baseline", "finetuned", "delta", "pct_change")]
    + ["n_improved", "status"]
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint",   required=True)
    p.add_argument("--adapt_data",   required=True)
    p.add_argument("--cluster_csv",  default=None,
                   help="CSV with columns [user_id, cluster_id]. "
                        "If omitted, train.py auto-generates per-K cluster CSVs from "
                        "user_drift_scores_final_subset.csv in the adapt_data directory.")
    p.add_argument("--test_data",    required=True)
    p.add_argument("--base_outdir",  required=True)
    p.add_argument("--device",       default="cuda")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num_neg_eval", type=int, default=100)
    return p.parse_args()


def main():
    args     = parse_args()
    base_out = Path(args.base_outdir)
    base_out.mkdir(parents=True, exist_ok=True)
    sweep_log = SweepLogger(base_out, CSV_COLUMNS)

    keys    = list(GRID.keys())
    configs = [dict(zip(keys, vals)) for vals in itertools.product(*GRID.values())]
    total   = len(configs)
    print(f"[sweep] {total} configs  →  {sweep_log.path}")

    for run_id, cfg in enumerate(configs, 1):
        name    = f"run{run_id:03d}_k{cfg['num_clusters']}_bd{cfg['bottleneck_dim']}_lr{cfg['lr']}"
        run_dir = base_out / name
        ev_dir  = base_out / f"{name}_eval"

        print(f"\n{'='*60}\n[sweep] {run_id}/{total}: {name}\n{'='*60}")

        # Resolve the cluster CSV for this K value.
        # If --cluster_csv was not provided, train.py will auto-generate it at
        # <adapt_data_dir>/user_clusters_K<K>.csv — use that same path for eval.
        if args.cluster_csv:
            run_cluster_csv = args.cluster_csv
        else:
            run_cluster_csv = str(
                Path(args.adapt_data).parent / f"user_clusters_K{cfg['num_clusters']}.csv"
            )

        train_cmd = [
            sys.executable, "adaptation/prototype_steering/train.py",
            "--checkpoint",    args.checkpoint,
            "--adapt_data",    args.adapt_data,
            "--cluster_csv",   run_cluster_csv,
            "--output_dir",    str(run_dir),
            "--device",        args.device,
            "--num_clusters",  str(cfg["num_clusters"]),
            "--bottleneck_dim",str(cfg["bottleneck_dim"]),
            "--lr",            str(cfg["lr"]),
            "--epochs",        str(cfg["epochs"]),
            "--seed",          str(args.seed),
        ]
        if not run_cmd(train_cmd, "TRAIN"):
            continue

        adapt_ckpt = run_dir / "prototype_best.pt"
        if not adapt_ckpt.exists():
            continue

        eval_cmd = [
            sys.executable, "adaptation/prototype_steering/eval.py",
            "--checkpoint",       args.checkpoint,
            "--adapt_checkpoint", str(adapt_ckpt),
            "--cluster_csv",      run_cluster_csv,
            "--test_data",        args.test_data,
            "--outdir",           str(ev_dir),
            "--device",           args.device,
            "--num_neg_eval",     str(args.num_neg_eval),
            "--seed",             str(args.seed),
        ]
        if not run_cmd(eval_cmd, "EVAL"):
            continue

        summary = load_summary(ev_dir / "summary.json")
        if summary is None:
            continue

        row = build_sweep_row(name, cfg, summary, key_adapted="adapted_metrics")
        sweep_log.log_run(row)
        print(f"[sweep] NDCG@10 delta={row['ndcg@10_delta']:+.6f}  status={row['status']}")

    print(f"\n[sweep] DONE — {len(sweep_log.rows)}/{total} runs")
    print_best(sweep_log.best_run(), METRIC_NAMES)


if __name__ == "__main__":
    main()
