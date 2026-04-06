#!/usr/bin/env python3
"""
adaptation/prototype_steering/multi_seed.py

Run the best prototype-steering config across N seeds for statistical significance testing.
Seed controls: DataLoader shuffle order, negative sampling, adapter weight initialisation.
(Highest variance of the three methods due to random negatives + random MLP init.)

Best config (from sweep — tail split): num_clusters=5, bottleneck_dim=32, lr=1e-3, epochs=20
Note: contiguous split showed degradation across all runs; tail best is used as default.

Usage:
python adaptation/prototype_steering/multi_seed.py \
  --checkpoint  results/backbone/sasrec_backbone_best.pt \
  --adapt_data  data/processed/split_10M_contiguous/future_adapt_high_drift_kcore.csv \
  --test_data   data/processed/split_10M_contiguous/future_test_high_drift_kcore.csv \
  --base_outdir results/multi_seed_prototype_steering_contiguous --device cuda
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
    "num_clusters":   5,
    "bottleneck_dim": 64,
    "lr":             1e-3,
    "epochs":         20,
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
    p.add_argument("--num_neg_eval", type=int, default=100)
    p.add_argument("--cluster_csv",  default=None,
                   help="Pre-computed cluster CSV. If omitted, auto-generated from "
                        "user_drift_scores_final_subset.csv in the adapt_data directory.")
    p.add_argument("--seeds",          type=int, nargs="+", default=SEEDS,
                   help="Seeds to run (default: 0–9)")
    # HP overrides — default to BEST dict above
    p.add_argument("--num_clusters",   type=int,   default=BEST["num_clusters"])
    p.add_argument("--bottleneck_dim", type=int,   default=BEST["bottleneck_dim"])
    p.add_argument("--lr",             type=float, default=BEST["lr"])
    p.add_argument("--epochs",         type=int,   default=BEST["epochs"])
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


def _cluster_csv_for(adapt_data: str, num_clusters: int, cluster_csv: str | None) -> str:
    """Resolve cluster CSV path — explicit arg > auto-derived path."""
    if cluster_csv:
        return cluster_csv
    return str(Path(adapt_data).parent / f"user_clusters_K{num_clusters}.csv")


def main():
    args     = parse_args()
    base_out = Path(args.base_outdir)
    base_out.mkdir(parents=True, exist_ok=True)
    sweep_log = SweepLogger(base_out, CSV_COLUMNS)

    cfg = {
        "num_clusters": args.num_clusters, "bottleneck_dim": args.bottleneck_dim,
        "lr": args.lr, "epochs": args.epochs,
    }
    seeds       = args.seeds
    cluster_csv = _cluster_csv_for(args.adapt_data, cfg["num_clusters"], args.cluster_csv)

    print(f"[multi_seed] prototype_steering | {len(seeds)} seeds | config: {cfg}")
    print(f"[multi_seed] cluster_csv: {cluster_csv}")
    print(f"[multi_seed] output → {sweep_log.path}\n")

    for seed in seeds:
        name    = f"seed{seed:02d}"
        run_dir = base_out / name
        ev_dir  = base_out / f"{name}_eval"

        print(f"\n{'='*60}\n[multi_seed] seed={seed}\n{'='*60}")

        train_cmd = [
            sys.executable, "adaptation/prototype_steering/train.py",
            "--checkpoint",    args.checkpoint,
            "--adapt_data",    args.adapt_data,
            "--cluster_csv",   cluster_csv,
            "--output_dir",    str(run_dir),
            "--device",        args.device,
            "--num_clusters",  str(cfg["num_clusters"]),
            "--bottleneck_dim",str(cfg["bottleneck_dim"]),
            "--lr",            str(cfg["lr"]),
            "--epochs",        str(cfg["epochs"]),
            "--seed",          str(seed),
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
            "--cluster_csv",      cluster_csv,
            "--test_data",        args.test_data,
            "--outdir",           str(ev_dir),
            "--device",           args.device,
            "--num_neg_eval",     str(args.num_neg_eval),
            "--seed",             str(seed),
        ]
        if not run_cmd(eval_cmd, "EVAL"):
            continue

        summary = load_summary(ev_dir / "summary.json")
        if summary is None:
            continue

        row = build_sweep_row(name, {"seed": seed}, summary, key_adapted="adapted_metrics")
        sweep_log.log_run(row)
        print(f"[multi_seed] seed={seed}  NDCG@10 delta={row['ndcg@10_delta']:+.6f}  status={row['status']}")

    print(f"\n[multi_seed] DONE — {len(sweep_log.rows)}/{len(seeds)} seeds completed")
    if sweep_log.rows:
        print_stats(sweep_log.rows)


if __name__ == "__main__":
    main()
