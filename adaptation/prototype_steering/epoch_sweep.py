#!/usr/bin/env python3
"""
adaptation/prototype_steering/epoch_sweep.py

Sweep over epochs [10, 20, 50] with all other HPs fixed to the best config.
(The original sweep only tried epochs=20 as a fixed value — this explores duration.)

Best config (fixed): num_clusters=5, bottleneck_dim=32, lr=1e-3
Swept: epochs in [10, 20, 50]

Usage:
python adaptation/prototype_steering/epoch_sweep.py \
  --checkpoint  results/backbone/sasrec_backbone_best.pt \
  --adapt_data  data/processed/split_10M_contiguous/future_adapt_high_drift_kcore.csv \
  --test_data   data/processed/split_10M_contiguous/future_test_high_drift_kcore.csv \
  --base_outdir results/epoch_sweep_prototype_steering_contiguous --device cuda
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from common.metrics.ranking import METRIC_NAMES
from common.logging.logger import SweepLogger
from sweep_utils import run_cmd, load_summary, build_sweep_row, print_best

# ---------------------------------------------------------------------------
# Fixed best config — only epochs varies
# ---------------------------------------------------------------------------

BEST = {
    "num_clusters":   5,
    "bottleneck_dim": 32,
    "lr":             1e-3,
}

EPOCHS = [10, 20, 50]

CSV_COLUMNS = (
    ["run_id", "epochs", "best_eval_loss"]
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
    p.add_argument("--seed",         type=int, default=42)
    p.add_argument("--num_neg_eval", type=int, default=100)
    p.add_argument("--cluster_csv",  default=None,
                   help="Pre-computed cluster CSV. If omitted, auto-generated from "
                        "user_drift_scores_final_subset.csv in the adapt_data directory.")
    p.add_argument("--epochs",       type=int, nargs="+", default=EPOCHS,
                   help="Epoch values to sweep (default: 10 20 50)")
    return p.parse_args()


def _cluster_csv_for(adapt_data: str, num_clusters: int, cluster_csv: str | None) -> str:
    if cluster_csv:
        return cluster_csv
    return str(Path(adapt_data).parent / f"user_clusters_K{num_clusters}.csv")


def main():
    args     = parse_args()
    base_out = Path(args.base_outdir)
    base_out.mkdir(parents=True, exist_ok=True)
    sweep_log = SweepLogger(base_out, CSV_COLUMNS)

    epoch_list  = args.epochs
    cluster_csv = _cluster_csv_for(args.adapt_data, BEST["num_clusters"], args.cluster_csv)

    print(f"[epoch_sweep] prototype_steering | epochs={epoch_list} | fixed config: {BEST}")
    print(f"[epoch_sweep] cluster_csv: {cluster_csv}")
    print(f"[epoch_sweep] output → {sweep_log.path}\n")

    for run_id, ep in enumerate(epoch_list, 1):
        name    = f"run{run_id:02d}_ep{ep}"
        run_dir = base_out / name
        ev_dir  = base_out / f"{name}_eval"

        print(f"\n{'='*60}\n[epoch_sweep] {run_id}/{len(epoch_list)}: epochs={ep}\n{'='*60}")

        train_cmd = [
            sys.executable, "adaptation/prototype_steering/train.py",
            "--checkpoint",    args.checkpoint,
            "--adapt_data",    args.adapt_data,
            "--cluster_csv",   cluster_csv,
            "--output_dir",    str(run_dir),
            "--device",        args.device,
            "--num_clusters",  str(BEST["num_clusters"]),
            "--bottleneck_dim",str(BEST["bottleneck_dim"]),
            "--lr",            str(BEST["lr"]),
            "--epochs",        str(ep),
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
            "--cluster_csv",      cluster_csv,
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

        row = build_sweep_row(name, {"epochs": ep}, summary, key_adapted="adapted_metrics")
        sweep_log.log_run(row)
        print(f"[epoch_sweep] ep={ep}  NDCG@10 delta={row['ndcg@10_delta']:+.6f}  status={row['status']}")

    print(f"\n[epoch_sweep] DONE — {len(sweep_log.rows)}/{len(epoch_list)} runs")
    print_best(sweep_log.best_run(), METRIC_NAMES)


if __name__ == "__main__":
    main()
