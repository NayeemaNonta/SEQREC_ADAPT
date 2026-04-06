#!/usr/bin/env python3
"""
run_comparison.py

Runs all adaptation methods + full backbone fine-tuning with best configs,
reports peak GPU memory and wall time for each, and produces a comparison table.

Methods run:
  1. Full backbone FT       (upper bound)
  2. Last Block FT          (best: lr=5e-4, ep=50, ln=True, wd=1e-4)
  3. Context Gate           (best: bd=8, lr=5e-4, ep=10)
  4. Context Gate (no gate) (ablation: plain residual MLP, same HPs)
  5. Prototype Steering     (best: k=5, bd=32, lr=1e-3, ep=20)

Outputs in --outdir:
  comparison_table.csv   — metrics + memory + wall time per method
  comparison_table.txt   — human-readable version

Usage:
python run_comparison.py \
  --checkpoint  results/backbone/sasrec_backbone_best.pt \
  --adapt_data  data/processed/split_10M_contiguous/future_adapt_high_drift_kcore.csv \
  --test_data   data/processed/split_10M_contiguous/future_test_high_drift_kcore.csv \
  --outdir      results/comparison_contiguous \
  --device cuda
"""

import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Best configs (from HP sweeps)
# ---------------------------------------------------------------------------

CONFIGS = [
    {
        "name":        "Full FT",
        "script":      "backbone/finetune_full.py",
        "extra_args":  ["--lr", "1e-4", "--epochs", "20", "--weight_decay", "1e-4"],
        "run_dir":     "full_ft",
        "ckpt_name":   None,           # no separate adapt ckpt; script does eval itself
        "eval_script": None,
        "summary_key": "finetuned_metrics",
    },
    {
        "name":        "Last Block FT",
        "script":      "adaptation/last_block/train.py",
        "extra_args":  ["--lr", "5e-4", "--epochs", "50",
                        "--weight_decay", "1e-4", "--include_last_layernorm"],
        "run_dir":     "last_block",
        "ckpt_name":   "last_block_best.pt",
        "eval_script": "adaptation/last_block/eval.py",
        "eval_flag":   "--ft_checkpoint",
        "summary_key": "finetuned_metrics",
    },
    {
        "name":        "Context Gate",
        "script":      "adaptation/context_steering/train.py",
        "extra_args":  ["--bottleneck_dim", "8", "--lr", "5e-4", "--epochs", "10"],
        "run_dir":     "context_gate",
        "ckpt_name":   "context_gate_best.pt",
        "eval_script": "adaptation/context_steering/eval.py",
        "eval_flag":   "--adapt_checkpoint",
        "summary_key": "adapted_metrics",
    },
    {
        "name":        "Context Gate (no gate)",
        "script":      "adaptation/context_steering/train.py",
        "extra_args":  ["--bottleneck_dim", "8", "--lr", "5e-4", "--epochs", "10", "--no_gate"],
        "run_dir":     "context_gate_no_gate",
        "ckpt_name":   "context_gate_best.pt",
        "eval_script": "adaptation/context_steering/eval.py",
        "eval_flag":   "--adapt_checkpoint",
        "summary_key": "adapted_metrics",
    },
    {
        "name":        "Prototype Steering",
        "script":      "adaptation/prototype_steering/train.py",
        "extra_args":  ["--num_clusters", "5", "--bottleneck_dim", "32",
                        "--lr", "1e-3", "--epochs", "20"],
        "run_dir":     "prototype_steering",
        "ckpt_name":   "prototype_best.pt",
        "eval_script": "adaptation/prototype_steering/eval.py",
        "eval_flag":   "--adapt_checkpoint",
        "summary_key": "adapted_metrics",
        "needs_cluster_csv": True,
    },
]

REPORT_METRICS = ["ndcg@10", "hr@10", "mrr@10", "ndcg@20", "hr@20"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def run_and_capture(cmd: list, label: str) -> tuple[bool, str]:
    """Run subprocess, stream output live, and return captured stdout+stderr."""
    print(f"\n[comparison] >>> {label}")
    t0 = time.time()
    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
    )
    lines = []
    for line in proc.stdout:
        print(line, end="")
        lines.append(line)
    proc.wait()
    elapsed = time.time() - t0
    output  = "".join(lines)
    ok      = proc.returncode == 0
    print(f"[comparison] {'OK' if ok else 'FAILED'} in {elapsed:.1f}s")
    return ok, output


def parse_memory(output: str) -> float | None:
    """Extract peak GPU mem from stdout: 'peak GPU mem: 247.1 MB'"""
    m = re.search(r"peak GPU mem:\s*([\d.]+)\s*MB", output)
    return float(m.group(1)) if m else None


def parse_wall_time(output: str) -> float | None:
    """Extract wall time from stdout: 'total wall time: 9.7s'"""
    m = re.search(r"total wall time:\s*([\d.]+)s", output)
    return float(m.group(1)) if m else None


def parse_trainable_params(output: str) -> int | None:
    """Extract trainable param count from stdout: 'trainable params=8,448'"""
    m = re.search(r"trainable params[=:]\s*([\d,]+)", output)
    return int(m.group(1).replace(",", "")) if m else None


def load_summary(path: Path) -> dict | None:
    if not path.exists():
        print(f"  [warning] summary not found: {path}")
        return None
    with open(path) as f:
        return json.load(f)


def cluster_csv_for(adapt_data: str, num_clusters: int = 5) -> str:
    return str(Path(adapt_data).parent / f"user_clusters_K{num_clusters}.csv")


def count_params(summary: dict) -> int | None:
    return summary.get("trainable_params")


# ---------------------------------------------------------------------------
# Table formatting
# ---------------------------------------------------------------------------

def build_row(cfg: dict, summary: dict, peak_mem: float | None, wall_time: float | None,
              trainable_params: int | None = None) -> dict:
    ev   = summary.get("eval", summary)   # full_ft nests under "eval"
    base = ev.get("baseline_metrics", {})
    adpt = ev.get(cfg["summary_key"], {})

    # prefer parsed-from-stdout value; fall back to summary.json
    tp = (trainable_params
          or summary.get("trainable_params")
          or ev.get("trainable_params")
          or "")

    row = {
        "method":           cfg["name"],
        "trainable_params": tp,
        "peak_gpu_mb":      round(peak_mem, 1) if peak_mem is not None else "",
        "wall_time_s":      round(wall_time, 1) if wall_time is not None else "",
    }
    for m in REPORT_METRICS:
        b  = float(base.get(m, float("nan")))
        a  = float(adpt.get(m, float("nan")))
        pct = (a - b) / b * 100 if b != 0 else float("nan")
        row[f"{m}_baseline"]   = round(b,   4)
        row[f"{m}_adapted"]    = round(a,   4)
        row[f"{m}_pct_change"] = round(pct, 2)
    return row


def print_table(rows: list[dict]):
    print(f"\n{'='*90}")
    print("  COMPARISON TABLE")
    print(f"{'='*90}")
    header = f"  {'Method':<28}  {'Params':>8}  {'GPU MB':>7}  {'Time(s)':>8}"
    for m in ["ndcg@10", "hr@10", "mrr@10"]:
        header += f"  {m+'_%Δ':>10}"
    print(header)
    print(f"  {'-'*86}")
    for r in rows:
        line = (
            f"  {r['method']:<28}  {str(r['trainable_params']):>8}  "
            f"{str(r['peak_gpu_mb']):>7}  {str(r['wall_time_s']):>8}"
        )
        for m in ["ndcg@10", "hr@10", "mrr@10"]:
            v = r.get(f"{m}_pct_change", "")
            line += f"  {(f'{v:+.2f}%' if isinstance(v, float) else str(v)):>10}"
        print(line)
    print()


def save_table(rows: list[dict], outdir: Path):
    import csv
    if not rows:
        return
    csv_path = outdir / "comparison_table.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    txt_path = outdir / "comparison_table.txt"
    with open(txt_path, "w") as f:
        cols = ["method", "trainable_params", "peak_gpu_mb", "wall_time_s"]
        for m in REPORT_METRICS:
            cols += [f"{m}_baseline", f"{m}_adapted", f"{m}_pct_change"]
        widths = {c: max(len(c), max((len(str(r.get(c, ""))) for r in rows), default=0))
                  for c in cols}
        header = "  ".join(str(c).ljust(widths[c]) for c in cols)
        f.write(header + "\n")
        f.write("-" * len(header) + "\n")
        for r in rows:
            f.write("  ".join(str(r.get(c, "")).ljust(widths[c]) for c in cols) + "\n")

    print(f"  Saved: {csv_path}")
    print(f"  Saved: {txt_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint",   required=True)
    p.add_argument("--adapt_data",   required=True)
    p.add_argument("--test_data",    required=True)
    p.add_argument("--outdir",       required=True)
    p.add_argument("--device",       default="cuda")
    p.add_argument("--seed",         type=int, default=42)
    p.add_argument("--num_neg_eval", type=int, default=100)
    p.add_argument("--cluster_csv",  default=None,
                   help="Cluster CSV for prototype steering. "
                        "If omitted, auto-derived from adapt_data directory.")
    p.add_argument("--skip",         nargs="*", default=[],
                   help="Method run_dirs to skip, e.g. --skip full_ft prototype_steering")
    return p.parse_args()


def main():
    args    = parse_args()
    outdir  = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    cluster_csv = args.cluster_csv or cluster_csv_for(args.adapt_data)
    rows = []

    for cfg in CONFIGS:
        if cfg["run_dir"] in args.skip:
            print(f"\n[comparison] SKIPPING {cfg['name']}")
            continue

        run_dir = outdir / cfg["run_dir"]
        run_dir.mkdir(parents=True, exist_ok=True)
        peak_mem         = None
        wall_time        = None
        trainable_params = None

        # ---- train ----
        train_cmd = [
            sys.executable, cfg["script"],
            "--checkpoint", args.checkpoint,
            "--adapt_data", args.adapt_data,
            "--output_dir", str(run_dir),
            "--device",     args.device,
            "--seed",       str(args.seed),
        ] + cfg["extra_args"]

        if cfg.get("needs_cluster_csv"):
            train_cmd += ["--cluster_csv", cluster_csv]

        # full_ft also needs test_data (does eval inline)
        if cfg["eval_script"] is None:
            train_cmd += ["--test_data", args.test_data,
                          "--num_neg_eval", str(args.num_neg_eval)]

        ok, output = run_and_capture(train_cmd, f"TRAIN {cfg['name']}")
        trainable_params = None
        if ok:
            peak_mem         = parse_memory(output)
            wall_time        = parse_wall_time(output)
            trainable_params = parse_trainable_params(output)

        # ---- eval (separate script methods) ----
        if cfg["eval_script"] is not None:
            adapt_ckpt = run_dir / cfg["ckpt_name"]
            if not adapt_ckpt.exists():
                print(f"  [warning] adapter ckpt not found: {adapt_ckpt}")
                continue

            ev_dir = outdir / f"{cfg['run_dir']}_eval"
            ev_dir.mkdir(parents=True, exist_ok=True)

            eval_cmd = [
                sys.executable, cfg["eval_script"],
                "--checkpoint",      args.checkpoint,
                cfg["eval_flag"],    str(adapt_ckpt),
                "--test_data",       args.test_data,
                "--outdir",          str(ev_dir),
                "--device",          args.device,
                "--num_neg_eval",    str(args.num_neg_eval),
                "--seed",            str(args.seed),
            ]
            if cfg.get("needs_cluster_csv"):
                eval_cmd += ["--cluster_csv", cluster_csv]

            ok, _ = run_and_capture(eval_cmd, f"EVAL {cfg['name']}")
            summary_path = ev_dir / "summary.json"
        else:
            summary_path = run_dir / "summary.json"

        # ---- collect results ----
        summary = load_summary(summary_path)
        if summary is None:
            continue

        # for methods without inline memory reporting, try reading from summary
        if peak_mem is None:
            peak_mem  = summary.get("peak_gpu_mb")
        if wall_time is None:
            wall_time = summary.get("wall_time_s")

        row = build_row(cfg, summary, peak_mem, wall_time, trainable_params)
        rows.append(row)
        print(f"  [{cfg['name']}] NDCG@10 %Δ = {row.get('ndcg@10_pct_change', 'N/A')}")

    print_table(rows)
    save_table(rows, outdir)
    print(f"\n[comparison] DONE — {len(rows)}/{len(CONFIGS)} methods completed")


if __name__ == "__main__":
    main()
