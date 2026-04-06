#!/usr/bin/env python3
"""
data/check_sequential_structure.py

Checks the sequential structure strength of each data subset following the
methodology from:
  Klenitskiy et al., "Does It Look Sequential? An Analysis of Datasets for
  Evaluation of Sequential Recommendations", RecSys '24.

Three checks are applied to each CSV:
  1. Sequential rules  — count 2-grams and 3-grams before/after shuffling
                         user sequences. Large relative drop → strong structure.
  2. Model performance degradation — evaluate a trained SASRec backbone on
                         original vs shuffled test sequences. Large NDCG/HR
                         drop → model relies on sequential patterns.
  3. Top-K Jaccard     — mean Jaccard similarity between top-K recommendation
                         lists on original vs shuffled sequences.
                         Low Jaccard → model is sensitive to order.

Datasets checked per split:
  - Raw future test     data/data_csv/splits/split_10M_*/interactions_future_test.csv
  - Overlap + k-core    results/processed/split_10M_*/future_test_overlap_items_kcore.csv
  - High-drift k-core   results/processed/split_10M_*/future_test_high_drift_kcore.csv

Usage:
python data/check_sequential_structure.py \
  --checkpoint_contiguous results/backbone/sasrec_backbone_best.pt \
  --checkpoint_tail       results/backbone_tail/sasrec_backbone_best.pt \
  --outdir                results/analysis/sequential_structure \
  --device cuda

# Model-agnostic only (no backbone required):
python data/check_sequential_structure.py --rules_only --outdir results/analysis/sequential_structure
"""

import argparse
import random
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sns.set_theme(style="whitegrid", font_scale=1.4)
PALETTE = sns.color_palette("viridis", 3)

# ---------------------------------------------------------------------------
# Dataset definitions
# ---------------------------------------------------------------------------

DATASETS = {
    "contiguous": [
        {
            "label": "Raw future test",
            "path":  "data/data_csv/splits/split_10M_contiguous/interactions_future_test.csv",
        },
        {
            "label": "Overlap + k-core",
            "path":  "results/processed/split_10M_contiguous/future_test_overlap_items_kcore.csv",
        },
        {
            "label": "High-drift k-core",
            "path":  "results/processed/split_10M_contiguous/future_test_high_drift_kcore.csv",
        },
    ],
    "tail": [
        {
            "label": "Raw future test",
            "path":  "data/data_csv/splits/split_10M_tail/interactions_future_test.csv",
        },
        {
            "label": "Overlap + k-core",
            "path":  "results/processed/split_10M_tail/future_test_overlap_items_kcore.csv",
        },
        {
            "label": "High-drift k-core",
            "path":  "results/processed/split_10M_tail/future_test_high_drift_kcore.csv",
        },
    ],
}

N_SHUFFLE_TRIALS = 5   # paper uses 5 trials, average result
SUPPORT_THRESH   = 5
CONFIDENCE_THRESH = 0.1
TOP_K = 10


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def load_seqs(path: str) -> dict[str, list]:
    """Return {user_id: [item_id, ...]} sorted by timestamp."""
    df = pd.read_csv(path)
    # normalise column names
    rename = {}
    if "user" in df.columns: rename["user"] = "user_id"
    if "item" in df.columns: rename["item"] = "item_id"
    df = df.rename(columns=rename)
    df = df.sort_values(["user_id", "timestamp"])
    seqs = {}
    for uid, grp in df.groupby("user_id"):
        seqs[str(uid)] = grp["item_id"].tolist()
    return seqs


def shuffle_seqs(seqs: dict, rng: random.Random) -> dict:
    """Shuffle each user's sequence independently (holdout item excluded — keep last)."""
    shuffled = {}
    for u, seq in seqs.items():
        if len(seq) < 2:
            shuffled[u] = seq
            continue
        context = seq[:-1]
        rng.shuffle(context)
        shuffled[u] = context + [seq[-1]]   # last item stays as holdout
    return shuffled


def seq_stats(seqs: dict) -> dict:
    lengths = [len(s) for s in seqs.values()]
    return {
        "n_users":    len(seqs),
        "n_items":    len({item for s in seqs.values() for item in s}),
        "avg_len":    float(np.mean(lengths)),
        "median_len": float(np.median(lengths)),
        "min_len":    int(np.min(lengths)),
        "max_len":    int(np.max(lengths)),
    }


# ---------------------------------------------------------------------------
# Check 1: Sequential rules (2-grams and 3-grams)
# ---------------------------------------------------------------------------

def count_ngrams(seqs: dict, n: int, support: int, confidence: float) -> int:
    """Count n-grams with given support and confidence thresholds."""
    # count occurrences of each n-gram and the prefix (n-1)-gram
    ngram_counts: dict = defaultdict(int)
    prefix_counts: dict = defaultdict(int)

    for seq in seqs.values():
        items = seq
        for i in range(len(items) - n + 1):
            gram  = tuple(items[i:i+n])
            prefix = gram[:-1]
            ngram_counts[gram]   += 1
            prefix_counts[prefix] += 1

    rules = 0
    for gram, cnt in ngram_counts.items():
        if cnt < support:
            continue
        prefix = gram[:-1]
        conf = cnt / prefix_counts[prefix] if prefix_counts[prefix] > 0 else 0
        if conf >= confidence:
            rules += 1
    return rules


def check_sequential_rules(seqs: dict, n_trials: int = N_SHUFFLE_TRIALS,
                            seed: int = 42) -> dict:
    rng = random.Random(seed)

    orig_2 = count_ngrams(seqs, 2, SUPPORT_THRESH, CONFIDENCE_THRESH)
    orig_3 = count_ngrams(seqs, 3, SUPPORT_THRESH, CONFIDENCE_THRESH)

    shuf_2_list, shuf_3_list = [], []
    for _ in range(n_trials):
        sh = shuffle_seqs(seqs, rng)
        shuf_2_list.append(count_ngrams(sh, 2, SUPPORT_THRESH, CONFIDENCE_THRESH))
        shuf_3_list.append(count_ngrams(sh, 3, SUPPORT_THRESH, CONFIDENCE_THRESH))

    shuf_2 = float(np.mean(shuf_2_list))
    shuf_3 = float(np.mean(shuf_3_list))

    rel_2 = (shuf_2 - orig_2) / orig_2 * 100 if orig_2 > 0 else float("nan")
    rel_3 = (shuf_3 - orig_3) / orig_3 * 100 if orig_3 > 0 else float("nan")

    return {
        "2gram_orig":    orig_2,
        "2gram_shuffled": round(shuf_2, 1),
        "2gram_rel_pct":  round(rel_2, 1),
        "3gram_orig":    orig_3,
        "3gram_shuffled": round(shuf_3, 1),
        "3gram_rel_pct":  round(rel_3, 1),
    }


# ---------------------------------------------------------------------------
# Check 2 & 3: Model-based (needs backbone)
# ---------------------------------------------------------------------------

def model_checks(seqs: dict, backbone, le_user, le_item, cfg: dict,
                 itemnum: int, device: str, n_trials: int = N_SHUFFLE_TRIALS,
                 seed: int = 42) -> dict:
    """Evaluate backbone on original and shuffled test sequences."""
    import torch
    from common.data_utils import build_input_tensor, sample_negatives
    from common.metrics.ranking import metrics_from_rank, summarize

    maxlen  = cfg["maxlen"]
    rng     = random.Random(seed)
    known_users = set(le_user.classes_.tolist())
    known_items = set(le_item.classes_.tolist())

    # filter seqs to known users/items, keep only seq >= 2
    enc_seqs = {}
    for u, seq in seqs.items():
        if str(u) not in known_users:
            continue
        enc_seq = []
        for it in seq:
            if str(it) in known_items:
                enc_seq.append(int(le_item.transform([str(it)])[0]) + 1)  # 1-indexed
        if len(enc_seq) >= 2:
            uid_enc = int(le_user.transform([str(u)])[0])
            enc_seqs[uid_enc] = enc_seq

    if not enc_seqs:
        return {}

    def eval_seqs(seqs_dict):
        rows = []
        top_k_lists = {}
        backbone.eval()
        with torch.no_grad():
            for uid, seq in seqs_dict.items():
                context = seq[:-1]
                target  = seq[-1]
                seen    = set(context) | {0}
                negs    = sample_negatives(itemnum, seen, target, n_neg=100)
                cands   = [target] + negs
                ids     = build_input_tensor(context, maxlen, device)
                cand_t  = torch.tensor([cands], dtype=torch.long, device=device)
                logits  = backbone.score_from_hidden(backbone.get_last_hidden(ids), cand_t)[0]
                rank    = int((logits > logits[0]).sum().item())
                rows.append({"user_idx": uid, "rank": rank, **metrics_from_rank(rank)})
                # top-K item indices for Jaccard
                top_k_lists[uid] = set(
                    torch.argsort(logits, descending=True)[:TOP_K].cpu().tolist()
                )
        return summarize(pd.DataFrame(rows)), top_k_lists

    # original
    orig_metrics, orig_topk = eval_seqs(enc_seqs)

    # shuffled (average over trials)
    shuf_metrics_list = []
    jaccard_list      = []
    for _ in range(n_trials):
        sh_seqs = {}
        for uid, seq in enc_seqs.items():
            ctx = seq[:-1].copy()
            rng.shuffle(ctx)
            sh_seqs[uid] = ctx + [seq[-1]]
        sh_metrics, sh_topk = eval_seqs(sh_seqs)
        shuf_metrics_list.append(sh_metrics)

        # Jaccard per user
        jacc = []
        for uid in orig_topk:
            if uid in sh_topk:
                inter = len(orig_topk[uid] & sh_topk[uid])
                union = len(orig_topk[uid] | sh_topk[uid])
                jacc.append(inter / union if union > 0 else 1.0)
        jaccard_list.append(float(np.mean(jacc)) if jacc else float("nan"))

    # average shuffled metrics
    metric_keys = ["ndcg@10", "hr@10", "mrr@10", "ndcg@20", "hr@20"]
    avg_shuf = {k: float(np.mean([m[k] for m in shuf_metrics_list])) for k in metric_keys}
    avg_jacc = float(np.mean(jaccard_list))

    result = {"n_users_eval": len(enc_seqs)}
    for k in metric_keys:
        orig_v = orig_metrics[k]
        shuf_v = avg_shuf[k]
        rel    = (shuf_v - orig_v) / orig_v * 100 if orig_v != 0 else float("nan")
        result[f"{k}_orig"]    = round(orig_v, 4)
        result[f"{k}_shuffled"] = round(shuf_v, 4)
        result[f"{k}_rel_pct"]  = round(rel, 2)
    result[f"jaccard@{TOP_K}_orig_vs_shuffled"] = round(avg_jacc, 4)

    return result


# ---------------------------------------------------------------------------
# Printing
# ---------------------------------------------------------------------------

def _verdict(pct2: float, pct3: float) -> str:
    avg = (abs(pct2) + abs(pct3)) / 2
    if avg >= 90:  return "Strong ✓"
    if avg >= 75:  return "Moderate"
    return "Weak ⚠"


def print_rules_table(all_rows: list[dict]):
    print(f"\n{'='*90}")
    print("  CHECK 1: SEQUENTIAL RULES  (relative change after shuffling)")
    print(f"  Threshold: support≥{SUPPORT_THRESH}, confidence≥{CONFIDENCE_THRESH}"
          f"  |  {N_SHUFFLE_TRIALS} shuffle trials averaged")
    print(f"  Interpretation: closer to -100% = stronger sequential structure")
    print(f"{'='*90}")

    # --- detailed table ---
    print(f"\n  {'Split':<12}  {'Dataset':<22}  {'Users':>6}  {'AvgLen':>7}"
          f"  {'2g orig':>8}  {'2g shuf':>8}  {'2g %Δ':>8}"
          f"  {'3g orig':>8}  {'3g shuf':>8}  {'3g %Δ':>8}")
    print(f"  {'-'*105}")
    for r in all_rows:
        s  = r.get("stats", {})
        ru = r.get("rules", {})
        flag2 = " ◄" if abs(ru.get("2gram_rel_pct", 0)) > 90 else ""
        flag3 = " ◄" if abs(ru.get("3gram_rel_pct", 0)) > 90 else ""
        print(f"  {r['split']:<12}  {r['label']:<22}  {s.get('n_users',0):>6,}"
              f"  {s.get('avg_len',0):>7.1f}"
              f"  {ru.get('2gram_orig',0):>8,}  {ru.get('2gram_shuffled',0):>8,}"
              f"  {ru.get('2gram_rel_pct',float('nan')):>+7.1f}%{flag2}"
              f"  {ru.get('3gram_orig',0):>8,}  {ru.get('3gram_shuffled',0):>8,}"
              f"  {ru.get('3gram_rel_pct',float('nan')):>+7.1f}%{flag3}")
    print(f"  ◄ = relative change > 90% (strong sequential structure per paper)")

    # --- verdict summary ---
    print(f"\n  SUMMARY")
    print(f"  {'Split':<12}  {'Dataset':<22}  {'AvgLen':>7}"
          f"  {'2-gram %Δ':>10}  {'3-gram %Δ':>10}  {'Verdict':<12}")
    print(f"  {'-'*80}")
    for r in all_rows:
        s  = r.get("stats", {})
        ru = r.get("rules", {})
        p2 = ru.get("2gram_rel_pct", float("nan"))
        p3 = ru.get("3gram_rel_pct", float("nan"))
        verdict = _verdict(p2, p3) if not (np.isnan(p2) or np.isnan(p3)) else "N/A"
        print(f"  {r['split']:<12}  {r['label']:<22}  {s.get('avg_len',0):>7.1f}"
              f"  {p2:>+9.1f}%  {p3:>+9.1f}%  {verdict:<12}")
    print()


def print_model_table(all_rows: list[dict]):
    has_model = any(r.get("model") for r in all_rows)
    if not has_model:
        return
    print(f"\n{'='*90}")
    print("  CHECK 2 & 3: MODEL-BASED  (SASRec backbone on original vs shuffled test seqs)")
    print(f"  Interpretation: large NDCG/HR drop = strong structure; "
          f"low Jaccard = order-sensitive")
    print(f"{'='*90}")
    print(f"  {'Split':<12}  {'Dataset':<22}  {'N':>5}  {'AvgLen':>7}"
          f"  {'NDCG@10 orig':>12}  {'NDCG@10 shuf':>12}  {'NDCG %Δ':>8}"
          f"  {'HR@10 %Δ':>9}  {'Jacc@10':>8}")
    print(f"  {'-'*108}")
    for r in all_rows:
        m = r.get("model", {})
        if not m:
            continue
        s    = r.get("stats", {})
        flag = " ◄" if abs(m.get("ndcg@10_rel_pct", 0)) > 10 else ""
        print(f"  {r['split']:<12}  {r['label']:<22}  {m.get('n_users_eval',0):>5,}"
              f"  {s.get('avg_len', float('nan')):>7.1f}"
              f"  {m.get('ndcg@10_orig', float('nan')):>12.4f}"
              f"  {m.get('ndcg@10_shuffled', float('nan')):>12.4f}"
              f"  {m.get('ndcg@10_rel_pct', float('nan')):>+7.2f}%{flag}"
              f"  {m.get('hr@10_rel_pct', float('nan')):>+8.2f}%"
              f"  {m.get(f'jaccard@{TOP_K}_orig_vs_shuffled', float('nan')):>8.4f}")
    print(f"  ◄ = NDCG@10 drop > 10% (weak sequential structure signal per paper)")
    print()


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_rules(all_rows: list[dict], outdir: Path):
    splits  = list(dict.fromkeys(r["split"] for r in all_rows))
    labels  = list(dict.fromkeys(r["label"] for r in all_rows))
    colors  = {l: PALETTE[i] for i, l in enumerate(labels)}

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, gram in zip(axes, ["2gram", "3gram"]):
        for split in splits:
            sub = [r for r in all_rows if r["split"] == split]
            x   = [r["label"] for r in sub]
            y   = [r["rules"].get(f"{gram}_rel_pct", float("nan")) for r in sub]
            ax.plot(x, y, marker="o", label=split, linewidth=2, markersize=8)

        ax.axhline(-90, color="red", linewidth=1, linestyle="--", alpha=0.6,
                   label="-90% threshold (strong)")
        ax.axhline(0,   color="black", linewidth=0.8, linestyle=":")
        ax.set_title(f"{gram.replace('gram', '-gram')} relative change after shuffling",
                     fontsize=13, fontweight="bold")
        ax.set_ylabel("Relative % change")
        ax.set_xlabel("Dataset subset")
        ax.legend(fontsize=10)
        ax.tick_params(axis="x", rotation=15)

    fig.suptitle("Sequential Rules Check — % Change After User Sequence Shuffling",
                 fontsize=14, fontweight="bold")
    fig.tight_layout()
    out = outdir / "seq_structure_rules.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


def plot_model_degradation(all_rows: list[dict], outdir: Path):
    model_rows = [r for r in all_rows if r.get("model")]
    if not model_rows:
        return

    metrics = ["ndcg@10_rel_pct", "hr@10_rel_pct", "mrr@10_rel_pct"]
    labels  = [r["label"] for r in model_rows]
    x_base  = [f"{r['split']}\n{r['label']}" for r in model_rows]
    x       = np.arange(len(model_rows))
    width   = 0.25
    colors  = sns.color_palette("viridis", len(metrics))

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # left: metric degradation
    ax = axes[0]
    for i, (m, label) in enumerate(zip(metrics, ["NDCG@10", "HR@10", "MRR@10"])):
        vals = [r["model"].get(m, float("nan")) for r in model_rows]
        ax.bar(x + i * width, vals, width, label=label, color=colors[i], alpha=0.85)
    ax.axhline(-10, color="red", linewidth=1, linestyle="--", alpha=0.7,
               label="-10% threshold")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x + width)
    ax.set_xticklabels(x_base, fontsize=10, rotation=15, ha="right")
    ax.set_ylabel("Relative % change after shuffling")
    ax.set_title("Model Performance Degradation", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)

    # right: Jaccard
    ax = axes[1]
    jacc_key = f"jaccard@{TOP_K}_orig_vs_shuffled"
    jacc_vals = [r["model"].get(jacc_key, float("nan")) for r in model_rows]
    bar_colors = [PALETTE[i % len(PALETTE)] for i in range(len(model_rows))]
    bars = ax.bar(x_base, jacc_vals, color=bar_colors, alpha=0.85)
    ax.axhline(1/3, color="red", linewidth=1, linestyle="--", alpha=0.7,
               label="1/3 threshold (paper: weak if above)")
    ax.set_ylabel(f"Mean Jaccard@{TOP_K}")
    ax.set_title(f"Top-{TOP_K} Jaccard Similarity (orig vs shuffled)",
                 fontsize=13, fontweight="bold")
    ax.set_xticks(range(len(x_base)))
    ax.set_xticklabels(x_base, fontsize=10, rotation=15, ha="right")
    ax.legend(fontsize=10)
    for bar, val in zip(bars, jacc_vals):
        if not np.isnan(val):
            ax.text(bar.get_x() + bar.get_width()/2, val + 0.005,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=10)

    fig.suptitle("Model-Based Sequential Structure Checks (SASRec backbone)",
                 fontsize=14, fontweight="bold")
    fig.tight_layout()
    out = outdir / "seq_structure_model.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint_contiguous", default=None,
                   help="Backbone checkpoint for contiguous split")
    p.add_argument("--checkpoint_tail",       default=None,
                   help="Backbone checkpoint for tail split")
    p.add_argument("--outdir",  default="results/analysis/sequential_structure")
    p.add_argument("--device",  default="cuda")
    p.add_argument("--rules_only", action="store_true",
                   help="Only run sequential rules check (no backbone required)")
    p.add_argument("--n_trials", type=int, default=N_SHUFFLE_TRIALS,
                   help="Number of shuffle trials to average (default: 5)")
    p.add_argument("--seed",     type=int, default=42)
    return p.parse_args()


def main():
    args   = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # optionally load backbones
    backbones = {}
    if not args.rules_only:
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
        from backbone.model import SASRec
        from common.data_utils import load_checkpoint, make_encoders_from_ckpt
        import torch

        for split, ckpt_path in [("contiguous", args.checkpoint_contiguous),
                                  ("tail",       args.checkpoint_tail)]:
            if ckpt_path and Path(ckpt_path).exists():
                print(f"[structure] loading backbone for {split}: {ckpt_path}")
                ckpt = load_checkpoint(ckpt_path, args.device)
                cfg  = ckpt["config"]
                model = SASRec(
                    item_num=ckpt["itemnum"], maxlen=cfg["maxlen"],
                    hidden_units=cfg["hidden_units"], num_blocks=cfg["num_blocks"],
                    num_heads=cfg["num_heads"], dropout_rate=cfg["dropout_rate"],
                    add_head=False, pos_enc=True,
                ).to(args.device)
                model.load_state_dict(ckpt["model_state_dict"])
                model.eval()
                le_user, le_item = make_encoders_from_ckpt(ckpt)
                backbones[split] = (model, le_user, le_item, cfg, ckpt["itemnum"])
            else:
                if not args.rules_only:
                    print(f"[structure] no backbone for {split} — skipping model checks")

    all_rows = []

    for split, datasets in DATASETS.items():
        backbone_data = backbones.get(split)

        for ds in datasets:
            path = ds["path"]
            if not Path(path).exists():
                print(f"  [skip] not found: {path}")
                continue

            print(f"\n[structure] {split} / {ds['label']}  ← {path}")
            seqs  = load_seqs(path)
            stats = seq_stats(seqs)
            print(f"  users={stats['n_users']:,}  items={stats['n_items']:,}"
                  f"  avg_len={stats['avg_len']:.1f}")

            # Check 1: sequential rules
            print(f"  running sequential rules ({args.n_trials} trials)...")
            rules = check_sequential_rules(seqs, n_trials=args.n_trials, seed=args.seed)
            print(f"  2-gram: {rules['2gram_orig']} → {rules['2gram_shuffled']}"
                  f"  ({rules['2gram_rel_pct']:+.1f}%)")
            print(f"  3-gram: {rules['3gram_orig']} → {rules['3gram_shuffled']}"
                  f"  ({rules['3gram_rel_pct']:+.1f}%)")

            # Checks 2 & 3: model-based
            model_result = {}
            if backbone_data and not args.rules_only:
                model, le_user, le_item, cfg, itemnum = backbone_data
                print(f"  running model checks ({args.n_trials} trials)...")
                model_result = model_checks(
                    seqs, model, le_user, le_item, cfg, itemnum,
                    args.device, n_trials=args.n_trials, seed=args.seed,
                )
                if model_result:
                    print(f"  NDCG@10: {model_result['ndcg@10_orig']:.4f} → "
                          f"{model_result['ndcg@10_shuffled']:.4f}"
                          f"  ({model_result['ndcg@10_rel_pct']:+.2f}%)"
                          f"  Jaccard@{TOP_K}={model_result[f'jaccard@{TOP_K}_orig_vs_shuffled']:.4f}")

            all_rows.append({
                "split":  split,
                "label":  ds["label"],
                "path":   path,
                "stats":  stats,
                "rules":  rules,
                "model":  model_result,
            })

    # print tables
    print_rules_table(all_rows)
    print_model_table(all_rows)

    # save CSV
    flat = []
    for r in all_rows:
        row = {"split": r["split"], "label": r["label"], "path": r["path"]}
        row.update({f"stats_{k}": v for k, v in r["stats"].items()})
        row.update({f"rules_{k}": v for k, v in r["rules"].items()})
        row.update({f"model_{k}": v for k, v in r["model"].items()})
        flat.append(row)
    df_out = pd.DataFrame(flat)
    csv_out = outdir / "sequential_structure.csv"
    df_out.to_csv(csv_out, index=False)
    print(f"  Saved: {csv_out}")

    # plots
    plot_rules(all_rows, outdir)
    plot_model_degradation(all_rows, outdir)

    print(f"\n[structure] Done — outputs in {outdir}/")


if __name__ == "__main__":
    main()
