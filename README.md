# Lightweight Adaptation for Sequential Recommenders

Research repository studying post-deployment adaptation of sequential recommenders under temporal and preference drift. A single pretrained SASRec backbone is kept frozen (or partially frozen) while multiple lightweight adaptation mechanisms are applied and compared under a consistent evaluation regime.

## Repository Structure

```
seqrec_adapt/
├── data/
│   ├── create_dataset/       Dataset creation pipeline (time-based splits)
│   └── preprocessing/        Drift-based user filtering, k-core, overlap items
│
├── backbone/
│   ├── model.py              SASRec implementation
│   ├── train_backbone.py     Pre-deployment backbone training (T1)
│   └── configs/
│
├── adaptation/
│   ├── last_block/           Weight-space: fine-tune final transformer block
│   ├── context_steering/     Activation-space: context-conditioned scalar gate
│   └── prototype_steering/   Activation-space: cluster-specific residual MLPs
│
├── common/
│   ├── data_utils.py         Shared I/O, encoding, sequence utilities
│   ├── evaluation/           Shared evaluation protocol (apples-to-apples)
│   ├── metrics/              NDCG@K, HR@K, MRR@K
│   ├── logging/              ExperimentLogger, SweepLogger
│   └── memory_profiler/      GPU memory and wall-time tracking
│
├── configs/                  Global experiment configuration
└── sweep_utils.py            Shared sweep helpers
```

## Adaptation Modes

| Mode | Location | Trainable Params | Backbone Frozen | Requires Clustering |
|---|---|---|---|---|
| Last-Block FT | `adaptation/last_block/` | ~25K (last block) | Partially | No |
| Context Gate | `adaptation/context_steering/` | ~1.2K (shared adapter) | Fully | No |
| Prototype Steering | `adaptation/prototype_steering/` | K × ~4K (per cluster) | Fully | Yes |

All modes share the same:
- Evaluation protocol (`common/evaluation/evaluator.py`)
- Metric definitions (`common/metrics/ranking.py`)
- Candidate set construction (1 target + 100 negatives, fixed seed)
- Backbone checkpoint and item/user encoders

## Dataset

This project uses the **ThirtyMusic** dataset — approximately 30 million music listening events collected from Last.fm via the [Idomaar](https://github.com/D2KLab/idomaar) framework.

**Download:** [https://recsys.deib.polimi.it/datasets/](https://recsys.deib.polimi.it/datasets/)  
Direct link to the dataset page: search for *"ThirtyMusic"* on the RecSys Polimi datasets page, or use the Idomaar repository linked above.

Once downloaded, place the raw file at:

```
data/ThirtyMusic/relations/events.idomaar
```

Then convert it to the CSV format expected by this pipeline:

```bash
python data/create_dataset/prepare_30music_csv.py
```

This produces `data/data_csv/30M.csv` with columns `user, item, timestamp`. No filtering is applied at this stage — k-core filtering and preprocessing happen in the pipeline steps below.

> The conversion filters out events with `playtime < 1` second. Approximately 30M rows are retained.

## Quickstart

### 1. Build dataset splits

Produces two temporal splits from the 30M interaction CSV (default: 10M total, 5M hist + 5M future):

```bash
python data/create_dataset/create_adaptation_split.py \
  --src data/data_csv/30M.csv

# To use a different total size (e.g. 5M):
python data/create_dataset/create_adaptation_split.py \
  --src data/data_csv/30M.csv --total 5000000
```

Outputs two directories under `data/data_csv/splits/`:
- `split_10M_contiguous/` — hist=first 5M, future=rows 5M–10M (immediately after)
- `split_10M_tail/` — hist=first 5M, future=last 5M of the full 30M dataset

Each contains: `interactions_hist.csv`, `interactions_future.csv`, `interactions_future_adapt.csv` (70%), `interactions_future_test.csv` (30%), `split_metadata.json`.

### 2. Run preprocessing pipeline

Applies k-core filtering, detects high-drift users, and produces final train/adapt/test subsets. Run once per split:

```bash
# Split A — contiguous
python data/preprocessing/run_pipeline.py \
  --hist_data         data/data_csv/splits/split_10M_contiguous/interactions_hist.csv \
  --future_adapt_data data/data_csv/splits/split_10M_contiguous/interactions_future_adapt.csv \
  --future_test_data  data/data_csv/splits/split_10M_contiguous/interactions_future_test.csv \
  --backbone_ckpt     results/backbone_contiguous/sasrec_backbone_best.pt \
  --output_dir        data/processed/split_10M_contiguous \
  --device cuda

# Split B — tail
python data/preprocessing/run_pipeline.py \
  --hist_data         data/data_csv/splits/split_10M_tail/interactions_hist.csv \
  --future_adapt_data data/data_csv/splits/split_10M_tail/interactions_future_adapt.csv \
  --future_test_data  data/data_csv/splits/split_10M_tail/interactions_future_test.csv \
  --backbone_ckpt     results/backbone_tail/sasrec_backbone_best.pt \
  --output_dir        data/processed/split_10M_tail \
  --device cuda
```

Pipeline steps (run in this order automatically):
1. `filter_to_overlap_items_kcore.py` — k-core on hist; restrict future files to surviving items
2. `detect_high_drift_users_overlap.py` — score users by preference drift using the backbone
3. `filter_to_selected_users_kcore.py` — filter to high-drift users and re-apply k-core
4. `build_final_drift_scores.py` — attach user indices to final drift scores

Outputs per split (in `data/processed/split_10M_*/`):
- `hist_high_drift_kcore.csv` — backbone training data
- `future_adapt_high_drift_kcore.csv` — adaptation data
- `future_test_high_drift_kcore.csv` — evaluation data

### 3. Train backbone (T1)

The backbone is trained automatically as step 2 of the preprocessing pipeline above. The commands below are for re-training or fine-tuning independently:

```bash
# Split A
python backbone/train_backbone.py \
  --hist_data  data/processed/split_10M_contiguous/hist_overlap_items_kcore.csv \
  --val_data   data/processed/split_10M_contiguous/future_adapt_overlap_items_kcore.csv \
  --output_dir results/backbone_contiguous --device cuda

# Split B
python backbone/train_backbone.py \
  --hist_data  data/processed/split_10M_tail/hist_overlap_items_kcore.csv \
  --val_data   data/processed/split_10M_tail/future_adapt_overlap_items_kcore.csv \
  --output_dir results/backbone_tail --device cuda
```

Each split gets its own backbone because k-core filtering produces a different item vocabulary per split.

### 4. Adapt and evaluate

Example using split A (contiguous). Replace `_contiguous` → `_tail` and `backbone_contiguous` → `backbone_tail` for split B.

```bash
# Last-block fine-tuning
python adaptation/last_block/train.py \
  --checkpoint results/backbone_contiguous/sasrec_backbone_best.pt \
  --adapt_data data/processed/split_10M_contiguous/future_adapt_high_drift_kcore.csv \
  --output_dir results/last_block_contiguous --device cuda

python adaptation/last_block/eval.py \
  --checkpoint    results/backbone_contiguous/sasrec_backbone_best.pt \
  --ft_checkpoint results/last_block_contiguous/last_block_best.pt \
  --test_data     data/processed/split_10M_contiguous/future_test_high_drift_kcore.csv \
  --outdir        results/last_block_contiguous/eval

# Context gate adapter
python adaptation/context_steering/train.py \
  --checkpoint results/backbone_contiguous/sasrec_backbone_best.pt \
  --adapt_data data/processed/split_10M_contiguous/future_adapt_high_drift_kcore.csv \
  --output_dir results/context_gate_contiguous --device cuda

python adaptation/context_steering/eval.py \
  --checkpoint       results/backbone_contiguous/sasrec_backbone_best.pt \
  --adapt_checkpoint results/context_gate_contiguous/context_gate_best.pt \
  --test_data        data/processed/split_10M_contiguous/future_test_high_drift_kcore.csv \
  --outdir           results/context_gate_contiguous/eval

# Prototype steering
python adaptation/prototype_steering/train.py \
  --checkpoint results/backbone_contiguous/sasrec_backbone_best.pt \
  --adapt_data data/processed/split_10M_contiguous/future_adapt_high_drift_kcore.csv \
  --output_dir results/prototype_steering_contiguous --device cuda

python adaptation/prototype_steering/eval.py \
  --checkpoint       results/backbone_contiguous/sasrec_backbone_best.pt \
  --adapt_checkpoint results/prototype_steering_contiguous/prototype_steering_best.pt \
  --test_data        data/processed/split_10M_contiguous/future_test_high_drift_kcore.csv \
  --outdir           results/prototype_steering_contiguous/eval
```

### 5. Hyperparameter sweeps

Each adaptation mode has a `sweep.py` that runs a full grid search and writes results to `sweep_results.csv`:

```bash
python adaptation/last_block/sweep.py \
  --checkpoint  results/backbone_contiguous/sasrec_backbone_best.pt \
  --adapt_data  data/processed/split_10M_contiguous/future_adapt_high_drift_kcore.csv \
  --test_data   data/processed/split_10M_contiguous/future_test_high_drift_kcore.csv \
  --base_outdir results/sweep_last_block_contiguous --device cuda
```

## Evaluation Protocol

All evaluations use leave-one-out on the `future_test` split:
- Context: all items except the last in each user's future sequence
- Target: last item
- Candidates: target + 100 randomly sampled negatives (fixed seed)
- Metrics: NDCG@10, HR@10, NDCG@20, HR@20, MRR@10

Baseline and adapted model score **the same candidate set per user** to ensure comparability.

## Design Principles

- **Strict separation**: data logic in `data/`, model architecture in `backbone/`, adaptation in `adaptation/`, shared utilities in `common/`
- **No shared state between adaptation modes**: each mode's train/eval/sweep scripts are self-contained
- **Config-driven**: all hyperparameters accessible via CLI; default configs in `adaptation/*/config.yaml`
- **Reproducibility**: explicit `--seed` argument in all scripts; fixed candidate sets per user
