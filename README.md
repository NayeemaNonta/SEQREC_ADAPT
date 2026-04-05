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

## Quickstart

### 1. Build dataset

```bash
python data/create_dataset/build_dataset.py \
  --raw_data data/raw/30music.tsv \
  --output_dir data/processed \
  --train_end 2014-01-01 --val_end 2014-04-01
```

### 2. Filter high-drift users

```bash
python data/preprocessing/run_pipeline.py \
  --hist_data data/processed/hist_kcore.csv \
  --future_data data/processed/future.csv \
  --backbone_ckpt results/backbone/sasrec_backbone_best.pt \
  --output_dir data/processed/high_drift
```

### 3. Train backbone (T1)

```bash
python backbone/train_backbone.py \
  --hist_data data/processed/hist_kcore.csv \
  --val_data  data/processed/future_val.csv \
  --output_dir results/backbone --device cuda
```

### 4. Adapt and evaluate

```bash
# Last-block fine-tuning
python adaptation/last_block/train.py \
  --checkpoint results/backbone/sasrec_backbone_best.pt \
  --adapt_data data/processed/high_drift/future_adapt.csv \
  --output_dir results/last_block --device cuda

python adaptation/last_block/eval.py \
  --checkpoint    results/backbone/sasrec_backbone_best.pt \
  --ft_checkpoint results/last_block/last_block_best.pt \
  --test_data     data/processed/high_drift/future_test.csv \
  --outdir        results/last_block/eval

# Context gate adapter
python adaptation/context_steering/train.py ...
python adaptation/context_steering/eval.py  ...

# Prototype steering
python adaptation/prototype_steering/train.py ...
python adaptation/prototype_steering/eval.py  ...
```

### 5. Hyperparameter sweeps

Each adaptation mode has a `sweep.py` that runs a full grid search and writes results to `sweep_results.csv`:

```bash
python adaptation/last_block/sweep.py \
  --checkpoint  results/backbone/sasrec_backbone_best.pt \
  --adapt_data  data/processed/high_drift/future_adapt.csv \
  --test_data   data/processed/high_drift/future_test.csv \
  --base_outdir results/sweep_last_block --device cuda
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
