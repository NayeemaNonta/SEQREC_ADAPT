# Last-Block Adaptation

## Intuition

SASRec's transformer blocks learn hierarchical temporal patterns. The final block aggregates all prior context into the last hidden state that is scored against item embeddings. Under temporal drift — where item co-occurrence patterns shift between train and deployment — the final block is most likely to be misaligned: it was trained on historical co-occurrences that no longer hold.

Fine-tuning only the last block lets the model update these final-stage aggregation patterns while leaving all earlier representation-building layers intact. This limits the risk of catastrophic forgetting while targeting the most drift-sensitive component.

## Adapted Parameters

| Module | Parameters | Notes |
|---|---|---|
| `attention_layernorms[-1]` | 2 × d | Layer norm before attention |
| `attention_layers[-1]` | 4 × d² + 4d | Q, K, V, output projections |
| `forward_layernorms[-1]` | 2 × d | Layer norm before FFN |
| `forward_layers[-1]` | 2 × d² + 2d | Two-layer point-wise FFN |
| `last_layernorm` (optional) | 2 × d | Final norm before scoring |

With d=64: ~25,216 trainable parameters out of ~500K total backbone parameters (~5%).

## Loss

Cross-entropy over the full item vocabulary — identical to the backbone pretraining objective:

```
L = -log p(target | context) = -log softmax(h̃ · E)[target]
```

This is stronger than BCE with sampled negatives because every item ranked above the target receives a gradient signal, not just a small fixed set.

## Tradeoffs

| Property | Value |
|---|---|
| Backbone frozen | Partially (all layers except last block) |
| Risk of catastrophic forgetting | Low-moderate (5% of params updated) |
| Training data required | future_adapt only (~30K examples) |
| Inference overhead vs baseline | Zero (same architecture, loaded weights differ) |
| Requires clustering | No |

## Usage

```bash
# Train
python adaptation/last_block/train.py \
  --checkpoint  results/backbone/sasrec_backbone_best.pt \
  --adapt_data  data/processed/future_adapt.csv \
  --output_dir  results/last_block --device cuda

# Evaluate
python adaptation/last_block/eval.py \
  --checkpoint    results/backbone/sasrec_backbone_best.pt \
  --ft_checkpoint results/last_block/last_block_best.pt \
  --test_data     data/processed/future_test.csv \
  --outdir        results/last_block/eval

# Hyperparameter sweep
python adaptation/last_block/sweep.py \
  --checkpoint   results/backbone/sasrec_backbone_best.pt \
  --adapt_data   data/processed/future_adapt.csv \
  --test_data    data/processed/future_test.csv \
  --base_outdir  results/sweep_last_block --device cuda
```
