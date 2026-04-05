# Context-Conditioned Scalar Gate (Context Steering)

## Intuition

Cluster-based adaptation applies the same correction to all users in a cluster, regardless of how their specific sequence looks. Two users assigned to "high-drift" may have very different sequence structures after filtering — one with minor gaps, one heavily fragmented — but receive identical weight updates.

A context-conditioned adapter resolves this by making the correction a function of the hidden state itself: the compressed representation of the actual sequence being processed. Users with intact sequences (strong co-occurrence signal in h) get a different correction than users with fragmented sequences (noisy or shifted h).

## Method

```
h̃ = h + α(h) · f_ϕ(h)

α(h) = σ(w_gate · h + b_gate)    ∈ (0, 1)   scalar gate
f_ϕ(h) = W₂ · GELU(W₁ · h)                  bottleneck MLP
```

The gate α is context-dependent: large when the backbone representation needs correction, small when it is already well-calibrated. Both α and f_ϕ are conditioned on h — no user ID, cluster ID, or external signal required.

## Adapted Parameters

| Component | Parameters (d=64, r=8) |
|---|---|
| Gate: w_gate, b_gate | 65 |
| Down: W₁, b₁ | 8 × 64 + 8 = 520 |
| Up: W₂, b₂ | 64 × 8 + 64 = 576 |
| **Total** | **1,161** |

Gate bias initialized to −2 → α ≈ 0.12 at init, ensuring small initial edits.

## Tradeoffs

| Property | Value |
|---|---|
| Backbone frozen | Fully |
| Risk of catastrophic forgetting | None |
| Trainable parameters | ~1,200 (shared across all users) |
| Requires clustering | No |
| Applies to unseen users | Yes (any input sequence) |
| Inference overhead | One linear + sigmoid + bottleneck MLP |

## Usage

```bash
python adaptation/context_steering/train.py \
  --checkpoint  results/backbone/sasrec_backbone_best.pt \
  --adapt_data  data/processed/future_adapt.csv \
  --output_dir  results/context_gate --device cuda

python adaptation/context_steering/eval.py \
  --checkpoint      results/backbone/sasrec_backbone_best.pt \
  --adapt_checkpoint results/context_gate/context_gate_best.pt \
  --test_data       data/processed/future_test.csv \
  --outdir          results/context_gate/eval

python adaptation/context_steering/sweep.py \
  --checkpoint   results/backbone/sasrec_backbone_best.pt \
  --adapt_data   data/processed/future_adapt.csv \
  --test_data    data/processed/future_test.csv \
  --base_outdir  results/sweep_context_gate --device cuda
```
