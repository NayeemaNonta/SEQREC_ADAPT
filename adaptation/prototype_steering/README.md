# Prototype-Based Steering

## Intuition

Users cluster into drift archetypes: some shift toward new item categories, others intensify existing patterns, others simply become inactive. A single global adapter cannot capture these structurally different correction directions.

Prototype steering assigns each user to a cluster (prototype) and learns an independent residual MLP per cluster. Each cluster's adapter is trained only on its own users' future interactions, so it specialises to that archetype's drift pattern.

## Method

```
h̃ = h + f_{ϕ_{z_u}}(h)    z_u ∈ {0, …, K-1}
```

The cluster assignment z_u is computed offline (e.g., k-means on user drift embeddings or interaction statistics) and stored in a CSV. At inference, the user's cluster determines which adapter applies.

## Adapted Parameters

K independent bottleneck MLPs:

| Component | Parameters per cluster (d=64, r=32) |
|---|---|
| Down: d → r | 64 × 32 + 32 = 2,080 |
| Up: r → d | 32 × 64 + 64 = 2,112 |
| **Per cluster** | **4,192** |
| **Total (K=5)** | **20,960** |

Output projection zero-initialized — residual starts at 0 for all clusters.

## Tradeoffs

| Property | Value |
|---|---|
| Backbone frozen | Fully |
| Risk of catastrophic forgetting | None |
| Trainable parameters | K × 4,192 (scales with clusters) |
| Requires clustering | Yes — offline, run once |
| Applies to unseen users | Only if cluster can be assigned at inference |
| Inference overhead | One cluster-indexed bottleneck MLP |

## Cluster Assignment

Provide a CSV with columns `[user_id, cluster_id]`. Cluster IDs must be integers in `[0, K-1]`. Any clustering strategy works (k-means on drift scores, spectral clustering, manual thresholds).

## Usage

```bash
python adaptation/prototype_steering/train.py \
  --checkpoint   results/backbone/sasrec_backbone_best.pt \
  --adapt_data   data/processed/future_adapt.csv \
  --cluster_csv  data/processed/user_clusters.csv \
  --output_dir   results/prototype \
  --num_clusters 5 --device cuda

python adaptation/prototype_steering/eval.py \
  --checkpoint       results/backbone/sasrec_backbone_best.pt \
  --adapt_checkpoint results/prototype/prototype_best.pt \
  --cluster_csv      data/processed/user_clusters.csv \
  --test_data        data/processed/future_test.csv \
  --outdir           results/prototype/eval

python adaptation/prototype_steering/sweep.py \
  --checkpoint   results/backbone/sasrec_backbone_best.pt \
  --adapt_data   data/processed/future_adapt.csv \
  --cluster_csv  data/processed/user_clusters.csv \
  --test_data    data/processed/future_test.csv \
  --base_outdir  results/sweep_prototype --device cuda
```
