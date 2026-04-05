"""
common/evaluation/evaluator.py

Shared evaluation infrastructure used by all adaptation modes.
Each adaptation mode implements a Scorer (callable: input_ids → hidden state),
then calls evaluate_shared() to get consistent apples-to-apples metrics.
"""

from __future__ import annotations

import random
from typing import Callable

import numpy as np
import pandas as pd
import torch

from common.data_utils import build_input_tensor, sample_negatives
from common.metrics.ranking import metrics_from_rank, summarize


# ---------------------------------------------------------------------------
# Example building — one shared candidate set per user
# ---------------------------------------------------------------------------

def build_eval_examples(
    context_map: dict,
    target_map:  dict,
    itemnum:     int,
    n_neg:       int = 100,
    cluster_map: dict | None = None,
) -> list[dict]:
    """
    Build one evaluation example per user. Candidate set = [target] + n_neg negatives.
    Both baseline and adapted models score the SAME candidate set per user.

    Args:
        cluster_map: optional {user_idx: cluster_id} — attached to each example
                     if provided (used by prototype-steering eval).
    """
    common = sorted(set(context_map) & set(target_map))
    if cluster_map is not None:
        common = [u for u in common if u in cluster_map]

    examples = []
    for user in common:
        context = context_map[user]
        target  = target_map[user]
        if len(context) == 0:
            continue
        seen = set(context) | {0}
        negatives = sample_negatives(itemnum, seen, target, n_neg=n_neg)
        ex = {
            "user_idx":   int(user),
            "context":    context,
            "target":     int(target),
            "candidates": [target] + negatives,
        }
        if cluster_map is not None:
            ex["cluster_id"] = int(cluster_map[user])
        examples.append(ex)
    return examples


# ---------------------------------------------------------------------------
# Scoring protocol
# ---------------------------------------------------------------------------

@torch.no_grad()
def score_examples(
    scorer:   Callable[[torch.Tensor], torch.Tensor],
    backbone, # SASRec — used for score_from_hidden
    examples: list[dict],
    maxlen:   int,
    device:   str,
) -> pd.DataFrame:
    """
    Score all examples with a single scorer.

    Args:
        scorer: function(input_ids: Tensor[1,T]) → hidden: Tensor[1,d]
                Pass backbone.get_last_hidden for baseline, or the adapted version.
        backbone: SASRec instance — provides score_from_hidden.

    Returns:
        DataFrame with one row per user and all metric columns.
    """
    rows = []
    for ex in examples:
        input_ids     = build_input_tensor(ex["context"], maxlen, device)
        candidate_ids = torch.tensor([ex["candidates"]], dtype=torch.long, device=device)

        h      = scorer(input_ids)
        logits = backbone.score_from_hidden(h, candidate_ids)[0]
        rank   = int((logits > logits[0]).sum().item())

        row = {"user_idx": ex["user_idx"], "rank": rank}
        if "cluster_id" in ex:
            row["cluster_id"] = ex["cluster_id"]
        row.update(metrics_from_rank(rank))
        rows.append(row)

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Paired evaluation (baseline vs adapted)
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_paired(
    baseline_scorer: Callable,
    adapted_scorer:  Callable,
    backbone,
    examples:        list[dict],
    maxlen:          int,
    device:          str,
) -> tuple[pd.DataFrame, dict, pd.DataFrame, dict]:
    """
    Evaluate baseline and adapted model on the SAME candidate set.

    Returns:
        base_df, base_metrics, adapted_df, adapted_metrics
    """
    base_df    = score_examples(baseline_scorer, backbone, examples, maxlen, device)
    adapted_df = score_examples(adapted_scorer,  backbone, examples, maxlen, device)

    return base_df, summarize(base_df), adapted_df, summarize(adapted_df)
