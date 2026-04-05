"""
common/data_utils.py

Shared data loading, encoding, and sequence utilities.
All adaptation modes and the backbone trainer import from here.
"""

import random

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import LabelEncoder


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------

def read_interactions(path: str) -> pd.DataFrame:
    """Load a CSV of interactions, normalising column names to user_id/item_id/timestamp."""
    df = pd.read_csv(path)
    rename_map = {}
    if "user" in df.columns:
        rename_map["user"] = "user_id"
    if "item" in df.columns:
        rename_map["item"] = "item_id"
    df = df.rename(columns=rename_map)
    required = {"user_id", "item_id", "timestamp"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path} missing columns: {missing}")
    return df[list(required)].dropna().copy()


def load_checkpoint(path: str, device: str) -> dict:
    return torch.load(path, map_location=device, weights_only=False)


# ---------------------------------------------------------------------------
# Label encoders
# ---------------------------------------------------------------------------

def make_encoders_from_ckpt(ckpt: dict):
    """Reconstruct LabelEncoders from checkpoint arrays."""
    le_item = LabelEncoder()
    le_item.classes_ = np.array(ckpt["le_item_classes"], dtype=object)
    le_user = LabelEncoder()
    le_user.classes_ = np.array(ckpt["le_user_classes"], dtype=object)
    return le_user, le_item


# ---------------------------------------------------------------------------
# Encoding
# ---------------------------------------------------------------------------

def encode_df(df: pd.DataFrame, le_user: LabelEncoder, le_item: LabelEncoder) -> pd.DataFrame:
    """
    Filter to known users/items and add integer index columns.
    item_idx is 1-indexed (0 reserved for padding).
    """
    known_users = set(le_user.classes_.tolist())
    known_items = set(le_item.classes_.tolist())
    out = df.copy()
    out["user_id"] = out["user_id"].astype(str)
    out["item_id"] = out["item_id"].astype(str)
    out = out[out["user_id"].isin(known_users) & out["item_id"].isin(known_items)].copy()
    out["user_idx"] = le_user.transform(out["user_id"])
    out["item_idx"] = le_item.transform(out["item_id"]) + 1  # 0 = padding
    return out.sort_values(["user_idx", "timestamp"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Sequence construction
# ---------------------------------------------------------------------------

def build_sequences_by_user(df: pd.DataFrame) -> dict:
    """Returns {user_idx: [item_idx, ...]} sorted by timestamp."""
    return df.groupby("user_idx")["item_idx"].agg(list).to_dict()


def leave_one_out(user_seqs: dict) -> tuple[dict, dict, int]:
    """
    Split each user sequence into context (all but last) and target (last item).
    Returns (context_map, target_map, n_skipped_users_with_seq_lt_2).
    """
    context_map, target_map = {}, {}
    skipped = 0
    for user, seq in user_seqs.items():
        if len(seq) < 2:
            skipped += 1
            continue
        context_map[user] = seq[:-1]
        target_map[user] = seq[-1]
    return context_map, target_map, skipped


# ---------------------------------------------------------------------------
# Tensor helpers
# ---------------------------------------------------------------------------

def pad_sequence(seq: list, maxlen: int) -> list:
    seq = seq[-maxlen:]
    return [0] * (maxlen - len(seq)) + seq


def build_input_tensor(seq: list, maxlen: int, device: str) -> torch.Tensor:
    return torch.tensor([pad_sequence(seq, maxlen)], dtype=torch.long, device=device)


# ---------------------------------------------------------------------------
# Negative sampling
# ---------------------------------------------------------------------------

def sample_negatives(itemnum: int, seen: set, target: int, n_neg: int = 100) -> list:
    negatives = []
    while len(negatives) < n_neg:
        t = random.randint(1, itemnum)
        if t not in seen and t != target:
            negatives.append(t)
    return negatives
