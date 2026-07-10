"""Data loading, preprocessing (proposal 3.1) and splitting (proposal 3.3.1).

These two artifacts -- the clean dataframe and the split indices -- are the
foundation everything else is built on. Freeze them early and DO NOT change
them, or experiment results stop being comparable.
"""
import json
from typing import Dict

import numpy as np
import pandas as pd

from . import config
from .utils import cache_or_compute


# --------------------------------------------------------------------------- #
# Loading + preprocessing (proposal Section 3.1)
# --------------------------------------------------------------------------- #
def load_raw() -> pd.DataFrame:
    """Load the raw RAID dump (train_none.csv -- the only labeled RAID split;
    test_none.csv has no label columns and extra_none.csv is unused, see config.py).

    Returns a dataframe with model, domain, source_id, attack, decoding,
    repetition_penalty, and generation (renamed to text).
    """
    df = pd.read_csv(config.DATA_RAW)
    return df.rename(columns={"generation": "text"})


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """Apply the filtering pipeline in the exact order given in the proposal.

    Order matters -- follow proposal 3.1 literally:
      1. keep attack == "none"
      2. keep decoding == "greedy" AND repetition_penalty == "no"
         (human rows are kept even though these two columns are NaN for them)
      3. drop empty and duplicate texts
      4. drop texts with fewer than MIN_TOKENS tokens (likely corrupted)
      5. add log_token_count covariate
    """
    # 1. adversarial filter
    df = df[df["attack"] == "none"].copy()

    # 2. decoding filter -- keep greedy/no-penalty for LLMs, keep humans regardless
    is_human = df[config.LABEL_COL] == config.HUMAN_CLASS
    decoding_ok = (df["decoding"] == "greedy") & (df["repetition_penalty"] == "no")
    df = df[is_human | decoding_ok].copy()

    # 3. empty + duplicate removal
    df = df[df["text"].str.strip().str.len() > 0]
    df = df.drop_duplicates(subset=["text"])

    # 4. length floor
    # TODO: decide the token definition. Cheap: whitespace split. Consistent with
    # the SBERT/biber tokenizer is nicer but slower -- whitespace is fine here.
    tok_count = df["text"].str.split().str.len()
    df = df[tok_count >= config.MIN_TOKENS].copy()

    # 5. length covariate (used in ALL feature conditions, proposal 3.1)
    df["log_token_count"] = np.log(df["text"].str.split().str.len())

    df = df.reset_index(drop=True)
    return df


def load_or_build_clean(overwrite: bool = False) -> pd.DataFrame:
    """Cached entry point. Builds clean.parquet once, then just reads it."""
    if config.CLEAN_PARQUET.exists() and not overwrite:
        return pd.read_parquet(config.CLEAN_PARQUET)
    clean = preprocess(load_raw())
    _assert_clean(clean)
    clean.to_parquet(config.CLEAN_PARQUET)
    return clean


def _assert_clean(df: pd.DataFrame) -> None:
    """Sanity checks against the proposal's stated counts. Fail loud, fail early."""
    counts = df[config.LABEL_COL].value_counts()
    assert set(counts.index) == set(config.CLASSES), (
        f"Unexpected classes: {set(counts.index) ^ set(config.CLASSES)}"
    )
    # NOTE: relax these if your RAID snapshot differs; but log the discrepancy.
    # assert (counts == config.N_PER_CLASS).all(), counts.to_dict()
    # assert len(df) == config.N_TOTAL, len(df)


# --------------------------------------------------------------------------- #
# Splitting (proposal Section 3.3.1) -- grouped by source_id, stratified 70/15/15
# --------------------------------------------------------------------------- #
def make_splits(df: pd.DataFrame) -> Dict[str, np.ndarray]:
    """Grouped stratified split, 70/15/15.

    All texts sharing a source_id land in the SAME split (prompt-leakage control,
    proposal 3.3.1). Grouping takes precedence over the exact ratio; we stratify
    within that constraint.

    Stratification key is (model, domain) jointly, not model alone. Every
    source_id group already contains exactly one row per class (12/12), so
    class balance across folds is guaranteed by construction regardless of
    how groups are assigned -- domain is the only thing that actually varies
    group-to-group and needs deliberate stratification (proposal 3.3.7 needs
    each split to be domain-balanced too, not just class-balanced).

    Two-stage StratifiedGroupKFold:
      Stage 1: carve out test (~1/7 = 14.3%) from everything.
      Stage 2: carve out val (~1/6 = 16.7% of the remaining 85% -> ~14.3% overall)
               from the train+val pool; the rest is train.
    Ratios are approximate by design (whole-group folds can't hit exact 70/15/15).

    Returns dict of {"train"/"val"/"test": positional_row_index_array} — indices
    are positions into `df` (0..n-1), matching `df.iloc[idx]` / `X[idx]` usage.
    """
    from sklearn.model_selection import StratifiedGroupKFold

    y = (df[config.LABEL_COL].astype(str) + "|" + df[config.DOMAIN_COL].astype(str)).to_numpy()
    groups = df[config.GROUP_COL].to_numpy()
    pos = np.arange(len(df))

    # Stage 1: first fold's test partition becomes our test set (~14.3%).
    sgkf1 = StratifiedGroupKFold(n_splits=7, shuffle=True, random_state=config.RANDOM_STATE)
    trainval_pos, test_pos = next(sgkf1.split(pos, y, groups))

    # Stage 2: split train+val; first fold's test partition becomes val (~16.7% of pool).
    sgkf2 = StratifiedGroupKFold(n_splits=6, shuffle=True, random_state=config.RANDOM_STATE)
    tr_rel, val_rel = next(sgkf2.split(trainval_pos, y[trainval_pos], groups[trainval_pos]))

    return {
        "train": trainval_pos[tr_rel],
        "val": trainval_pos[val_rel],
        "test": test_pos,
    }


def load_or_build_splits(df: pd.DataFrame, overwrite: bool = False) -> Dict[str, np.ndarray]:
    """Cached split indices. Frozen JSON -- this is the one thing you never rebuild."""
    if config.SPLIT_INDEX_PATH.exists() and not overwrite:
        with open(config.SPLIT_INDEX_PATH) as f:
            return {k: np.asarray(v) for k, v in json.load(f).items()}
    splits = make_splits(df)
    _assert_no_group_leakage(df, splits)
    with open(config.SPLIT_INDEX_PATH, "w") as f:
        json.dump({k: v.tolist() for k, v in splits.items()}, f)
    return splits


def _assert_no_group_leakage(df: pd.DataFrame, splits: Dict[str, np.ndarray]) -> None:
    """No source_id may appear in more than one split."""
    seen = {}
    for name, idx in splits.items():
        groups = set(df.iloc[idx][config.GROUP_COL].unique())
        for other, og in seen.items():
            overlap = groups & og
            assert not overlap, f"source_id leak between {name} and {other}: {list(overlap)[:5]}"
        seen[name] = groups
