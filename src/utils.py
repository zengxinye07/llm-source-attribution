"""Reproducibility + caching helpers.

The caching helper is the backbone of the "cache everything expensive" rule:
clean df, split indices, SBERT embeddings, spaCy POS tags all go through it so
re-running a notebook is cheap.
"""
import os
import pickle
import random
from pathlib import Path
from typing import Any, Callable

import numpy as np

from . import config


def set_seed(seed: int = config.RANDOM_STATE) -> None:
    """Fix all RNGs we touch. Call at the top of every notebook."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    # TODO: if you add torch (SBERT backend), also seed torch here.


def cache_or_compute(key: str, compute_fn: Callable[[], Any], *, overwrite: bool = False) -> Any:
    """Return artifacts/<key>.pkl if it exists, else compute, save, and return.

    Parameters
    ----------
    key : str
        Filename stem under artifacts/ (no extension). Keep it descriptive,
        e.g. "sbert_embeddings" or "biber_pos_tags".
    compute_fn : callable
        Zero-arg function that produces the object when the cache is cold.
    overwrite : bool
        Force recompute even if the cache exists.
    """
    path: Path = config.ARTIFACTS / f"{key}.pkl"
    if path.exists() and not overwrite:
        with open(path, "rb") as f:
            return pickle.load(f)
    obj = compute_fn()
    with open(path, "wb") as f:
        pickle.dump(obj, f)
    return obj


def save_fig(fig, name: str) -> Path:
    """Save a matplotlib figure to figures/ for the report. Returns the path."""
    path = config.FIGURES / f"{name}.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    return path
