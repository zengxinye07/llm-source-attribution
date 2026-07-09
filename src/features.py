"""Feature extraction (proposal 3.2) and assembly for the ablation (3.3.3).

Golden rule: every transformer is fit on TRAIN ONLY, then applied to all rows.
TF-IDF vocabulary, StandardScaler stats, everything -- train only. Fitting on
the full set leaks and invalidates the whole leakage-control argument (3.3.1).

Cost order (write them in this order): stylometric < sbert < biber.
Cache SBERT embeddings and spaCy POS tags -- they are reused downstream.
"""
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import scipy.sparse as sp

from . import config
from .utils import cache_or_compute

Matrix = np.ndarray  # dense; TF-IDF returns scipy sparse instead


# --------------------------------------------------------------------------- #
# 1. TF-IDF lexical baseline (cheap, but sparse)
# --------------------------------------------------------------------------- #
def build_tfidf(texts_train: pd.Series, texts_all: pd.Series) -> Tuple[sp.csr_matrix, object]:
    """Fit TfidfVectorizer on train, transform all. Returns (sparse_matrix_all, vectorizer).

    Fit on TRAIN ONLY (vocabulary + idf) to avoid leakage, then transform the
    full corpus so callers can slice the result by split index. Return the fitted
    vectorizer too, for vocabulary/coefficient inspection later.
    """
    from sklearn.feature_extraction.text import TfidfVectorizer

    vec = TfidfVectorizer(**config.TFIDF_PARAMS)
    vec.fit(texts_train)
    X_all = vec.transform(texts_all)
    return X_all, vec


# --------------------------------------------------------------------------- #
# 2. Stylometric features (dense) -- write this one FIRST, it's pure stats
# --------------------------------------------------------------------------- #
def build_stylometric(texts: pd.Series) -> pd.DataFrame:
    """Per-text stylometric indicators (proposal 3.2).

    TODO: compute at least
      - avg sentence length, avg word length
      - lexical diversity (type-token ratio; consider MTLD for length robustness)
      - punctuation-based ratios (commas, periods, question/exclamation, etc.)
    Return a DataFrame indexed like `texts`. Scale later (fit on train only).
    """
    raise NotImplementedError


# --------------------------------------------------------------------------- #
# 3. Biber-inspired linguistic features (dense, SLOW -- spaCy POS)
# --------------------------------------------------------------------------- #
def _pos_tag_all(texts: pd.Series) -> list:
    """Run spaCy once over everything and cache the tag output.

    TODO: nlp = spacy.load(config.SPACY_MODEL_NAME, disable=["ner","lemmatizer"]).
    Use nlp.pipe(texts, batch_size=...) for speed. Return a lightweight structure
    (e.g. list of POS-tag lists + the dependency info you need), NOT Doc objects.
    """
    raise NotImplementedError


def build_biber(texts: pd.Series) -> pd.DataFrame:
    """Register-based linguistic features (Biber 1988; proposal 3.2).

    Uses cached POS tags. Compute POS distributions + grammatical markers:
      pronoun use, modal verbs, passive constructions, nominalisation,
      discourse markers.
    """
    pos = cache_or_compute("biber_pos_tags", lambda: _pos_tag_all(texts))
    # TODO: turn `pos` into feature columns. Return DataFrame indexed like texts.
    raise NotImplementedError


# --------------------------------------------------------------------------- #
# 4. SBERT semantic embeddings (dense) -- cache; reused by analysis (3.3.6)
# --------------------------------------------------------------------------- #
def build_sbert(texts: pd.Series) -> np.ndarray:
    """Encode every text once with SBERT and cache the matrix.

    The SAME embeddings feed classification (Exp 3/5/6) and the centroid /
    similarity analysis (proposal 3.3.6). Compute once.

    TODO: SentenceTransformer(config.SBERT_MODEL_NAME).encode(
              texts.tolist(), batch_size=..., show_progress_bar=True,
              convert_to_numpy=True, normalize_embeddings=True)
    """
    return cache_or_compute("sbert_embeddings", lambda: _encode(texts))


def _encode(texts: pd.Series) -> np.ndarray:
    raise NotImplementedError


# --------------------------------------------------------------------------- #
# Scaling helper -- fit on train rows only
# --------------------------------------------------------------------------- #
def scale_dense(X: np.ndarray, train_idx: np.ndarray) -> Tuple[np.ndarray, object]:
    """StandardScaler fit on X[train_idx], applied to all of X. Returns (X_scaled, scaler).

    NEVER pass the TF-IDF block here -- only dense feature blocks (stylometric,
    biber, sbert, length). Fitting on train rows only keeps the split honest.
    """
    from sklearn.preprocessing import StandardScaler

    X = np.asarray(X, dtype=float)
    scaler = StandardScaler()
    scaler.fit(X[train_idx])
    return scaler.transform(X), scaler


# --------------------------------------------------------------------------- #
# Assembler -- build the six ablation conditions (proposal 3.3.3)
# --------------------------------------------------------------------------- #
# Note (3.2): stylometric + Biber are treated as one combined "style" block.
EXPERIMENTS = {
    "exp1_tfidf":            ["tfidf"],
    "exp2_style":            ["stylometric", "biber"],
    "exp3_sbert":            ["sbert"],
    "exp4_tfidf_style":      ["tfidf", "stylometric", "biber"],
    "exp5_style_sbert":      ["stylometric", "biber", "sbert"],
    "exp6_all":              ["tfidf", "stylometric", "biber", "sbert"],
}


def assemble(blocks: Dict[str, object], which: list, add_length: bool = True):
    """Horizontally stack the requested feature blocks into one design matrix.

    KEY GOTCHA: TF-IDF is sparse, the rest are dense. If TF-IDF is included the
    result must be sparse -- use scipy.sparse.hstack and keep the whole matrix
    sparse. Only the dense blocks get scaled (done upstream in scale_dense).

    Parameters
    ----------
    blocks : dict
        Precomputed matrices keyed by name: "tfidf" (sparse), "stylometric",
        "biber", "sbert" (dense), plus "length" (dense, single column).
    which : list
        Subset of block names for this experiment (see EXPERIMENTS).
    add_length : bool
        Append the log_token_count covariate (proposal 3.1 says all conditions).
    """
    parts = [blocks[name] for name in which]
    if add_length:
        parts.append(blocks["length"])

    has_sparse = any(sp.issparse(p) for p in parts)
    if has_sparse:
        parts = [p if sp.issparse(p) else sp.csr_matrix(np.asarray(p)) for p in parts]
        return sp.hstack(parts, format="csr")
    return np.hstack([np.asarray(p) for p in parts])
