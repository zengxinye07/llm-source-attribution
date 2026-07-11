"""Feature extraction (proposal 3.2) and assembly for the ablation (3.3.3).

Golden rule: every transformer is fit on TRAIN ONLY, then applied to all rows.
TF-IDF vocabulary, StandardScaler stats, everything -- train only. Fitting on
the full set leaks and invalidates the whole leakage-control argument (3.3.1).

Cost order (write them in this order): stylometric < sbert < biber.
Cache SBERT embeddings and spaCy POS tags -- they are reused downstream.
"""
import re
from collections import Counter
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

    # float32: sklearn defaults to float64, which doubles this matrix's memory
    # for no accuracy benefit, and (worse) upcasts every other block to
    # float64 too once assemble() hstacks them together -- see assemble().
    vec = TfidfVectorizer(**config.TFIDF_PARAMS, dtype=np.float32)
    vec.fit(texts_train)
    X_all = vec.transform(texts_all)
    return X_all, vec


# --------------------------------------------------------------------------- #
# 2. Stylometric features (dense) -- write this one FIRST, it's pure stats
# --------------------------------------------------------------------------- #
_SENT_SPLIT_RE = re.compile(r"[.!?]+(?:\s+|$)")
_WORD_RE = re.compile(r"[A-Za-z']+")


def _mtld(words: list, ttr_threshold: float = 0.72) -> float:
    """McCarthy & Jarvis (2010) MTLD -- length-robust lexical diversity.

    TTR degrades on long texts (running vocabulary saturates); MTLD instead
    counts how many word "factors" it takes for the running TTR to decay past
    `ttr_threshold`, then averages a forward and backward pass so both tails
    of the text count. Falls back to raw type count on very short texts,
    where MTLD's factor-based average is undefined/unstable.
    """
    def _one_pass(seq):
        factors, types, token_count = 0.0, set(), 0
        for w in seq:
            token_count += 1
            types.add(w)
            if len(types) / token_count <= ttr_threshold:
                factors += 1
                types, token_count = set(), 0
        if token_count > 0:
            partial_ttr = len(types) / token_count
            factors += (1 - partial_ttr) / (1 - ttr_threshold)
        return len(seq) / factors if factors > 0 else float(len(seq))

    if len(words) < 10:
        return float(len(set(words)))
    return (_one_pass(words) + _one_pass(list(reversed(words)))) / 2


def build_stylometric(texts: pd.Series) -> pd.DataFrame:
    """Per-text stylometric indicators (proposal 3.2).

    - avg_sentence_len, avg_word_len
    - ttr (type-token ratio) and mtld (length-robust lexical diversity)
    - punctuation ratios (per character, so they're comparable across text
      lengths): commas, periods, question marks, exclamation marks

    Returns a DataFrame indexed like `texts`. Scale later (fit on train only).
    """
    rows = []
    for text in texts:
        words = _WORD_RE.findall(text)
        n_words = len(words)
        n_sents = max(len([s for s in _SENT_SPLIT_RE.split(text) if s.strip()]), 1)
        n_chars = max(len(text), 1)
        lower_words = [w.lower() for w in words]

        rows.append({
            "avg_sentence_len": n_words / n_sents,
            "avg_word_len": np.mean([len(w) for w in words]) if words else 0.0,
            "ttr": len(set(lower_words)) / n_words if n_words else 0.0,
            "mtld": _mtld(lower_words),
            "comma_ratio": text.count(",") / n_chars,
            "period_ratio": text.count(".") / n_chars,
            "question_ratio": text.count("?") / n_chars,
            "exclam_ratio": text.count("!") / n_chars,
        })
    return pd.DataFrame(rows, index=texts.index)


# --------------------------------------------------------------------------- #
# 3. Biber-inspired linguistic features (dense, SLOW -- spaCy POS)
# --------------------------------------------------------------------------- #
def _pos_tag_all(texts: pd.Series) -> list:
    """Run spaCy once over everything and cache the tag output.

    Returns a list (one entry per text) of (token_text, pos_, tag_, dep_) tuples
    -- a lightweight structure, not Doc objects, so it pickles cheaply.
    """
    import spacy

    nlp = spacy.load(config.SPACY_MODEL_NAME, disable=["ner", "lemmatizer"])
    out = []
    for doc in nlp.pipe(texts.tolist(), batch_size=64):
        out.append([(tok.text, tok.pos_, tok.tag_, tok.dep_) for tok in doc])
    return out


_POS_TAGS = [
    "NOUN", "PROPN", "VERB", "AUX", "ADJ", "ADV", "PRON", "DET",
    "ADP", "CCONJ", "SCONJ", "PART", "INTJ", "NUM",
]
# Common nominal suffixes (nominalisation proxy -- verb/adj turned into a noun).
_NOMINAL_SUFFIXES = ("tion", "sion", "ment", "ness", "ity", "ance", "ence")
# spaCy's dependency label for the passive-voice subject/aux varies by model
# version (nsubjpass/auxpass vs. the newer nsubj:pass/aux:pass); check both.
_PASSIVE_DEPS = {"nsubjpass", "auxpass", "nsubj:pass", "aux:pass"}
_DISCOURSE_MARKERS = [
    "however", "therefore", "moreover", "furthermore", "thus", "meanwhile",
    "nonetheless", "consequently", "additionally", "nevertheless",
    "in fact", "for example", "in other words", "in addition",
    "on the other hand", "as a result", "in contrast", "overall", "in conclusion",
]


def build_biber(texts: pd.Series) -> pd.DataFrame:
    """Register-based linguistic features (Biber 1988; proposal 3.2).

    Uses cached POS tags to compute, per text: POS-tag distribution (share of
    tokens per tag, which covers pronoun use via pos_pron), modal-verb ratio
    (tag_ == "MD"), passive-construction ratio (dependency labels above),
    nominalisation ratio (NOUN tokens ending in a nominal suffix), and
    discourse-marker ratio (fixed marker list, counted per word).
    """
    pos = cache_or_compute("biber_pos_tags", lambda: _pos_tag_all(texts))

    rows = []
    for text, tagged in zip(texts, pos):
        n_tokens = max(len(tagged), 1)
        pos_counts = Counter(p for _, p, _, _ in tagged)

        row = {f"pos_{tag.lower()}": pos_counts.get(tag, 0) / n_tokens for tag in _POS_TAGS}
        row["modal_ratio"] = sum(1 for _, _, tag, _ in tagged if tag == "MD") / n_tokens
        row["passive_ratio"] = sum(1 for _, _, _, dep in tagged if dep in _PASSIVE_DEPS) / n_tokens
        row["nominalisation_ratio"] = sum(
            1 for tok, p, _, _ in tagged if p == "NOUN" and tok.lower().endswith(_NOMINAL_SUFFIXES)
        ) / n_tokens

        n_words = max(len(text.split()), 1)
        lower_text = text.lower()
        row["discourse_marker_ratio"] = sum(lower_text.count(m) for m in _DISCOURSE_MARKERS) / n_words

        rows.append(row)
    return pd.DataFrame(rows, index=texts.index)


# --------------------------------------------------------------------------- #
# 4. SBERT semantic embeddings (dense) -- cache; reused by analysis (3.3.6)
# --------------------------------------------------------------------------- #
def build_sbert(texts: pd.Series) -> np.ndarray:
    """Encode every text once with SBERT and cache the matrix.

    The SAME embeddings feed classification (Exp 3/5/6) and the centroid /
    similarity analysis (proposal 3.3.6). Compute once.
    """
    return cache_or_compute("sbert_embeddings", lambda: _encode(texts))


def _encode(texts: pd.Series) -> np.ndarray:
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(config.SBERT_MODEL_NAME)
    return model.encode(
        texts.tolist(),
        batch_size=64,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )


# --------------------------------------------------------------------------- #
# Scaling helper -- fit on train rows only
# --------------------------------------------------------------------------- #
def scale_dense(X: np.ndarray, train_idx: np.ndarray) -> Tuple[np.ndarray, object]:
    """StandardScaler fit on X[train_idx], applied to all of X. Returns (X_scaled, scaler).

    NEVER pass the TF-IDF block here -- only dense feature blocks (stylometric,
    biber, sbert, length). Fitting on train rows only keeps the split honest.
    """
    from sklearn.preprocessing import StandardScaler

    # float32, not float64 (np.asarray's default via `dtype=float`) -- keeps
    # this block from upcasting the whole assembled matrix in assemble().
    X = np.asarray(X, dtype=np.float32)
    scaler = StandardScaler()
    scaler.fit(X[train_idx])
    return scaler.transform(X).astype(np.float32), scaler


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
        # scipy.sparse.hstack upcasts EVERY part to the widest dtype among
        # them -- one stray float64 block (e.g. the "length" column, which
        # nobody thinks to cast) silently doubles the whole assembled matrix.
        # Force float32 everywhere so hstack has nothing to upcast to.
        parts = [
            p.astype(np.float32) if sp.issparse(p) else sp.csr_matrix(np.asarray(p, dtype=np.float32))
            for p in parts
        ]
        return sp.hstack(parts, format="csr")
    return np.hstack([np.asarray(p) for p in parts])
