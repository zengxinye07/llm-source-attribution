"""Post-classification analysis.

Covers: error analysis (proposal 3.3.5, RQ3), embedding similarity (3.3.6,
RQ2/RQ3), and domain-stratified helpers (3.3.7, RQ4). Everything here reads
cached artifacts (confusion matrices, SBERT embeddings) -- no re-training.
"""
from typing import Dict

import numpy as np
import pandas as pd

from . import config


# --------------------------------------------------------------------------- #
# Error analysis (proposal 3.3.5, RQ3)
# --------------------------------------------------------------------------- #
def _row_normalize(confusion: np.ndarray) -> np.ndarray:
    conf = np.asarray(confusion, dtype=float)
    row_sums = conf.sum(axis=1, keepdims=True)
    return np.divide(conf, row_sums, out=np.zeros_like(conf), where=row_sums != 0)


def pairwise_confusion_scores(confusion: np.ndarray, labels: list) -> pd.DataFrame:
    """Symmetric confusion score for every model pair.

    For each (i, j): average the two directional misclassification rates,
    i.e. mean(P(pred=j | true=i), P(pred=i | true=j)). Returns a tidy DataFrame
    (model_a, model_b, score, same_family), sorted descending by score.
    """
    rate = _row_normalize(confusion)
    rows = []
    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            rows.append({
                "model_a": labels[i],
                "model_b": labels[j],
                "score": (rate[i, j] + rate[j, i]) / 2,
                "same_family": config.MODEL_FAMILIES.get(labels[i]) == config.MODEL_FAMILIES.get(labels[j]),
            })
    return pd.DataFrame(rows).sort_values("score", ascending=False).reset_index(drop=True)


def cluster_confusion_rows(confusion: np.ndarray, labels: list):
    """Hierarchical clustering on confusion-matrix rows (each model's misclass
    distribution as a feature vector). Returns a linkage matrix for a dendrogram.

    Compare resulting groupings against config.MODEL_FAMILIES.
    """
    from scipy.cluster.hierarchy import linkage

    return linkage(_row_normalize(confusion), method="ward")


def within_vs_cross_family(pairwise_df: pd.DataFrame) -> Dict[str, float]:
    """Summarise mean confusion for within-family vs cross-family pairs (RQ3)."""
    within = pairwise_df.loc[pairwise_df["same_family"], "score"]
    cross = pairwise_df.loc[~pairwise_df["same_family"], "score"]
    return {
        "within_family_mean": float(within.mean()) if len(within) else float("nan"),
        "cross_family_mean": float(cross.mean()) if len(cross) else float("nan"),
        "within_family_n": int(len(within)),
        "cross_family_n": int(len(cross)),
    }


# --------------------------------------------------------------------------- #
# Embedding similarity (proposal 3.3.6, RQ2 / RQ3)
# --------------------------------------------------------------------------- #
def class_centroids(embeddings: np.ndarray, y: np.ndarray, labels: list = None) -> Dict[str, np.ndarray]:
    """Mean SBERT embedding per source class. labels default = config.CLASSES."""
    labels = labels or list(config.CLASSES)
    return {label: embeddings[y == label].mean(axis=0) for label in labels}


def centroid_cosine_matrix(centroids: Dict[str, np.ndarray]) -> pd.DataFrame:
    """Pairwise cosine similarity between class centroids (proposal 3.3.6, RQ3)."""
    from sklearn.metrics.pairwise import cosine_similarity

    labels = list(centroids.keys())
    mat = np.stack([centroids[l] for l in labels])
    return pd.DataFrame(cosine_similarity(mat), index=labels, columns=labels)


def ward_dendrogram_linkage(centroids: Dict[str, np.ndarray]):
    """Ward linkage over centroid cosine distances -> dendrogram input (RQ3).

    Converts similarity to distance (1 - sim), symmetrises away floating-point
    noise, condenses, and runs Ward linkage. Leaf order matches
    `list(centroids.keys())`.
    """
    from scipy.cluster.hierarchy import linkage
    from scipy.spatial.distance import squareform
    from sklearn.metrics.pairwise import cosine_similarity

    labels = list(centroids.keys())
    mat = np.stack([centroids[l] for l in labels])
    dist = 1 - cosine_similarity(mat)
    np.fill_diagonal(dist, 0)
    dist = (dist + dist.T) / 2
    return linkage(squareform(dist, checks=False), method="ward")


def domain_human_distances(
    embeddings: np.ndarray, df: pd.DataFrame, labels: list = None
) -> pd.DataFrame:
    """Per-domain distance from each LLM centroid to the human centroid (RQ2).

    For each domain, compute the human centroid within that domain, then each
    LLM's cosine distance to it. Report mean and std of these domain-wise
    distances per LLM -> which LLMs are closest to human writing, and how
    stable that is across domains. Sorted ascending (closest to human first).
    """
    from sklearn.metrics.pairwise import cosine_similarity

    labels = labels or [c for c in config.CLASSES if c != config.HUMAN_CLASS]
    domain_vals = df[config.DOMAIN_COL].to_numpy()
    label_vals = df[config.LABEL_COL].to_numpy()

    distances = {label: [] for label in labels}
    for dom in config.DOMAINS:
        dom_mask = domain_vals == dom
        human_mask = dom_mask & (label_vals == config.HUMAN_CLASS)
        if not human_mask.any():
            continue
        human_centroid = embeddings[human_mask].mean(axis=0, keepdims=True)
        for label in labels:
            llm_mask = dom_mask & (label_vals == label)
            if not llm_mask.any():
                continue
            llm_centroid = embeddings[llm_mask].mean(axis=0, keepdims=True)
            sim = cosine_similarity(llm_centroid, human_centroid)[0, 0]
            distances[label].append(1 - sim)

    out = pd.DataFrame({
        label: {
            "mean_distance": np.mean(d) if d else float("nan"),
            "std_distance": np.std(d) if d else float("nan"),
            "n_domains": len(d),
        }
        for label, d in distances.items()
    }).T
    out.index.name = "model"
    return out.sort_values("mean_distance")


# --------------------------------------------------------------------------- #
# Domain-stratified analysis (proposal 3.3.7, RQ4)
# --------------------------------------------------------------------------- #
def run_per_domain(df, splits, feature_blocks, clf_name: str = "logreg") -> pd.DataFrame:
    """Re-run core classification separately for each of the 8 domains (RQ4).

    Assembles all blocks in `feature_blocks` once (same dict shape as
    `features.EXPERIMENTS['exp6_all']`'s inputs), then for each domain
    restricts train/val to that domain's rows (intersected with the global
    split, so no leakage) and trains+evaluates from scratch. Returns a tidy
    DataFrame keyed by (domain, class) with per-class precision/recall/f1 plus
    that domain's overall macro_f1 for convenience.
    """
    from . import features as _features
    from . import modeling as _modeling

    y = df[config.LABEL_COL].to_numpy()
    domain_vals = df[config.DOMAIN_COL].to_numpy()
    which = [k for k in feature_blocks if k != "length"]
    X = _features.assemble(feature_blocks, which)

    train_pos = np.asarray(splits["train"])
    val_pos = np.asarray(splits["val"])

    rows = []
    for dom in config.DOMAINS:
        dom_mask = domain_vals == dom
        tr_idx = train_pos[dom_mask[train_pos]]
        val_idx = val_pos[dom_mask[val_pos]]
        if len(tr_idx) == 0 or len(val_idx) == 0:
            continue
        res = _modeling.train_and_evaluate(
            clf_name, X[tr_idx], y[tr_idx], X[val_idx], y[val_idx], labels=list(config.CLASSES)
        )
        for cls, metrics in res.per_class.items():
            rows.append({"domain": dom, "class": cls, "macro_f1_domain": res.macro_f1, **metrics})
    return pd.DataFrame(rows)
