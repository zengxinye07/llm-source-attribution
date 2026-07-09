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
def pairwise_confusion_scores(confusion: np.ndarray, labels: list) -> pd.DataFrame:
    """Symmetric confusion score for every model pair.

    For each (i, j): average the two directional misclassification rates,
    i.e. mean(P(pred=j | true=i), P(pred=i | true=j)). Return a tidy DataFrame
    (model_a, model_b, score, same_family) sorted descending.

    TODO: row-normalise the confusion matrix first (divide by row sums) so each
    entry is a rate, then average the (i,j) and (j,i) rates. Tag same_family
    using config.MODEL_FAMILIES for the within- vs cross-family comparison.
    """
    raise NotImplementedError


def cluster_confusion_rows(confusion: np.ndarray, labels: list):
    """Hierarchical clustering on confusion-matrix rows (each model's misclass
    distribution as a feature vector). Returns a linkage matrix for a dendrogram.

    TODO: row-normalise -> scipy.cluster.hierarchy.linkage(..., method='ward').
    Compare resulting groupings against config.MODEL_FAMILIES.
    """
    raise NotImplementedError


def within_vs_cross_family(pairwise_df: pd.DataFrame) -> Dict[str, float]:
    """Summarise mean confusion for within-family vs cross-family pairs (RQ3)."""
    raise NotImplementedError


# --------------------------------------------------------------------------- #
# Embedding similarity (proposal 3.3.6, RQ2 / RQ3)
# --------------------------------------------------------------------------- #
def class_centroids(embeddings: np.ndarray, y: np.ndarray, labels: list = None) -> Dict[str, np.ndarray]:
    """Mean SBERT embedding per source class. labels default = config.CLASSES."""
    labels = labels or list(config.CLASSES)
    # TODO: for each label, embeddings[y == label].mean(axis=0).
    raise NotImplementedError


def centroid_cosine_matrix(centroids: Dict[str, np.ndarray]) -> pd.DataFrame:
    """Pairwise cosine similarity between class centroids (proposal 3.3.6, RQ3).

    TODO: sklearn.metrics.pairwise.cosine_similarity on stacked centroids.
    Return a labelled square DataFrame.
    """
    raise NotImplementedError


def ward_dendrogram_linkage(centroids: Dict[str, np.ndarray]):
    """Ward linkage over centroid distances -> dendrogram input (RQ3).

    TODO: convert cosine sim to distance (1 - sim), condense, linkage(method='ward').
    """
    raise NotImplementedError


def domain_human_distances(
    embeddings: np.ndarray, df: pd.DataFrame, labels: list = None
) -> pd.DataFrame:
    """Per-domain distance from each LLM centroid to the human centroid (RQ2).

    For each domain compute human centroid within that domain, then each LLM's
    distance to it. Report mean and std of these domain-wise distances per LLM
    -> which LLMs are closest to human writing, and how stable that is.

    TODO: loop config.DOMAINS; within-domain centroids; distance (cosine or L2);
    aggregate to (llm, mean_distance, std_distance).
    """
    raise NotImplementedError


# --------------------------------------------------------------------------- #
# Domain-stratified analysis (proposal 3.3.7, RQ4)
# --------------------------------------------------------------------------- #
def run_per_domain(df, splits, feature_blocks, clf_name: str = "logreg") -> pd.DataFrame:
    """Re-run core classification separately for each of the 8 domains (RQ4).

    For each domain: restrict train/eval to that domain's rows, call
    modeling.train_and_evaluate, collect per-class F1 / confusion / feature
    importance. Return a tidy DataFrame keyed by (domain, class).

    TODO: import modeling.train_and_evaluate; slice by df[DOMAIN_COL]==domain
    intersected with each split index; aggregate results.
    """
    raise NotImplementedError
