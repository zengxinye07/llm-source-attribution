"""Classification framework (proposal 3.2) and evaluation (3.3.2).

train_and_evaluate is the single most-reused function in the project -- every
ablation condition, the detection baseline, and every domain slice call it.
Get this abstraction right and everything downstream is one-liners.
"""
from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np

from . import config


# --------------------------------------------------------------------------- #
# Classifier factory (proposal 3.2)
# --------------------------------------------------------------------------- #
def get_classifier(name: str):
    """Return an unfitted estimator.

      "logreg" -> multinomial LogisticRegression (PRIMARY classifier)
      "svm"    -> LinearSVC (all feature conditions)
      "nb"     -> MultinomialNB (TF-IDF conditions ONLY -- needs non-negative
                  features; do NOT call it on scaled/dense blocks)

    Note: modern sklearn LogisticRegression handles multinomial natively with
    'lbfgs'/'saga', so no multi_class arg is needed. For large sparse TF-IDF,
    switch solver to 'saga' if lbfgs is slow to converge.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import LinearSVC
    from sklearn.naive_bayes import MultinomialNB

    if name == "logreg":
        return LogisticRegression(
            solver="lbfgs", C=1.0, max_iter=1000, random_state=config.RANDOM_STATE
        )
    if name == "svm":
        return LinearSVC(C=1.0, random_state=config.RANDOM_STATE)
    if name == "nb":
        return MultinomialNB()
    raise ValueError(f"unknown classifier: {name!r}")


@dataclass
class EvalResult:
    """Everything a notebook needs to report + drive downstream analysis."""
    macro_f1: float
    weighted_f1: float
    accuracy: float
    per_class: Dict[str, Dict[str, float]]  # class -> {precision, recall, f1}
    confusion: np.ndarray                    # rows = true, cols = pred (label order = config.CLASSES)
    labels: list = field(default_factory=lambda: list(config.CLASSES))
    model: Optional[object] = None
    y_pred: Optional[np.ndarray] = None


# --------------------------------------------------------------------------- #
# The core reusable routine (proposal 3.3.2)
# --------------------------------------------------------------------------- #
def train_and_evaluate(
    clf_name: str,
    X_train, y_train,
    X_eval, y_eval,
    labels: Optional[list] = None,
) -> EvalResult:
    """Fit `clf_name` on train, evaluate on X_eval/y_eval, return full metrics.

    Use X_eval = validation during tuning; X_eval = test only for final numbers.

    Metrics (proposal 3.3.2): Macro-F1 (primary), Accuracy, Weighted-F1,
    per-class Precision/Recall/F1, and the confusion matrix.

    `labels` fixes the class ordering for the confusion matrix / per-class table.
    Defaults to the sorted union of labels seen in train+eval, so this also works
    for the binary detection baseline (nb04), not just the 12-way task.
    """
    from sklearn.metrics import (
        accuracy_score,
        classification_report,
        confusion_matrix,
        f1_score,
    )

    y_train = np.asarray(y_train)
    y_eval = np.asarray(y_eval)
    if labels is None:
        labels = sorted(np.unique(np.concatenate([y_train, y_eval])).tolist())

    clf = get_classifier(clf_name)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_eval)

    macro_f1 = f1_score(y_eval, y_pred, average="macro", labels=labels, zero_division=0)
    weighted_f1 = f1_score(y_eval, y_pred, average="weighted", labels=labels, zero_division=0)
    accuracy = accuracy_score(y_eval, y_pred)

    report = classification_report(
        y_eval, y_pred, labels=labels, output_dict=True, zero_division=0
    )
    per_class = {
        c: {
            "precision": report[c]["precision"],
            "recall": report[c]["recall"],
            "f1": report[c]["f1-score"],
        }
        for c in labels
    }
    conf = confusion_matrix(y_eval, y_pred, labels=labels)

    return EvalResult(
        macro_f1=macro_f1,
        weighted_f1=weighted_f1,
        accuracy=accuracy,
        per_class=per_class,
        confusion=conf,
        labels=list(labels),
        model=clf,
        y_pred=y_pred,
    )


def tune_hyperparams(clf_name: str, X_train, y_train, X_val, y_val, grid: dict):
    """Small val-based sweep (proposal 3.3.1: val is for tuning/model selection).

    TODO: loop over `grid`, pick the config with best val Macro-F1, return it.
    Keep it lightweight -- LR/SVM don't need a huge search here.
    """
    raise NotImplementedError
