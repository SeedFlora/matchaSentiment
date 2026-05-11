from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    precision_recall_fscore_support,
    roc_auc_score,
)

from .config import ID2LABEL


def binary_metrics(y_true, y_pred, y_score=None) -> dict:
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="binary",
        pos_label=1,
        zero_division=0,
    )
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }
    if y_score is not None and len(np.unique(y_true)) == 2:
        try:
            metrics["roc_auc"] = float(roc_auc_score(y_true, y_score))
        except ValueError:
            metrics["roc_auc"] = float("nan")
    return metrics


def report_dict(y_true, y_pred) -> dict:
    return classification_report(
        y_true,
        y_pred,
        labels=[0, 1],
        target_names=[ID2LABEL[0], ID2LABEL[1]],
        zero_division=0,
        output_dict=True,
    )
