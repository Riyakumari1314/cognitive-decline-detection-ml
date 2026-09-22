"""
Evaluation metrics module.

This module will calculate the performance metrics
used for cognitive-decline classification.
"""

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)


def calculate_metrics(y_true, y_pred, y_probability=None):
    """
    Calculate classification performance metrics.

    Class labels:
    0 = Healthy Control
    1 = Cognitive Decline
    """

    results = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1_score": f1_score(y_true, y_pred, zero_division=0)
    }

    if y_probability is not None:
        results["roc_auc"] = roc_auc_score(
            y_true,
            y_probability
        )

    return results
