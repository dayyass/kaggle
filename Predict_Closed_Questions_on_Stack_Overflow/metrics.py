from typing import Dict

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> Dict[str, float]:
    """
    Compute multiclass classification metrics.

    Args:
        y_true (np.ndarray): targets.
        y_pred (np.ndarray): predictions.

    Returns:
        Dict[str, float]: metrics.
    """

    metrics_kwargs = {
        'y_true': y_true,
        'y_pred': y_pred,
    }

    accuracy = accuracy_score(**metrics_kwargs)

    precision_micro = precision_score(
        **metrics_kwargs,
        average='micro',
        zero_division=0,
    )
    precision_macro = precision_score(
        **metrics_kwargs,
        average='macro',
        zero_division=0,
    )
    precision_weighted = precision_score(
        **metrics_kwargs,
        average='weighted',
        zero_division=0,
    )

    recall_micro = recall_score(
        **metrics_kwargs,
        average='micro',
        zero_division=0,
    )
    recall_macro = recall_score(
        **metrics_kwargs,
        average='macro',
        zero_division=0,
    )
    recall_weighted = recall_score(
        **metrics_kwargs,
        average='weighted',
        zero_division=0,
    )

    f1_micro = f1_score(
        **metrics_kwargs,
        average='micro',
        zero_division=0,
    )
    f1_macro = f1_score(
        **metrics_kwargs,
        average='macro',
        zero_division=0,
    )
    f1_weighted = f1_score(
        **metrics_kwargs,
        average='weighted',
        zero_division=0,
    )

    metrics = {
        "accuracy": accuracy,

        "precision_micro": precision_micro,
        "precision_macro": precision_macro,
        "precision_weighted": precision_weighted,

        "recall_micro": recall_micro,
        "recall_macro": recall_macro,
        "recall_weighted": recall_weighted,

        "f1_micro": f1_micro,
        "f1_macro": f1_macro,
        "f1_weighted": f1_weighted,
    }

    return metrics
