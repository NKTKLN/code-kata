import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def evaluate_classification(
    y_pred: NDArray[np.float64], y_true: NDArray[np.float64]
) -> pd.DataFrame:
    """Evaluate multiple classification model predictions on the provided dataset.

    Args:
        y_pred (NDArray[np.float64]): Predicted probabilities or binary outputs.
        y_true (NDArray[np.float64]): True binary target values of shape (n_samples,).

    Returns:
        pd.DataFrame: DataFrame containing classification metrics — Accuracy, Precision,
            Recall, F1-score, and ROC AUC. Rows correspond to the metric names.
    """
    # If predictions are probabilities, binarize with threshold 0.5
    y_pred_bin = (y_pred > 0.5).astype(int)

    metrics = {
        "Accuracy": accuracy_score(y_true, y_pred_bin),
        "Precision": precision_score(y_true, y_pred_bin),
        "Recall": recall_score(y_true, y_pred_bin),
        "F1-score": f1_score(y_true, y_pred_bin),
        "ROC AUC": roc_auc_score(y_true, y_pred),
    }

    return pd.DataFrame.from_dict(metrics, orient="index", columns=["score"]).T
