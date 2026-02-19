import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from numpy.typing import NDArray
from sklearn.metrics import auc, average_precision_score, precision_recall_curve


def plot_precision_recall_curve(
    y_true: NDArray[np.int64], y_probs: NDArray[np.int64]
) -> None:
    """Plot the Precision-Recall curve and display the AP and PR-AUC score.

    Args:
        y_true (NDArray[np.int64]): True binary class labels.
        y_probs (NDArray[np.float64]): Predicted probabilities for the positive class.
    """
    precision, recall, _ = precision_recall_curve(y_true, y_probs)
    ap = average_precision_score(y_true, y_probs)
    pr_auc = auc(recall, precision)

    sns.set_theme(style="whitegrid")
    sns.set_palette("tab10")

    plt.plot(
        recall,
        precision,
        alpha=0.8,
        marker="o",
        label=f"AP = {ap:.2f} | PR-AUC = {pr_auc:.2f}",
    )

    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.legend()
    plt.grid(True)
    plt.show()
