import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from numpy.typing import NDArray
from sklearn.metrics import roc_auc_score, roc_curve


def plot_roc_curve(y_true: NDArray[np.int64], y_probs: NDArray[np.int64]) -> None:
    """Plot the ROC curve and display the ROC-AUC score.

    Args:
        y_true (NDArray[np.int64]): True binary class labels.
        y_probs (NDArray[np.float64]): Predicted probabilities for the positive class.
    """
    fpr, tpr, _ = roc_curve(y_true, y_probs)
    auc = roc_auc_score(y_true, y_probs)

    sns.set_theme(style="whitegrid")
    sns.set_palette("tab10")

    plt.plot(fpr, tpr, alpha=0.8, marker="o", label=f"AUC = {auc:.2f}")
    plt.plot([0, 1], [0, 1], alpha=0.8, linestyle="--", label="Random guessing")

    plt.xlabel("FPR (False Positive Rate)")
    plt.ylabel("TPR (True Positive Rate)")
    plt.title("ROC Curve")
    plt.legend()
    plt.grid(True)
    plt.show()
