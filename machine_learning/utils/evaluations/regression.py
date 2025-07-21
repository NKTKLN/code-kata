import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.metrics import (
    explained_variance_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    root_mean_squared_error,
)


def evaluate_regression_models(
    predictions: dict[str, NDArray[np.float64]], y: NDArray[np.float64]
) -> pd.DataFrame:
    """Evaluate multiple regression model predictions on the given dataset.

    Args:
        predictions (dict[str, NDArray[np.float64]]): Dictionary where keys are
            model names and values are predicted target arrays corresponding to X.
        y (NDArray[np.float64]): True target values of shape (n_samples,).

    Returns:
        pd.DataFrame: DataFrame with evaluation metrics (MAE, MSE, RMSE, R², EVS) for
            each model. Columns correspond to model names; rows correspond to metrics.
    """
    evaluations = pd.DataFrame(
        columns=predictions.keys(), index=["MAE", "MSE", "RMSE", "R²", "EVS"]
    )

    for name, y_pred in predictions.items():
        metrics = {
            "MAE": mean_absolute_error(y, y_pred),
            "MSE": mean_squared_error(y, y_pred),
            "RMSE": root_mean_squared_error(y, y_pred),
            "R²": r2_score(y, y_pred),
            "EVS": explained_variance_score(y, y_pred),
        }

        evaluations.loc[:, name] = metrics

    return evaluations.astype(float)
