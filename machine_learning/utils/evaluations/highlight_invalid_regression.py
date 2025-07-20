import numpy as np
import pandas as pd


def highlight_invalid_regression_metrics(column: pd.Series) -> list[str]:
    """Returns a list of CSS styles to highlight regression metric values.

    Highlights depend on the metric type (column name) and value ranges:
    - Green: good values,
    - Orange: warning,
    - Red: poor values.

    Args:
        column (pd.Series): Metric values.

    Returns:
        list[str]: CSS styles for each cell in the column.
    """
    metric = column.name
    highlight_styles = []
    for val in column:
        if val == "NaN" or (isinstance(val, float) and np.isnan(val)):
            highlight_styles.append("")
            continue

        try:
            val_float = float(val)
        except (ValueError, TypeError):
            highlight_styles.append("")
            continue

        # R²
        if metric == "R²":
            if val_float < 0.5:
                style = "background-color: red; color: white"
            elif val_float <= 0.8:
                style = "background-color: orange; color: white"
            else:
                style = "background-color: green; color: white"

        # MAPE / WAPE / RMSLE
        elif metric in {"MAPE %", "RMSLE %", "WAPE %"}:
            if val_float > 20:
                style = "background-color: red; color: white"
            elif val_float >= 10:
                style = "background-color: orange; color: white"
            else:
                style = "background-color: green; color: white"

        else:
            style = ""

        highlight_styles.append(style)

    return highlight_styles
