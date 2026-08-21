"""Evaluation utilities for model performance assessment.

This module provides metric calculation functions tailored for 
regression models using scikit-learn metrics.
"""

from typing import Dict, Union
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, root_mean_squared_error

# Define a reusable type alias for array-like numerical inputs
ArrayLike = Union[pd.Series, np.ndarray]


def evaluate_model(y_true: ArrayLike, y_pred: ArrayLike) -> Dict[str, float]:
    """Calculate standard regression evaluation metrics.

    Args:
        y_true: Ground truth (correct) target values. Accepts pandas 
            Series or NumPy arrays.
        y_pred: Estimated target values corresponding to y_true. 
            Accepts pandas Series or NumPy arrays.

    Returns:
        A dictionary containing the calculated metric names as keys 
        and their corresponding float values. Currently computes:
            - 'mae': Mean Absolute Error
            - 'rmse': Root Mean Squared Error
    """
    # Cast metrics explicitly to standard Python floats to ensure 
    # JSON serializability (e.g., when logging to MLflow or saving to JSON).
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(root_mean_squared_error(y_true, y_pred)),
    }