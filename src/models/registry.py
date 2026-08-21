"""Model registry and factory module.

This module provides factory functions to instantiate machine learning 
regression models based on configuration dictionaries.
"""

from typing import Any, Dict, Union
from lightgbm import LGBMRegressor
from sklearn.dummy import DummyRegressor
from xgboost import XGBRegressor


def get_model(
    model_name: str, 
    params: Dict[str, Any]
) -> Union[XGBRegressor, LGBMRegressor, DummyRegressor]:
    """Instantiate a machine learning regression model with given parameters.

    Args:
        model_name: The identifier of the model to instantiate. 
            Supported values are 'baseline', 'xgboost', and 'lightgbm'.
        params: A nested configuration dictionary containing training 
            and model-specific hyperparameters.

    Returns:
        An instantiated scikit-learn compatible regression model.

    Raises:
        ValueError: If the provided model_name does not match any 
            supported models.
    """
    # Extract global training configurations with safe defaults
    train_config = params.get("train", {})
    random_state = train_config.get("random_state", 42)

    if model_name == "baseline":
        strategy = train_config.get("baseline", {}).get("strategy", "mean")
        return DummyRegressor(strategy=strategy)

    if model_name == "xgboost":
        xgb_params = train_config.get("xgboost", {})
        return XGBRegressor(**xgb_params, random_state=random_state)

    if model_name == "lightgbm":
        lgb_params = train_config.get("lightgbm", {})
        return LGBMRegressor(**lgb_params, random_state=random_state, verbose=-1)

    raise ValueError(f"Unknown model name: {model_name}. Supported models: baseline, xgboost, lightgbm.")