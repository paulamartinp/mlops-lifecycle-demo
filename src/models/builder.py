"""
This module provides functions to build end-to-end scikit-learn pipelines
that combine custom feature engineering, preprocessing steps, and ML models.
"""

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder

from src.models.registry import get_model
from src.features.preprocess import RetailFeaturePipeline


def build_preprocessor(
    numeric_features: list[str], categorical_features: list[str]
) -> ColumnTransformer:
    """Build a column-wise preprocessing transformer for numerical and categorical features.

    Args:
        numeric_features: List of column names corresponding to continuous variables.
        categorical_features: List of column names corresponding to categorical variables.

    Returns:
        A configured ColumnTransformer that imputes numerical features (median) and
        categorical features (most frequent) with ordinal encoding.
    """
    numeric_transformer = Pipeline([("imputer", SimpleImputer(strategy="median"))])

    # Reemplazamos/añadimos el OrdinalEncoder para transformar texto a enteros
    categorical_transformer = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "ordinal",
                OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
            ),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ],
        remainder="passthrough",  # Preserve any columns not explicitly specified
    )


def build_model_pipeline(
    model_name: str,
    params: dict,
    numeric_features: list[str],
    categorical_features: list[str],
) -> Pipeline:
    """Construct an end-to-end ML training pipeline.

    The pipeline executes custom feature engineering, automated column preprocessing, and the specified estimator.

    Args:
        model_name: The string identifier of the model to instantiate.
        params: Configuration dictionary containing global and model hyperparameters.
        numeric_features: List of numerical column names to be preprocessed.
        categorical_features: List of categorical column names to be preprocessed.

    Returns:
        A scikit-learn Pipeline encapsulating feature engineering, preprocessing,
        and the final model estimator.
    """
    # Instantiate the base regression estimator using the model registry
    model = get_model(model_name, params)

    # Build the transformation matrix for tabular columns
    preprocessor = build_preprocessor(
        numeric_features=numeric_features, categorical_features=categorical_features
    )

    # Assemble the sequential execution steps
    pipeline = Pipeline(
        [
            ("feature_engineering", RetailFeaturePipeline(is_train=True)),
            ("preprocessing", preprocessor),
            ("model", model),
        ]
    )

    return pipeline
