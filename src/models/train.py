# src/models/train.py

"""Model training module for the retail pipeline.

This module handles dataset loading, temporal data splitting, feature engineering,
model training across multiple algorithms, evaluation, and selection of the best performing model.
"""

from pathlib import Path
from time import perf_counter
import pandas as pd

from src.models.builder import build_model_pipeline
from src.models.evaluate import evaluate_model
from src.features.preprocess import RetailFeaturePipeline
from src.utils.helpers import setup_logger, load_params, load_dataset_from_db
from src.utils.paths import PROJECT_ROOT, DB_DIR, PARAMS_FILE

logger = setup_logger(__name__)


def split_train_valid(df: pd.DataFrame):
    """Split the dataset into training and validation sets based on a temporal quantile.

    Args:
        df (pd.DataFrame): The raw input dataframe containing a 'Date' column.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: A tuple containing the training dataframe
        and validation dataframe.
    """
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.sort_values("Date")

    split_date = df["Date"].quantile(0.8)
    train_df = df[df["Date"] < split_date]
    valid_df = df[df["Date"] >= split_date]

    return train_df, valid_df


def run_training():
    """Execute the full model training and evaluation pipeline.

    Loads configurations, prepares and splits data temporally, applies feature
    engineering, trains multiple machine learning models defined in params.yaml,
    evaluates them, and identifies the best model based on RMSE.

    Returns:
        tuple: A tuple containing:
            - best_pipeline: The best trained model pipeline.
            - best_metrics (dict): Evaluation metrics of the best model.
            - best_model_name (str): Name/identifier of the best model.
            - all_results (list[dict]): Results and metadata for all trained models.
    """
    logger.info("Starting training module")

    # Load params from central path
    params = load_params(PARAMS_FILE)

    # Load dataset
    df_raw = load_dataset_from_db(PROJECT_ROOT)

    # Split BEFORE FE
    train_raw, valid_raw = split_train_valid(df_raw)

    # Feature engineering
    fe = RetailFeaturePipeline(is_train=True)
    train_fe = fe.transform(train_raw)
    valid_fe = fe.transform(valid_raw)

    # Extract config
    target = params["train"]["target"]
    feature_columns = params["train"]["feature_set"]
    numeric_features = params["train"]["numeric_features"]
    categorical_features = params["train"]["categorical_features"]

    X_train = train_fe[feature_columns]
    y_train = train_fe[target]

    X_valid = valid_fe[feature_columns]
    y_valid = valid_fe[target]

    best_rmse = float("inf")
    best_model_name = None
    best_pipeline = None
    best_metrics = None

    all_results = []

    for model_name in params["train"]["models"]:
        logger.info(f"Training model: {model_name}")

        pipeline = build_model_pipeline(
            model_name=model_name,
            params=params,
            numeric_features=numeric_features,
            categorical_features=categorical_features,
        )

        start = perf_counter()
        pipeline.fit(X_train, y_train)
        train_time = perf_counter() - start

        preds = pipeline.predict(X_valid)
        metrics = evaluate_model(y_valid, preds)

        all_results.append(
            {
                "name": model_name,
                "model": pipeline,
                "metrics": metrics,
                "train_time": train_time,
            }
        )

        if metrics["rmse"] < best_rmse:
            best_rmse = metrics["rmse"]
            best_model_name = model_name
            best_pipeline = pipeline
            best_metrics = metrics

    logger.info(f"Best model: {best_model_name} | RMSE: {best_rmse:.4f}")

    return best_pipeline, best_metrics, best_model_name, all_results


if __name__ == "__main__":
    # 1. Execute the training pipeline
    best_pipeline, best_metrics, best_model_name, all_results = run_training()

    # 2. Print execution summary
    print("\n" + "=" * 50)
    print("MODEL TRAINING SUMMARY")
    print("=" * 50)
    for res in all_results:
        print(f"Model: {res['name']}")
        print(f"  - Metrics: {res['metrics']}")
        print(f"  - Training Time: {res['train_time']:.2f} seconds")

    print(f"BEST MODEL SELECTED: {best_model_name}")
    print(f"Best model metrics: {best_metrics}")
    print("=" * 50 + "\n")
