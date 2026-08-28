import argparse
from pathlib import Path
import joblib
import mlflow
import mlflow.sklearn
from mlflow import register_model
from mlflow.tracking import MlflowClient

from src.data.download_kaggle import download_raw_data
from src.data.sqlite_layer import create_sqlite_db
from src.models.train import run_training
from src.utils.helpers import setup_logger, load_params, load_environment

logger = setup_logger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parents[2]

mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_registry_uri("http://127.0.0.1:5000")
mlflow.set_experiment("retail-forecasting")


def ensure_raw_data() -> None:
    """Ensure that the raw data directory exists and contains files.

    Checks if the local `data/raw` directory exists and is not empty. If
    it is missing or empty, it triggers the download of the raw Kaggle dataset.
    Otherwise, it logs that the data is already present.
    """
    raw_path = PROJECT_ROOT / "data" / "raw"
    if not raw_path.exists() or not any(raw_path.iterdir()):
        logger.info("Raw data missing → downloading Kaggle dataset")
        download_raw_data()
    else:
        logger.info("Raw data already exists ✓")


def ensure_database() -> None:
    """Ensure that the SQLite database file exists.

    Checks if `db/sales.db` exists within the project directory. If missing,
    it creates the SQLite layer. Otherwise, it logs that the database is already present.
    """
    db_path = PROJECT_ROOT / "db" / "sales.db"
    if not db_path.exists():
        logger.info("Database missing → creating SQLite layer")
        create_sqlite_db()
    else:
        logger.info("Database already exists ✓")


client = MlflowClient()


from mlflow.exceptions import RestException


def get_current_champion(model_name: str) -> dict | None:
    """Retrieve the current champion model version from the MLflow Model Registry.

    This function queries the MLflow Model Registry to obtain the latest model
    version in either the "Production" or "Staging" stage for the specified
    registered model. If the model does not exist in the registry or no versions
    are available in those stages, the function returns ``None``.

    Args:
        model_name (str): Name of the registered model in MLflow.

    Returns:
        dict | None: A dictionary containing metadata about the current champion
        model
        Returns ``None`` if the model is not registered or no champion exists.

    Raises:
        None: Any MLflow registry lookup errors are internally handled and
        result in ``None`` being returned.
    """
    try:
        versions = client.get_latest_versions(
            model_name, stages=["Production", "Staging"]
        )
        if not versions:
            return None  # No champion yet

        champion = versions[0]
        run_id = champion.run_id
        run_data = client.get_run(run_id).data

        return {
            "version": champion.version,
            "stage": champion.current_stage,
            "metrics": run_data.metrics,
            "run_id": run_id,
            "model_uri": champion.source,
        }
    except RestException:
        # Model hasn't been registered in MLflow yet
        return None


def challenger_is_better(challenger_metrics: dict, champion_metrics: dict) -> bool:
    """Compare challenger and champion metrics to determine if the challenger performs better.

    Args:
        challenger_metrics (dict): Dictionary of evaluation metrics for the new model.
        champion_metrics (dict): Dictionary of evaluation metrics for the current champion.

    Returns:
        bool: True if the challenger's RMSE is lower than the champion's RMSE, False otherwise.
    """
    return challenger_metrics["rmse"] < champion_metrics["rmse"]


def run_full_pipeline() -> None:
    """Execute the complete machine learning training and MLOps pipeline.

    Handles raw data verification, database preparation, model training,
    logging runs to MLflow, evaluating challenger models against the existing
    champion in the Model Registry, promoting the best model if criteria are met,
    and saving the final production champion locally as a joblib pickle file.
    """

    ensure_raw_data()
    ensure_database()

    params = load_params(PROJECT_ROOT / "params.yaml")

    # Train all models
    best_model, best_metrics, best_name, all_results = run_training()

    # Log all models to MLflow Tracking
    best_run_id = None

    for result in all_results:
        with mlflow.start_run(run_name=f"{result['name']}-training") as run:
            mlflow.log_metrics(result["metrics"])
            mlflow.log_param("model_name", result["name"])
            mlflow.log_param("train_time", result["train_time"])
            mlflow.sklearn.log_model(
                sk_model=result["model"],
                artifact_path="model",
                serialization_format="pickle",
            )

            if result["name"] == best_name:
                best_run_id = run.info.run_id

    # Get current Champion
    client = MlflowClient()
    champion = get_current_champion(f"retail-{best_name}")

    if champion:
        logger.info(f"Current champion: v{champion['version']} ({champion['stage']})")
        logger.info(f"Available champion metrics: {list(champion['metrics'].keys())}")

        if not challenger_is_better(best_metrics, champion["metrics"]):
            logger.info(
                "The new model does NOT improve upon the Champion → keeping current champion."
            )
        else:
            logger.info(
                "The new model improves upon the Champion → registering and promoting."
            )

            # Register ONLY the best model
            with mlflow.start_run(run_name=f"{best_name}-registration") as run:
                mlflow.log_metrics(best_metrics)
                model_info = mlflow.sklearn.log_model(
                    sk_model=best_model,
                    artifact_path="model",
                    registered_model_name=f"retail-{best_name}",
                    serialization_format="pickle",
                )

            result = register_model(
                model_uri=model_info.model_uri,
                name=f"retail-{best_name}",
            )

            # Promote to Production
            client.transition_model_version_stage(
                name=result.name,
                version=result.version,
                stage="Production",
                archive_existing_versions=True,
            )
            logger.info(f"New Champion: {result.name} v{result.version} ✓")
    else:
        logger.info("No Champion found → registering the first model.")
        with mlflow.start_run(run_name=f"{best_name}-registration") as run:
            mlflow.log_metrics(best_metrics)
            model_info = mlflow.sklearn.log_model(
                sk_model=best_model,
                artifact_path="model",
                registered_model_name=f"retail-{best_name}",
                serialization_format="pickle",
            )

        result = register_model(
            model_uri=model_info.model_uri,
            name=f"retail-{best_name}",
        )

        client.transition_model_version_stage(
            name=result.name,
            version=result.version,
            stage="Production",
            archive_existing_versions=True,
        )
        logger.info(f"New Champion: {result.name} v{result.version} ✓")

    # Save Champion locally
    target_model_name = f"retail-{best_name}"
    champion_uri = f"models:/{target_model_name}/Production"

    logger.info(f"Downloading production champion from: {champion_uri}")
    champion_model = mlflow.sklearn.load_model(champion_uri)

    models_dir = PROJECT_ROOT / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    champion_local_path = models_dir / "champion.pkl"
    joblib.dump(champion_model, champion_local_path)

    logger.info(f"Champion saved locally at: {champion_local_path}")


if __name__ == "__main__":
    run_full_pipeline()
