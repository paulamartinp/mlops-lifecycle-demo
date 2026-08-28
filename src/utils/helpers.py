import logging
import sys
import pandas as pd
from pathlib import Path
import yaml
import sqlite3
from dotenv import load_dotenv
import os
from src.utils.paths import LOGS_DIR, PARAMS_FILE, PROJECT_ROOT


def setup_logger(name: str) -> logging.Logger:
    """Configure and return a logger with both console and file handlers.

    Args:
        name (str): Name of the logger, typically `__name__` of the calling module.

    Returns:
        logging.Logger: Configured logger instance.
    """
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    log_file_path = LOGS_DIR / "app.log"

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    if logger.hasHandlers():
        return logger

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    )

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    file_handler = logging.FileHandler(log_file_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger


def load_params(path: Path = PARAMS_FILE) -> dict:
    """Load configuration parameters from a YAML file.

    Args:
        path (Path, optional): Path to the YAML parameter file. Defaults to PARAMS_FILE.

    Returns:
        dict: Dictionary containing the loaded configuration parameters.

    Raises:
        FileNotFoundError: If the parameter file does not exist.
    """
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_dataset_from_db(project_root: Path) -> pd.DataFrame:
    """Load the training dataset view directly from the SQLite database.

    Args:
        project_root (Path): The root directory of the project.

    Returns:
        pd.DataFrame: The loaded training dataset containing sales and features.
    """
    db_path = project_root / "db" / "sales.db"

    if not db_path.exists():
        raise FileNotFoundError(
            f"Database not found at {db_path}. Please run your database ingestion script first."
        )
    logger = setup_logger(__name__)
    logger.info("Loading dataset from SQLite database at %s", db_path)
    with sqlite3.connect(db_path) as conn:
        # Consultamos la vista analítica creada por tu script anterior
        df = pd.read_sql("SELECT * FROM train_dataset", conn)

    logger.info(
        "Dataset loaded successfully (%s rows, %s columns)", df.shape[0], df.shape[1]
    )
    return df


def load_environment() -> None:
    logger = setup_logger(__name__)

    # Cargar .env si existe
    load_dotenv(PROJECT_ROOT / ".env", override=True)

    # Detectar si hay API Key
    username = os.getenv("KAGGLE_USERNAME")
    key = os.getenv("KAGGLE_KEY")

    if username and key:
        logger.info("Using Kaggle API Key authentication")
    else:
        logger.info("Using Kaggle OAuth authentication (no API Key found)")
