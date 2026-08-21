from pathlib import Path
import logging
import sys
import pandas as pd
import yaml
import sqlite3

PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOGS_DIR = PROJECT_ROOT / "logs"


def setup_logger(name: str) -> logging.Logger:
    """Configures and returns a logger instance with console and file handlers.

    Args:
        name (str): The name of the logger, typically __name__ of the calling module.

    Returns:
        logging.Logger: Configured logger instance.
    """
    # Create logs directory if it doesn't exist
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    log_file_path = LOGS_DIR / "app.log"

    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # Prevent adding multiple handlers if the logger is already configured
    if logger.hasHandlers():
        return logger

    # Define standard log format
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    )

    # Handler for console output (stdout)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # Handler for file output
    file_handler = logging.FileHandler(log_file_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger


def load_params(path: Path) -> dict:
    """Load configuration parameters from a YAML file.

    Args:
        path (Path): Absolute or relative path to the YAML configuration file.

    Returns:
        dict: Dictionary containing configuration parameters.
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
