from pathlib import Path

"""Project path configuration module.

This module defines the central project root and standard directory paths
used across the entire codebase to prevent path resolution errors.
"""

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
DB_DIR = PROJECT_ROOT / "db"
MODELS_DIR = PROJECT_ROOT / "models"
LOGS_DIR = PROJECT_ROOT / "logs"
PARAMS_FILE = PROJECT_ROOT / "params.yaml"
