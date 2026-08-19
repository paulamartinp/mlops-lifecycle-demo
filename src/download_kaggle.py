from pathlib import Path
import logging
import os

from dotenv import load_dotenv
import yaml
from kaggle.api.kaggle_api_extended import KaggleApi

from src.utils import setup_logger

PROJECT_ROOT = Path(__file__).resolve().parents[1]


# Initialize the logger for this module
logger = setup_logger(__name__)


def load_environment() -> None:
    """Load and validate Kaggle credentials from the .env file.

    Reads the KAGGLE_USERNAME and KAGGLE_KEY environment variables, ensuring
    that they are properly configured in the system to allow authentication
    with the Kaggle API.

    Raises:
        ValueError: If either of the Kaggle credentials is not defined.
    """
    load_dotenv()

    username = os.getenv("KAGGLE_USERNAME")
    key = os.getenv("KAGGLE_KEY")

    if not username or not key:
        raise ValueError(
            "KAGGLE_USERNAME and KAGGLE_KEY must be defined in .env"
        )

    # Explicitly set environment variables for the Kaggle client
    os.environ["KAGGLE_USERNAME"] = username
    os.environ["KAGGLE_KEY"] = key

    logger.info("Kaggle credentials loaded")


def load_params() -> dict:
    """Load the global configuration parameters of the project.

    Reads the `params.yaml` file located in the project root to extract
    configurations such as dataset identifiers and directory paths.

    Returns:
        dict: Dictionary containing the loaded configuration parameters.

    Raises:
        FileNotFoundError: If the `params.yaml` file does not exist in the root.
    """
    params_path = PROJECT_ROOT / "params.yaml"

    if not params_path.exists():
        raise FileNotFoundError(
            f"params.yaml not found: {params_path}"
        )

    with open(params_path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def create_output_directory(output_path: str) -> None:
    """Create the destination directory for data if it does not exist.

    Args:
        output_path (str): Path of the directory to verify or create.
    """
    Path(output_path).mkdir(
        parents=True,
        exist_ok=True,
    )

    logger.info("Output directory ready: %s", output_path)


def download_dataset(dataset: str, output_path: str) -> None:
    """Authenticate and download a remote dataset from Kaggle.

    Uses the official Kaggle API to automatically download and unzip
    the files of the specified dataset into the destination path.

    Args:
        dataset (str): Unique identifier of the dataset on Kaggle (e.g., 'user/dataset-name').
        output_path (str): Local path where the downloaded files will be stored.
    """
    logger.info("Authenticating Kaggle API")

    # Initialize and authenticate the Kaggle API
    api = KaggleApi()
    api.authenticate()

    logger.info("Downloading dataset: %s", dataset)

    # Automatic download and extraction of the dataset
    api.dataset_download_files(
        dataset=dataset,
        path=output_path,
        unzip=True,
    )

    # Count downloaded files for log validation
    downloaded_files = list(Path(output_path).glob("*"))

    logger.info(
        "Download completed. %s files found in %s",
        len(downloaded_files),
        output_path,
    )


def main() -> None:
    """Main function that orchestrates the data ingestion process."""
    logger.info("Starting data ingestion process")

    # 1. Load environment variables and credentials
    load_environment()

    # 2. Load configuration parameters from YAML
    params = load_params()

    # 3. Extract key variables for the download
    dataset = params["dataset"]["kaggle_id"]
    output_path = PROJECT_ROOT / params["paths"]["raw_data"]

    # 4. Ensure the output directory exists
    create_output_directory(output_path)

    # 5. Execute dataset download
    download_dataset(
        dataset=dataset,
        output_path=output_path,
    )

    logger.info("Data ingestion finished successfully")


if __name__ == "__main__":
    main()