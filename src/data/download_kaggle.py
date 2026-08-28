from pathlib import Path
import logging
import os

from dotenv import load_dotenv
import yaml
from kaggle.api.kaggle_api_extended import KaggleApi

from src.utils.helpers import setup_logger, load_params, load_environment
from src.utils.paths import RAW_DATA_DIR

# Initialize the logger for this module
logger = setup_logger(__name__)


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


def download_raw_data() -> None:
    """Main function that orchestrates the data ingestion process."""
    logger.info("Starting data ingestion process")

    # 1. Load environment variables and credentials
    load_environment()

    # 2. Load configuration parameters from YAML
    params = load_params()

    # 3. Extract key variables for the download
    dataset = params["dataset"]["kaggle_id"]
    output_path = RAW_DATA_DIR

    # 4. Ensure the output directory exists
    create_output_directory(output_path)

    # 5. Execute dataset download
    download_dataset(
        dataset=dataset,
        output_path=output_path,
    )

    logger.info("Data ingestion finished successfully")


if __name__ == "__main__":
    download_raw_data()
