"""Script to generate the final Kaggle submission CSV file using batch inference."""

from pathlib import Path
import pandas as pd
from api.predictor import SalesPredictor
from src.utils.paths import RAW_DATA_DIR
from src.utils.helpers import setup_logger

logger = setup_logger(__name__)


def generate_submission() -> None:
    """Loads the test dataset, runs optimized batch predictions, and saves the final submission file.

    Returns:
        None
    """
    logger.info("Loading test.csv...")
    test_df = pd.read_csv(RAW_DATA_DIR / "test.csv")
    logger.info(f"Test dataset loaded → shape={test_df.shape}")

    predictor = SalesPredictor()

    logger.info("Starting batch inference...")

    preds = predictor.predict_batch(test_df)

    logger.info("Constructing submission dataframe...")
    submission = pd.DataFrame(
        {
            "Id": (
                test_df["Store"].astype(str)
                + "_"
                + test_df["Dept"].astype(str)
                + "_"
                + test_df["Date"].astype(str)
            ),
            "Weekly_Sales": preds,
        }
    )

    output_path = Path("submission.csv")
    submission.to_csv(output_path, index=False)

    logger.info(f"Submission saved at {output_path}")


if __name__ == "__main__":
    generate_submission()
