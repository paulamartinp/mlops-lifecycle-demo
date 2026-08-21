import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from src.utils.helpers import setup_logger

logger = setup_logger(__name__)


class RetailFeaturePipeline(BaseEstimator, TransformerMixin):
    """Unified feature engineering and preprocessing pipeline

    for retail sales forecasting (training and inference).
    """

    def __init__(self, is_train: bool = True):
        """Initialize the pipeline.

        Args:
            is_train (bool): Flag indicating if the pipeline runs for training
              (True) or inference (False).
        """
        self.is_train = is_train

    def fit(self, X: pd.DataFrame, y=None) -> "RetailFeaturePipeline":
        """Fit the pipeline.

        No statistical fitting is required, maintained for scikit-learn
        compatibility.

        Args:
            X (pd.DataFrame): Input training features.
            y: Target variable (optional).

        Returns:
            RetailFeaturePipeline: Fitted instance.
        """
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply feature engineering, data typing, and preprocessing.

        Args:
            X (pd.DataFrame): Raw input DataFrame.

        Returns:
            pd.DataFrame: Transformed and type-casted DataFrame.
        """
        df = X.copy()

        # 1. Date parsing and chronological sorting
        if "Date" in df.columns:
            df["Date"] = pd.to_datetime(df["Date"])
            if {"Store", "Dept"}.issubset(df.columns):
                df = df.sort_values(by=["Store", "Dept", "Date"]).reset_index(drop=True)

        # 2. Calendar features extraction
        if "Date" in df.columns:
            df["Month"] = df["Date"].dt.month.astype("int32")
            df["Week"] = df["Date"].dt.isocalendar().week.astype("int32")
            df["Quarter"] = df["Date"].dt.quarter.astype("int32")

        # 3. Lag and rolling features calculation with historical checks
        if {"Store", "Dept", "Weekly_Sales"}.issubset(df.columns):
            if not self.is_train:
                min_records = df.groupby(["Store", "Dept"]).size().min()
                if min_records < 5:
                    logger.warning(
                        "Some series have fewer than 5 historical records. "
                        "Lag and rolling features will contain NaN values."
                    )

            df["lag_1"] = df.groupby(["Store", "Dept"])["Weekly_Sales"].shift(1)
            df["lag_4"] = df.groupby(["Store", "Dept"])["Weekly_Sales"].shift(4)
            df["rolling_mean_4"] = df.groupby(["Store", "Dept"])[
                "Weekly_Sales"
            ].transform(lambda x: x.shift(1).rolling(4).mean())

            # Ensure numeric stability (float32 for memory efficiency)
            for col in ["lag_1", "lag_4", "rolling_mean_4"]:
                df[col] = df[col].astype("float32")

        # 4. Markdown imputation (only fill specific promotional NaNs with 0)
        markdown_cols = [
            "MarkDown1",
            "MarkDown2",
            "MarkDown3",
            "MarkDown4",
            "MarkDown5",
        ]
        existing_markdowns = [col for col in markdown_cols if col in df.columns]
        if existing_markdowns:
            logger.info("Imputing missing markdown values with zero.")
            df[existing_markdowns] = df[existing_markdowns].fillna(0).astype("float32")

        # 5. Categorical encoding (preserving pandas category type for tree-based models)
        if "Store_Type" in df.columns:
            logger.info("Converting 'Store_Type' to pandas category dtype.")
            df["Store_Type"] = df["Store_Type"].astype("category")

        remaining_missing = int(df.isna().sum().sum())
        logger.info(
            "Pipeline transformation completed. Total remaining missing values: %s",
            remaining_missing,
        )
        # 6. Remove raw Date column before sklearn preprocessing
        if "Date" in df.columns:
            df = df.drop(columns=["Date"])

        return df
