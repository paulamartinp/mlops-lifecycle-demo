import joblib
import pandas as pd
from src.utils.paths import RAW_DATA_DIR, MODELS_DIR
from src.utils.helpers import load_params, setup_logger

logger = setup_logger(__name__)


class SalesPredictor:
    """Predictor that replicates the feature engineering pipeline used during training.

    Attributes:
        params (dict): Loaded configuration parameters.
        feature_columns (list): List of feature names expected by the model.
        stores_df (pd.DataFrame): Metadata for stores.
        features_df (pd.DataFrame): Additional contextual features.
        model (Pipeline): Trained machine learning champion model.
    """

    def __init__(self):
        # Load params
        self.params = load_params()
        self.feature_columns = self.params["train"]["feature_set"]

        # Load metadata tables
        self.stores_df = pd.read_csv(RAW_DATA_DIR / "stores.csv")
        self.features_df = pd.read_csv(RAW_DATA_DIR / "features.csv")

        # Load champion model (pipeline included)
        model_path = MODELS_DIR / "champion.pkl"
        if not model_path.exists():
            raise FileNotFoundError(f"Champion model not found at {model_path}")

        logger.info(f"Loading champion model from {model_path}")
        self.model = joblib.load(model_path)

    def process_raw_dataframe(self, raw_df: pd.DataFrame) -> pd.DataFrame:
        """Replicates the feature engineering process prior to model inference.

        Args:
            raw_df (pd.DataFrame): Raw input dataframe containing raw records.

        Returns:
            pd.DataFrame: Processed dataframe containing exactly the features expected by the model.
        """
        # Merge metadata
        df = raw_df.merge(
            self.features_df, on=["Store", "Date", "IsHoliday"], how="left"
        )
        df = df.merge(self.stores_df, on="Store", how="left")

        # Temporal features
        df["Date"] = pd.to_datetime(df["Date"])
        df["Month"] = df["Date"].dt.month.astype("int32")
        df["Week"] = df["Date"].dt.isocalendar().week.astype("int64")
        df["Quarter"] = df["Date"].dt.quarter.astype("int32")
        df["IsHoliday"] = df["IsHoliday"].astype("int64")

        # Fill NA
        df = df.fillna(0.0)

        # Standardize naming
        if "Type" in df.columns:
            df["Store_Type"] = df["Type"]
        if "Size" in df.columns:
            df["Store_Size"] = df["Size"]

        # Select feature set EXACTLY as in training
        return df[self.feature_columns]

    def predict_single(self, raw_df: pd.DataFrame) -> float:
        """Predicts weekly sales for a single raw input dataframe.

        Args:
            raw_df (pd.DataFrame): Raw input dataframe representing a single record.

        Returns:
            float: Predicted weekly sales amount.
        """
        X_test = self.process_raw_dataframe(raw_df)
        prediction = float(self.model.predict(X_test)[0])

        return prediction

    def predict_batch(self, raw_df: pd.DataFrame) -> list[float]:
        """Predicts weekly sales for a batch of raw input records.

        Args:
            raw_df (pd.DataFrame): Raw input dataframe containing multiple records.

        Returns:
            list[float]: List of predicted weekly sales amounts.
        """
        X_test = self.process_raw_dataframe(raw_df)
        predictions = self.model.predict(X_test)
        return predictions.tolist()
