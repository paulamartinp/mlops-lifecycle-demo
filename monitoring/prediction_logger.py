import pandas as pd
from datetime import datetime
from src.utils.paths import PROJECT_ROOT

MONITORING_DIR = PROJECT_ROOT / "monitoring"
LOGS_DIR = MONITORING_DIR / "logs"
LOGS_DIR.mkdir(parents=True, exist_ok=True)

LOG_FILE = LOGS_DIR / "prediction_logs.csv"


def log_prediction(payload: dict, prediction: float) -> None:
    """Appends a prediction event with a UTC timestamp and input features to the CSV log file.

    Args:
        payload (dict): Input features used for inference.
        prediction (float): Model output value.
    """
    event = {
        "timestamp": datetime.utcnow().isoformat(),
        **payload,
        "prediction": prediction,
    }

    pd.DataFrame([event]).to_csv(
        LOG_FILE, mode="a", header=not LOG_FILE.exists(), index=False
    )
