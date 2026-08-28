import pandas as pd
from evidently import Report
from evidently.presets import DataDriftPreset
from src.utils.paths import PROJECT_ROOT, DATA_DIR, PARAMS_FILE
import yaml

MONITORING_DIR = PROJECT_ROOT / "monitoring"
REFERENCE_DIR = MONITORING_DIR / "reference"
LOGS_DIR = MONITORING_DIR / "logs"
REPORTS_DIR = MONITORING_DIR / "reports"

REFERENCE_PATH = REFERENCE_DIR / "reference_data.parquet"
PREDICTIONS_PATH = LOGS_DIR / "prediction_logs.csv"
REPORT_PATH = REPORTS_DIR / "drift_report.html"

REFERENCE_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)


def create_reference_dataset() -> pd.DataFrame:
    """Samples and saves a baseline reference dataset from historical features based on project parameters."""
    with open(PARAMS_FILE, "r") as f:
        params = yaml.safe_load(f)

    feature_columns = params["train"]["feature_set"]
    source_path = DATA_DIR / "features" / "forecast_features.parquet"
    df = pd.read_parquet(source_path)

    reference_df = df[feature_columns].sample(
        n=min(10_000, len(df)),
        random_state=42,
    )

    reference_df.to_parquet(REFERENCE_PATH, index=False)
    return reference_df


def load_reference() -> pd.DataFrame:
    """Loads the reference dataset, generating and saving a baseline if it does not already exist."""
    if not REFERENCE_PATH.exists():
        print("Reference dataset not found. Creating baseline...")
        return create_reference_dataset()

    return pd.read_parquet(REFERENCE_PATH)


def load_production() -> pd.DataFrame:
    """Loads current production data logs used for monitoring."""
    return pd.read_csv(PREDICTIONS_PATH)


def generate_drift_report() -> str:
    """Generates an Evidently data drift report comparing reference and production data, saving it as HTML."""
    ref = load_reference()
    prod = load_production()

    common = [c for c in ref.columns if c in prod.columns]
    ref = ref[common]
    prod = prod[common]

    report = Report(metrics=[DataDriftPreset()])

    # Run the report and capture the output result object
    result = report.run(reference_data=ref, current_data=prod)

    # Save HTML on the result object
    result.save_html(str(REPORT_PATH))

    return str(REPORT_PATH)


if __name__ == "__main__":
    path = generate_drift_report()
    print(f"Drift report generated at: {path}")
