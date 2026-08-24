from pathlib import Path
import sqlite3

import pandas as pd

from src.utils.helpers import setup_logger
from src.utils.paths import RAW_DATA_DIR, DB_DIR

logger = setup_logger(__name__)


def load_raw_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load raw Walmart datasets from locally stored CSV files.

    Reads the files corresponding to sales (train), test, store features,
    and store metadata from the raw data directory.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]: A tuple containing
            the sales, test, features, and stores DataFrames respectively.
    """

    # --- Validate presence of expected CSV files ---
    expected_files = ["train.csv", "test.csv", "features.csv", "stores.csv"]
    for filename in expected_files:
        file_path = RAW_DATA_DIR / filename

        # Check file exists
        if not file_path.exists():
            logger.error(f"Missing required file: {file_path}")
            raise FileNotFoundError(f"Required file not found: {file_path}")

        # Check file size in MB
        size_mb = file_path.stat().st_size / (1024 * 1024)
        if size_mb < 0.0001:  # ~1 KB threshold
            logger.error(f"File {filename} is too small ({size_mb:.4f} MB)")
            raise ValueError(f"File {filename} is empty or corrupted.")

        logger.info(f"Validated file {filename} ({size_mb:.2f} MB)")

    # --- Load CSVs ---
    sales_df = pd.read_csv(RAW_DATA_DIR / "train.csv")
    test_df = pd.read_csv(RAW_DATA_DIR / "test.csv")
    features_df = pd.read_csv(RAW_DATA_DIR / "features.csv")
    stores_df = pd.read_csv(RAW_DATA_DIR / "stores.csv")

    logger.info("Raw files validated and loaded successfully")

    return sales_df, test_df, features_df, stores_df


def create_connection() -> sqlite3.Connection:
    """Create and ensure connection to the local SQLite database.

    Returns:
        sqlite3.Connection: Active database connection object.
    """
    DB_DIR.mkdir(parents=True, exist_ok=True)

    db_path = DB_DIR / "sales.db"
    logger.info("Connecting to database at %s", db_path)

    return sqlite3.connect(db_path)


def load_tables(
    conn: sqlite3.Connection,
    sales_df: pd.DataFrame,
    test_df: pd.DataFrame,
    features_df: pd.DataFrame,
    stores_df: pd.DataFrame,
) -> None:
    """Load Pandas DataFrames into relational tables inside SQLite.

    Args:
        conn (sqlite3.Connection): Active SQLite database connection.
        sales_df (pd.DataFrame): DataFrame containing training sales information.
        test_df (pd.DataFrame): DataFrame containing test information.
        features_df (pd.DataFrame): DataFrame containing additional features.
        stores_df (pd.DataFrame): DataFrame containing store metadata.
    """
    tables = {
        "sales": sales_df,
        "test": test_df,
        "features": features_df,
        "stores": stores_df,
    }

    for table_name, df in tables.items():
        df.to_sql(table_name, conn, if_exists="replace", index=False)

    logger.info("Tables successfully loaded into SQLite")


def create_views(conn: sqlite3.Connection) -> None:
    """Create analytical views by executing an external SQL script.

    Locates the 'create_views.sql' file, reads its contents, and executes them
    in the database to structure the model's analytical queries.

    Args:
        conn (sqlite3.Connection): Active SQLite database connection.

    Raises:
        FileNotFoundError: If the SQL script file is not found in the expected path.
    """
    sql_file = DB_DIR / "sql" / "create_views.sql"

    if not sql_file.exists():
        raise FileNotFoundError(f"SQL file not found at: {sql_file}")

    with open(sql_file, "r", encoding="utf-8") as file:
        sql_script = file.read()

    conn.executescript(sql_script)
    logger.info("Views created successfully")


def validate_database(conn: sqlite3.Connection) -> None:
    """Validate that the main train and test analytical views contain valid records.

    Args:
        conn (sqlite3.Connection): Active SQLite database connection.
    """
    cursor = conn.cursor()

    cursor.execute("SELECT COUNT(*) FROM train_dataset")
    train_rows = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(*) FROM test_dataset")
    test_rows = cursor.fetchone()[0]

    logger.info("Train dataset rows: %s", train_rows)
    logger.info("Test dataset rows: %s", test_rows)


def create_sqlite_db() -> None:
    """Main function that orchestrates the construction of the SQLite analytical layer."""
    logger.info("Building SQLite analytics layer")

    sales_df, test_df, features_df, stores_df = load_raw_data()

    with create_connection() as conn:
        load_tables(conn, sales_df, test_df, features_df, stores_df)
        create_views(conn)
        validate_database(conn)

    logger.info("SQLite layer built successfully")


if __name__ == "__main__":
    create_sqlite_db()
