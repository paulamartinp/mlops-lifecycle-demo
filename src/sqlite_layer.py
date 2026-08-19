from pathlib import Path
import sqlite3

import pandas as pd


from src.utils import setup_logger
logger = setup_logger(__name__)

# Define project root directory (goes up 1 level from the src/ folder)
PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_raw_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load raw Walmart datasets from locally stored CSV files.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: A tuple containing
            the sales, features, and stores DataFrames respectively.
    """
    raw_path = PROJECT_ROOT / "data" / "raw"

    sales_df = pd.read_csv(raw_path / "train.csv")
    features_df = pd.read_csv(raw_path / "features.csv")
    stores_df = pd.read_csv(raw_path / "stores.csv")

    logger.info("Raw files loaded successfully")

    return sales_df, features_df, stores_df


def create_connection() -> sqlite3.Connection:
    """Create and ensure connection to the local SQLite database.

    Returns:
        sqlite3.Connection: Active database connection object.
    """
    db_dir = PROJECT_ROOT / "db"
    db_dir.mkdir(parents=True, exist_ok=True)

    db_path = db_dir / "sales.db"
    logger.info("Connecting to database at %s", db_path)

    return sqlite3.connect(db_path)


def load_tables(
    conn: sqlite3.Connection,
    sales_df: pd.DataFrame,
    features_df: pd.DataFrame,
    stores_df: pd.DataFrame,
) -> None:
    """Load Pandas DataFrames into relational tables inside SQLite.

    Inserts or replaces the 'sales', 'features', and 'stores' tables using
    the provided data.

    Args:
        conn (sqlite3.Connection): Active SQLite database connection.
        sales_df (pd.DataFrame): DataFrame containing sales information.
        features_df (pd.DataFrame): DataFrame containing additional features.
        stores_df (pd.DataFrame): DataFrame containing store metadata.
    """
    tables = {
        "sales": sales_df,
        "features": features_df,
        "stores": stores_df,
    }

    # Iterate over the dictionary to dump each DataFrame into its respective table
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
    sql_file = PROJECT_ROOT / "db" / "sql" / "create_views.sql"

    if not sql_file.exists():
        raise FileNotFoundError(f"SQL file not found at: {sql_file}")

    with open(sql_file, "r", encoding="utf-8") as file:
        sql_script = file.read()

    # Execute the complete SQL script sequentially
    conn.executescript(sql_script)
    logger.info("Views created successfully")


def validate_database(conn: sqlite3.Connection) -> None:
    """Validate that the main analytical views contain valid records.

    Args:
        conn (sqlite3.Connection): Active SQLite database connection.
    """
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM forecasting_dataset")
    total_rows = cursor.fetchone()[0]

    logger.info("Forecasting dataset rows: %s", total_rows)


def main() -> None:
    """Main function that orchestrates the construction of the SQLite analytical layer."""
    logger.info("Building SQLite analytics layer")

    # 1. Load original CSV files
    sales_df, features_df, stores_df = load_raw_data()

    # 2. Manage database connection using a context manager to ensure safe closing
    with create_connection() as conn:
        load_tables(conn, sales_df, features_df, stores_df)
        create_views(conn)
        validate_database(conn)

    logger.info("SQLite layer built successfully")


if __name__ == "__main__":
    main()