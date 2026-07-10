from __future__ import annotations

import sqlite3
from datetime import datetime, UTC

import pandas as pd

from src.config import DB_FILE, SQL_SCHEMA_FILE


def initialize_database() -> None:
    DB_FILE.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(DB_FILE) as connection:
        schema = SQL_SCHEMA_FILE.read_text(encoding="utf-8")
        connection.executescript(schema)


def save_dataframe(df: pd.DataFrame, table_name: str) -> None:
    with sqlite3.connect(DB_FILE) as connection:
        df.to_sql(table_name, connection, if_exists="replace", index=False)


def record_pipeline_run(
    run_id: str,
    stage: str,
    status: str,
    records_processed: int,
    issue_count: int,
    notes: str,
) -> None:
    query = """
        INSERT INTO pipeline_runs (
            run_id, stage, status, records_processed, issue_count, created_at, notes
        ) VALUES (?, ?, ?, ?, ?, ?, ?)
    """
    values = (
        run_id,
        stage,
        status,
        records_processed,
        issue_count,
        datetime.now(UTC).isoformat(),
        notes,
    )
    with sqlite3.connect(DB_FILE) as connection:
        connection.execute(query, values)
        connection.commit()
