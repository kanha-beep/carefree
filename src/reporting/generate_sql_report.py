from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd

from src.config import DB_FILE, REPORT_DIR, SQL_REPORT_FILE, BASE_DIR


def generate_sql_kpi_report() -> pd.DataFrame:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    query_file = BASE_DIR / "sql" / "analytics_queries.sql"
    query_text = query_file.read_text(encoding="utf-8")
    query_blocks = [block.strip() for block in query_text.split(";") if block.strip()]
    rows: list[dict[str, object]] = []
    with sqlite3.connect(DB_FILE) as connection:
        for index, query in enumerate(query_blocks, start=1):
            lines = [line.strip() for line in query.splitlines() if line.strip()]
            title = lines[0].replace("--", "").strip() if lines and lines[0].startswith("--") else f"query_{index}"
            sql = "\n".join(line for line in lines if not line.startswith("--"))
            result_df = pd.read_sql_query(sql, connection)
            result_df.insert(0, "query_name", title)
            rows.append(result_df)

    report_df = pd.concat(rows, ignore_index=True)
    report_df.to_csv(SQL_REPORT_FILE, index=False)
    return report_df


if __name__ == "__main__":
    result = generate_sql_kpi_report()
    print(result.to_string(index=False))
