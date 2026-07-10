from __future__ import annotations

import json
from datetime import datetime, UTC

import pandas as pd

from src.config import QUALITY_REPORT_FILE, REPORT_DIR


def build_quality_report(df: pd.DataFrame, issues: list[str]) -> dict[str, object]:
    required_missing = df.isnull().sum()
    missing_columns = {
        column: int(count) for column, count in required_missing.items() if int(count) > 0
    }

    report = {
        "generated_at": datetime.now(UTC).isoformat(),
        "row_count": int(len(df)),
        "column_count": int(len(df.columns)),
        "duplicate_rows": int(df.duplicated().sum()),
        "issue_count": len(issues),
        "issues": issues,
        "missing_values": missing_columns,
        "target_distribution": df["readmitted"].astype(str).value_counts().to_dict()
        if "readmitted" in df.columns
        else {},
    }
    return report


def save_quality_report(report: dict[str, object]) -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    QUALITY_REPORT_FILE.write_text(json.dumps(report, indent=2), encoding="utf-8")
