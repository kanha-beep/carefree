from __future__ import annotations

import pandas as pd

from src.config import REQUIRED_COLUMNS


def validate_raw_data(df: pd.DataFrame) -> list[str]:
    issues: list[str] = []

    missing_columns = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    if missing_columns:
        issues.append(f"Missing columns: {missing_columns}")
        return issues

    if df.empty:
        issues.append("Dataset is empty.")

    duplicate_count = int(df.duplicated().sum())
    if duplicate_count:
        issues.append(f"Found {duplicate_count} duplicated rows.")

    missing_summary = df[REQUIRED_COLUMNS].isnull().sum()
    missing_summary = missing_summary[missing_summary > 0]
    if not missing_summary.empty:
        issues.append(f"Missing values found: {missing_summary.to_dict()}")

    if (df["time_in_hospital"].astype(str) == "?").any():
        issues.append("time_in_hospital contains placeholder values.")

    unexpected_targets = sorted(set(df["readmitted"].astype(str)) - {"NO", "<30", ">30"})
    if unexpected_targets:
        issues.append(f"Unexpected target labels: {unexpected_targets}")

    return issues
