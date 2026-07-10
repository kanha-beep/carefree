from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st
import torch
import sqlite3

from src.config import DB_FILE, FEATURE_FILE, MODEL_FILE, QUALITY_REPORT_FILE, RAW_FILE, SQL_REPORT_FILE


st.set_page_config(
    page_title="CareFlow AI Dashboard",
    layout="wide",
)


def load_dataframe(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    return pd.read_csv(path)


raw_df = load_dataframe(RAW_FILE)
feature_df = load_dataframe(FEATURE_FILE)
sql_report_df = load_dataframe(SQL_REPORT_FILE)
model_artifact_exists = MODEL_FILE.exists()
model_metrics = None
quality_report = None
feature_importance = None

if model_artifact_exists:
    checkpoint = torch.load(MODEL_FILE, map_location="cpu")
    model_metrics = checkpoint.get("metrics")
    feature_importance = checkpoint.get("feature_importance")

if QUALITY_REPORT_FILE.exists():
    quality_report = pd.read_json(QUALITY_REPORT_FILE, typ="series")

st.title("CareFlow AI")
st.caption("Hospital readmission risk analytics dashboard built on a real public healthcare dataset.")

if raw_df is None or feature_df is None:
    st.warning(
        "Pipeline outputs are missing. Run `py -m src.run_pipeline` and `py -m src.train` first."
    )
    st.stop()

target_rate = float(feature_df["target"].mean()) * 100
total_rows = int(len(raw_df))
feature_count = int(len(feature_df.columns) - 3)
avg_stay = float(raw_df["time_in_hospital"].replace("?", pd.NA).dropna().astype(float).mean())

col1, col2, col3, col4 = st.columns(4)
col1.metric("Hospital Encounters", f"{total_rows:,}")
col2.metric("Readmitted <30 Days", f"{target_rate:.2f}%")
col3.metric("Model Features", f"{feature_count}")
col4.metric("Avg Stay (Days)", f"{avg_stay:.2f}")

left, right = st.columns((1.2, 1))

with left:
    st.subheader("Readmission Distribution")
    readmission_counts = (
        raw_df["readmitted"].astype(str).value_counts().rename_axis("label").reset_index(name="count")
    )
    st.bar_chart(readmission_counts.set_index("label"))

    st.subheader("Operational Drivers")
    numeric_cols = [
        "time_in_hospital",
        "num_lab_procedures",
        "num_procedures",
        "num_medications",
        "number_outpatient",
        "number_emergency",
        "number_inpatient",
        "number_diagnoses",
    ]
    numeric_df = raw_df[numeric_cols].replace("?", pd.NA).apply(pd.to_numeric, errors="coerce")
    st.dataframe(numeric_df.describe().round(2), use_container_width=True)

with right:
    st.subheader("Top Medical Specialties")
    top_specialties = (
        raw_df["medical_specialty"]
        .fillna("Unknown")
        .astype(str)
        .replace("?", "Unknown")
        .value_counts()
        .head(10)
    )
    st.bar_chart(top_specialties)

    st.subheader("Data Assets")
    st.write(f"Raw dataset: `{RAW_FILE}`")
    st.write(f"Processed features: `{FEATURE_FILE}`")
    st.write(f"SQLite database: `{DB_FILE}`")
    st.write(f"Trained model available: `{'Yes' if model_artifact_exists else 'No'}`")
    if model_metrics:
        st.write("Latest model metrics:")
        st.json({key: round(value, 4) for key, value in model_metrics.items()})
    if feature_importance:
        st.write("Top predictive features:")
        st.dataframe(pd.DataFrame(feature_importance[:10]), use_container_width=True)
    if quality_report is not None:
        st.write("Latest data quality summary:")
        st.json(
            {
                "row_count": int(quality_report.get("row_count", 0)),
                "issue_count": int(quality_report.get("issue_count", 0)),
                "duplicate_rows": int(quality_report.get("duplicate_rows", 0)),
            }
        )

st.subheader("Preview")
preview_tab, features_tab, ops_tab = st.tabs(["Raw Data", "Model Features", "Pipeline Ops"])

with preview_tab:
    st.dataframe(raw_df.head(20), use_container_width=True)

with features_tab:
    st.dataframe(feature_df.head(20), use_container_width=True)

with ops_tab:
    st.subheader("SQL KPI Report")
    if sql_report_df is not None:
        st.dataframe(sql_report_df, use_container_width=True)
    else:
        st.info("Run `py -m src.run_workflow` to generate the SQL KPI report.")

    st.subheader("Recent Pipeline Runs")
    if DB_FILE.exists():
        with sqlite3.connect(DB_FILE) as connection:
            runs_df = pd.read_sql_query(
                """
                SELECT run_id, stage, status, records_processed, issue_count, created_at
                FROM pipeline_runs
                ORDER BY created_at DESC
                LIMIT 10
                """,
                connection,
            )
        st.dataframe(runs_df, use_container_width=True)
