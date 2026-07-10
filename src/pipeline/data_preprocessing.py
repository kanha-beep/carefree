from __future__ import annotations

import re

import pandas as pd


def _age_bucket_to_midpoint(bucket: str) -> float:
    match = re.findall(r"\d+", str(bucket))
    if len(match) < 2:
        raise ValueError(f"Unexpected age bucket format: {bucket}")
    start, end = match[:2]
    return (float(start) + float(end)) / 2


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    clean_df = df.copy()
    clean_df = clean_df.replace("?", pd.NA)

    text_columns = [
        "race",
        "gender",
        "age",
        "payer_code",
        "medical_specialty",
        "max_glu_serum",
        "A1Cresult",
        "insulin",
        "change",
        "diabetesMed",
        "readmitted",
    ]
    for column in text_columns:
        clean_df[column] = clean_df[column].astype("string[python]").str.strip()

    numeric_columns = [
        "time_in_hospital",
        "num_lab_procedures",
        "num_procedures",
        "num_medications",
        "number_outpatient",
        "number_emergency",
        "number_inpatient",
        "number_diagnoses",
        "admission_type_id",
        "discharge_disposition_id",
        "admission_source_id",
    ]
    for column in numeric_columns:
        clean_df[column] = pd.to_numeric(clean_df[column], errors="coerce")

    clean_df = clean_df[clean_df["gender"].str.lower() != "unknown/invalid"].copy()
    clean_df = clean_df.dropna(subset=["readmitted", "age"])

    clean_df["race"] = clean_df["race"].fillna("Unknown")
    clean_df["payer_code"] = clean_df["payer_code"].fillna("Unknown")
    clean_df["medical_specialty"] = clean_df["medical_specialty"].fillna("Unknown")
    clean_df["max_glu_serum"] = clean_df["max_glu_serum"].fillna("None")
    clean_df["A1Cresult"] = clean_df["A1Cresult"].fillna("None")
    clean_df["insulin"] = clean_df["insulin"].fillna("No")
    clean_df["change"] = clean_df["change"].fillna("No")
    clean_df["diabetesMed"] = clean_df["diabetesMed"].fillna("No")

    clean_df["age_midpoint"] = clean_df["age"].map(_age_bucket_to_midpoint)
    clean_df["utilization_total"] = (
        clean_df["number_outpatient"].fillna(0)
        + clean_df["number_emergency"].fillna(0)
        + clean_df["number_inpatient"].fillna(0)
    )
    clean_df["target"] = (clean_df["readmitted"] == "<30").astype(int)

    return clean_df.drop_duplicates(subset=["encounter_id"])
