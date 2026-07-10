from __future__ import annotations

import pandas as pd


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    model_df = pd.get_dummies(
        df[
            [
                "encounter_id",
                "patient_nbr",
                "race",
                "gender",
                "age",
                "age_midpoint",
                "admission_type_id",
                "discharge_disposition_id",
                "admission_source_id",
                "time_in_hospital",
                "payer_code",
                "medical_specialty",
                "num_lab_procedures",
                "num_procedures",
                "num_medications",
                "number_outpatient",
                "number_emergency",
                "number_inpatient",
                "number_diagnoses",
                "max_glu_serum",
                "A1Cresult",
                "insulin",
                "change",
                "diabetesMed",
                "utilization_total",
                "target",
            ]
        ],
        columns=[
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
        ],
        drop_first=False,
        dtype=int,
    )

    return model_df


def align_feature_columns(df: pd.DataFrame, feature_columns: list[str]) -> pd.DataFrame:
    aligned = df.reindex(columns=feature_columns, fill_value=0)
    return aligned.astype(float)
