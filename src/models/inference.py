from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch

from src.config import MODEL_FILE
from src.models.pytorch_model import ReadmissionPredictor
from src.pipeline.data_preprocessing import preprocess_data
from src.pipeline.feature_engineering import align_feature_columns, create_features


@dataclass
class LoadedModelArtifacts:
    model: ReadmissionPredictor
    feature_columns: list[str]
    train_mean: np.ndarray
    train_std: np.ndarray
    metrics: dict[str, float]
    feature_importance: list[dict[str, float]]
    threshold: float


def load_model_artifacts() -> LoadedModelArtifacts:
    checkpoint = torch.load(MODEL_FILE, map_location="cpu")
    feature_columns = list(checkpoint["feature_columns"])
    model = ReadmissionPredictor(input_dim=len(feature_columns))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return LoadedModelArtifacts(
        model=model,
        feature_columns=feature_columns,
        train_mean=np.array(checkpoint["train_mean"], dtype=float),
        train_std=np.array(checkpoint["train_std"], dtype=float),
        metrics=dict(checkpoint["metrics"]),
        feature_importance=list(checkpoint.get("feature_importance", [])),
        threshold=float(checkpoint.get("threshold", 0.5)),
    )


def build_prediction_frame(payload: dict[str, object]) -> pd.DataFrame:
    defaults = {
        "encounter_id": "API-ENC-1",
        "patient_nbr": "API-PAT-1",
        "race": "Caucasian",
        "gender": "Female",
        "age": "[50-60)",
        "admission_type_id": 1,
        "discharge_disposition_id": 1,
        "admission_source_id": 7,
        "time_in_hospital": 3,
        "payer_code": "Unknown",
        "medical_specialty": "InternalMedicine",
        "num_lab_procedures": 40,
        "num_procedures": 1,
        "num_medications": 15,
        "number_outpatient": 0,
        "number_emergency": 0,
        "number_inpatient": 0,
        "number_diagnoses": 8,
        "max_glu_serum": "None",
        "A1Cresult": "None",
        "insulin": "No",
        "change": "No",
        "diabetesMed": "Yes",
        "readmitted": "NO",
    }
    defaults.update(payload)
    raw_df = pd.DataFrame([defaults])
    processed_df = preprocess_data(raw_df)
    feature_df = create_features(processed_df)
    return feature_df


def predict_readmission_probability(payload: dict[str, object]) -> dict[str, object]:
    artifacts = load_model_artifacts()
    feature_df = build_prediction_frame(payload)
    aligned = align_feature_columns(
        feature_df.drop(columns=["encounter_id", "patient_nbr", "target"], errors="ignore"),
        artifacts.feature_columns,
    )
    std = artifacts.train_std.copy()
    std[std == 0] = 1.0
    standardized = (aligned.to_numpy() - artifacts.train_mean) / std
    with torch.no_grad():
        probability = float(
            torch.sigmoid(artifacts.model(torch.tensor(standardized, dtype=torch.float32))).item()
        )

    return {
        "readmission_probability": probability,
        "predicted_label": int(probability >= artifacts.threshold),
        "decision_threshold": artifacts.threshold,
        "top_features": artifacts.feature_importance[:5],
    }
