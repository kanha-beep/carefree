from __future__ import annotations

from fastapi import FastAPI
from pydantic import BaseModel

from src.models.inference import load_model_artifacts, predict_readmission_probability


app = FastAPI(title="CareFlow AI API", version="1.0.0")


class PredictionRequest(BaseModel):
    encounter_id: str | None = None
    patient_nbr: str | None = None
    race: str = "Caucasian"
    gender: str = "Female"
    age: str = "[50-60)"
    admission_type_id: int = 1
    discharge_disposition_id: int = 1
    admission_source_id: int = 7
    time_in_hospital: int = 3
    payer_code: str = "Unknown"
    medical_specialty: str = "InternalMedicine"
    num_lab_procedures: int = 40
    num_procedures: int = 1
    num_medications: int = 15
    number_outpatient: int = 0
    number_emergency: int = 0
    number_inpatient: int = 0
    number_diagnoses: int = 8
    max_glu_serum: str = "None"
    A1Cresult: str = "None"
    insulin: str = "No"
    change: str = "No"
    diabetesMed: str = "Yes"


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/metrics")
def metrics() -> dict[str, object]:
    artifacts = load_model_artifacts()
    return {
        "metrics": artifacts.metrics,
        "top_features": artifacts.feature_importance[:10],
    }


@app.post("/predict")
def predict(request: PredictionRequest) -> dict[str, object]:
    return predict_readmission_probability(request.model_dump(exclude_none=True))
