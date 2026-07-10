from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
MODEL_DIR = BASE_DIR / "artifacts"
REPORT_DIR = BASE_DIR / "reports"
LOG_DIR = BASE_DIR / "logs"
API_DIR = BASE_DIR / "src" / "api"
RAW_ZIP_FILE = RAW_DIR / "diabetes_130_us_hospitals.zip"
RAW_FILE = RAW_DIR / "diabetic_data.csv"
FEATURE_FILE = PROCESSED_DIR / "features.csv"
DB_FILE = DATA_DIR / "careflow.db"
MODEL_FILE = MODEL_DIR / "readmission_model.pt"
SQL_SCHEMA_FILE = BASE_DIR / "sql" / "schema.sql"
QUALITY_REPORT_FILE = REPORT_DIR / "data_quality_report.json"
SQL_REPORT_FILE = REPORT_DIR / "sql_kpi_report.csv"
MODEL_METRICS_FILE = REPORT_DIR / "model_metrics.json"
FEATURE_IMPORTANCE_FILE = REPORT_DIR / "feature_importance.csv"
PIPELINE_LOG_FILE = LOG_DIR / "pipeline.log"
DATASET_URL = (
    "https://archive.ics.uci.edu/static/public/296/"
    "diabetes+130-us+hospitals+for+years+1999-2008.zip"
)

REQUIRED_COLUMNS = [
    "encounter_id",
    "patient_nbr",
    "race",
    "gender",
    "age",
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
    "readmitted",
]
