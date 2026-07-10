CREATE TABLE IF NOT EXISTS raw_hospital_records (
    encounter_id TEXT,
    patient_nbr TEXT,
    race TEXT,
    gender TEXT,
    age TEXT,
    weight TEXT,
    admission_type_id TEXT,
    discharge_disposition_id TEXT,
    admission_source_id TEXT,
    time_in_hospital INTEGER,
    payer_code TEXT,
    medical_specialty TEXT,
    num_lab_procedures INTEGER,
    num_procedures INTEGER,
    num_medications INTEGER,
    number_outpatient INTEGER,
    number_emergency INTEGER,
    number_inpatient INTEGER,
    diag_1 TEXT,
    diag_2 TEXT,
    diag_3 TEXT,
    number_diagnoses INTEGER,
    max_glu_serum TEXT,
    A1Cresult TEXT,
    insulin TEXT,
    change TEXT,
    diabetesMed TEXT,
    readmitted TEXT
);

CREATE TABLE IF NOT EXISTS model_features (
    encounter_id TEXT,
    patient_nbr TEXT,
    age_midpoint REAL,
    time_in_hospital REAL,
    num_lab_procedures REAL,
    num_procedures REAL,
    num_medications REAL,
    number_outpatient REAL,
    number_emergency REAL,
    number_inpatient REAL,
    number_diagnoses REAL,
    utilization_total REAL,
    target REAL
);

CREATE TABLE IF NOT EXISTS pipeline_runs (
    run_id TEXT,
    stage TEXT,
    status TEXT,
    records_processed INTEGER,
    issue_count INTEGER,
    created_at TEXT,
    notes TEXT
);
