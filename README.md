# CareFlow AI

CareFlow AI is a portfolio-ready AI/ML data engineering project built around a real institutional problem: **hospital readmission risk**.

Hospital readmissions create operational strain, increase care costs, and often indicate continuity-of-care gaps. This project predicts whether a diabetic patient is likely to be **readmitted within 30 days** and builds the supporting data pipeline needed to prepare that AI workflow.

## Why this project fits the job description

This project demonstrates:

- Python programming for ETL, preprocessing, feature engineering, and automation
- SQL and data querying with SQLite
- Data pipelines for structured data
- Data cleaning and data quality validation
- PyTorch model integration for binary classification
- model explainability through permutation feature importance
- FastAPI service endpoints for production-style inference
- pipeline monitoring, run tracking, and troubleshooting artifacts
- SQL analytics for institution-facing KPI reporting
- Streamlit dashboarding for stakeholder-facing analytics
- Documentation and project organization suitable for an AI/ML engineering portfolio

## Problem Statement

Hospitals often struggle with avoidable readmissions. A prediction system can help institutions:

- identify high-risk discharges
- prioritize follow-up and care coordination
- improve operational planning
- reduce avoidable readmission-related costs

## Project Structure

```text
CareFlow AI/
|-- data/
|   |-- processed/
|   `-- raw/
|-- docs/
|   `-- architecture.md
|-- sql/
|   `-- schema.sql
|-- src/
|   |-- models/
|   |   `-- pytorch_model.py
|   |-- pipeline/
|   |   |-- data_ingestion.py
|   |   |-- data_preprocessing.py
|   |   |-- data_validation.py
|   |   |-- database.py
|   |   `-- feature_engineering.py
|   |-- config.py
|   |-- api/
|   |-- run_pipeline.py
|   |-- run_workflow.py
|   |-- reporting/
|   `-- train.py
|-- streamlit_app.py
`-- requirements.txt
```

## Pipeline Flow

1. Download and ingest a real public hospital dataset
2. Validate schema and data quality
3. Clean and preprocess records
4. Engineer model-ready features
5. Save raw and processed data into SQLite
6. Generate data quality and SQL KPI reports
7. Train a PyTorch model to predict 30-day readmission
8. Serve predictions through FastAPI endpoints
9. Surface outcomes in a Streamlit dashboard

## Four Requirement-Aligned Upgrades

1. Pipeline monitoring and troubleshooting
   Added structured pipeline logging and a `pipeline_runs` tracking table for ingestion, feature engineering, and model training stages.
2. Stronger data quality validation
   Added a reusable JSON data quality report with issue counts, duplicates, missing values, and target distribution summaries.
3. SQL-first analytics workflow
   Added KPI SQL queries and a report generator that writes stakeholder-friendly metrics from SQLite outputs.
4. End-to-end workflow automation
   Added a single Python entrypoint that runs pipeline execution, model training, and SQL report generation in sequence.

## Verified Run Summary

The project was re-run end to end on **July 8, 2026** using the existing codebase with the UCI dataset.

- Primary dataset download URL worked during verification.
- A fallback using the `ucimlrepo` package is now implemented in code if the UCI ZIP URL is blocked or unavailable.
- Verified commands:
  - `python -m src.run_pipeline`
  - `python -m src.train`
  - `python -m src.run_workflow`
  - `python -m uvicorn src.api.app:app --host 127.0.0.1 --port 8001`
  - `streamlit run streamlit_app.py --server.headless true --server.port 8501`

## Verified Model Metrics

Metrics below are from the latest successful verified run written by the code to `reports/model_metrics.json`:

- Accuracy: `0.6968`
- Precision: `0.1683`
- Recall: `0.4417`
- F1 Score: `0.2438`
- ROC-AUC: `0.6198`
- Decision Threshold: `0.55`

## Model Explainability

The training workflow now computes permutation feature importance and stores it in `reports/feature_importance.csv`. Top predictive features from the latest verified run:

- `number_inpatient`
- `utilization_total`
- `medical_specialty_Unknown`
- `payer_code_MC`
- `medical_specialty_InternalMedicine`
- `medical_specialty_Cardiology`
- `num_lab_procedures`
- `admission_source_id`
- `discharge_disposition_id`
- `payer_code_Unknown`

These features are also surfaced in the Streamlit dashboard and FastAPI `/metrics` endpoint, which makes the model easier to explain in interviews.

## SQL Analytics

The SQLite reporting layer now runs 5 verified analytics queries and writes real outputs to `reports/sql_kpi_report.csv`:

- overall readmission distribution
- readmission rate by age bucket
- readmission rate by admission type
- readmission rate by length of stay
- top primary diagnoses associated with readmission

Sample verified outputs from the latest run:

- Overall distribution: `NO=54,864`, `>30=35,545`, `<30=11,357`
- Highest readmission age buckets included `[20-30)` at `14.24%`, `[80-90)` at `12.08%`, and `[70-80)` at `11.77%`
- Length of stay showed increasing risk, including `8 days -> 14.23%` and `10 days -> 14.35%`
- Top diagnosis codes among `<30` readmissions included `428`, `414`, `410`, `434`, and `486`

## FastAPI Service

The project now includes a minimal FastAPI app in `src/api/app.py` with these verified endpoints:

- `GET /health`
- `GET /metrics`
- `POST /predict`

Verified API behavior from the latest run:

- `GET /health` returned `{"status":"ok"}`
- `POST /predict` returned a readmission probability of `0.8082` for a sample high-risk patient payload

## Features Used

- race, gender, and age bucket
- admission type, discharge disposition, and admission source
- time in hospital
- lab procedures, medications, and diagnoses
- outpatient, emergency, and inpatient visit history
- insulin, A1C, glucose result, medication change, and diabetes medication flags

## Tech Stack

- Python
- Pandas
- NumPy
- SQLite
- PyTorch

## How to Run

1. Create a virtual environment and install dependencies:

```bash
pip install -r requirements.txt
```

2. Run the full pipeline:

```bash
py -m src.run_pipeline
```

3. Train the model:

```bash
py -m src.train
```

4. Launch the dashboard:

```bash
streamlit run streamlit_app.py
```

5. Run the full automated workflow:

```bash
py -m src.run_workflow
```

6. Run the FastAPI service:

```bash
uvicorn src.api.app:app --host 127.0.0.1 --port 8001
```

## Output

Running the project creates:

- downloaded UCI raw data in `data/raw/diabetic_data.csv`
- processed dataset in `data/processed/features.csv`
- SQLite database in `data/careflow.db`
- pipeline log in `logs/pipeline.log`
- data quality report in `reports/data_quality_report.json`
- SQL KPI report in `reports/sql_kpi_report.csv`
- model metrics in `reports/model_metrics.json`
- feature importance report in `reports/feature_importance.csv`
- model artifact in `artifacts/readmission_model.pt`
- FastAPI inference service in `src/api/app.py`
- an interactive analytics dashboard

## Public Dataset

This project uses the public **Diabetes 130-US Hospitals for Years 1999-2008** dataset from the UCI Machine Learning Repository.

- Dataset page: https://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999-2008
- Citation: Clore, J., Cios, K., DeShazo, J., & Strack, B. (2014). *Diabetes 130-US Hospitals for Years 1999-2008*. UCI Machine Learning Repository. https://doi.org/10.24432/C5230J

## Resume-Friendly Description

Built an end-to-end AI/ML data engineering project to predict 30-day hospital readmission risk using Python, Pandas, SQLite, PyTorch, FastAPI, and Streamlit. Designed ETL workflows, data validation checks, preprocessing and feature engineering pipelines, SQL KPI reporting, model explainability, and a production-style inference API.

## Resume Bullet Points

- Built an end-to-end AI/ML data pipeline on the UCI Diabetes 130-US Hospitals dataset to predict 30-day readmission risk using Python, Pandas, SQLite, PyTorch, FastAPI, and Streamlit.
- Developed ETL workflows for ingestion, validation, preprocessing, missing-value handling, and feature engineering across 100K+ real hospital encounter records.
- Engineered model-ready features from demographic, utilization, medication, and lab-related variables and trained a binary classification model with verified metrics of `0.6968` accuracy and `0.6198` ROC-AUC on the latest run.
- Added permutation-based feature importance to explain the top drivers of readmission risk and improve interview readiness for model interpretability discussions.
- Created a FastAPI inference service with `/health`, `/metrics`, and `/predict` endpoints to expose the trained PyTorch model as a backend application component.
- Created a Streamlit dashboard to present readmission trends, SQL KPIs, data quality summaries, feature importance, and pipeline-run evidence for non-technical stakeholders.
- Implemented pipeline monitoring, run-level logging, a `pipeline_runs` tracking table, SQL KPI reporting, and automated workflow orchestration to mirror production-style data engineering operations.
- Structured the project with reusable modules, SQL-backed storage, documentation, and automation-oriented scripts to mirror production-style AI/ML engineering workflows.

## GitHub Description

CareFlow AI is an end-to-end AI/ML data engineering project that predicts 30-day hospital readmission risk using the UCI Diabetes 130-US Hospitals dataset. It combines ETL pipelines, data validation, feature engineering, SQLite storage, PyTorch model training, and a Streamlit dashboard in a portfolio-ready structure.

## Notes

- The first pipeline run downloads the public dataset from UCI.
- If the ZIP download is blocked or unavailable, the code now falls back to the `ucimlrepo` package automatically.
- If both network-based options are unavailable, place `diabetic_data.csv` inside `data/raw/` and rerun the pipeline.

## Pipeline Run Proof

Sample real rows from the verified `pipeline_runs` table:

```text
run_id,stage,status,records_processed,issue_count,created_at
2f51b305-c05b-4c05-8e7c-945d360152f8,model_training,success,101763,0,2026-07-08T18:05:34.250256+00:00
f4699033-88b3-4208-8a73-957b8bc79c5d,model_training,success,101763,0,2026-07-08T18:05:14.009404+00:00
bdb0c0cc-c871-477d-9d2d-084bd1358aa9,feature_engineering,success,101763,1,2026-07-08T18:02:51.660333+00:00
bdb0c0cc-c871-477d-9d2d-084bd1358aa9,ingestion_validation,warning,101766,1,2026-07-08T18:02:38.435292+00:00
```
"# carefree" 
