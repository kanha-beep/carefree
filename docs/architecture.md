# Architecture Overview

## Goal

Predict 30-day hospital readmission risk while demonstrating a compact AI/ML data engineering workflow.

## Components

### 1. Ingestion

- Downloads the public UCI Diabetes 130-US Hospitals dataset
- Loads raw data from `data/raw/diabetic_data.csv`

### 2. Validation

- checks required columns
- validates expected target labels
- reports duplicates and missing values

### 3. Preprocessing

- standardizes text fields
- converts categorical buckets and target labels
- handles placeholder values and missing values

### 4. Feature Engineering

- derives age midpoint
- derives utilization totals
- one-hot encodes selected categorical fields

### 5. Data Storage

- stores raw and processed data in SQLite
- supports simple SQL-based inspection and downstream analysis

### 6. Model Training

- trains a small PyTorch neural network
- evaluates binary classification performance for readmission risk

## Future Improvements

- add Airflow or Prefect orchestration
- add API inference service
- connect to cloud storage and warehouse
- track experiments with MLflow
- add advanced model monitoring and drift checks
