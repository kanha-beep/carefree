from __future__ import annotations

from uuid import uuid4

from src.config import FEATURE_FILE, PROCESSED_DIR
from src.pipeline.data_ingestion import load_raw_data
from src.pipeline.data_preprocessing import preprocess_data
from src.pipeline.data_validation import validate_raw_data
from src.pipeline.database import initialize_database, record_pipeline_run, save_dataframe
from src.pipeline.feature_engineering import create_features
from src.pipeline.logging_utils import get_pipeline_logger
from src.pipeline.quality_reporting import build_quality_report, save_quality_report


def main() -> None:
    run_id = str(uuid4())
    logger = get_pipeline_logger()
    logger.info("Pipeline run started | run_id=%s", run_id)

    raw_df = load_raw_data()
    issues = validate_raw_data(raw_df)

    initialize_database()
    save_dataframe(raw_df, "raw_hospital_records")
    record_pipeline_run(
        run_id=run_id,
        stage="ingestion_validation",
        status="warning" if issues else "success",
        records_processed=len(raw_df),
        issue_count=len(issues),
        notes="Raw dataset loaded and validation completed.",
    )

    if issues:
        print("Validation issues detected:")
        for issue in issues:
            print(f"- {issue}")
            logger.warning("%s | run_id=%s", issue, run_id)
    else:
        print("Raw data validation passed.")
        logger.info("Raw data validation passed | run_id=%s", run_id)

    quality_report = build_quality_report(raw_df, issues)
    save_quality_report(quality_report)
    logger.info("Data quality report saved | run_id=%s", run_id)

    cleaned_df = preprocess_data(raw_df)
    feature_df = create_features(cleaned_df)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    feature_df.to_csv(FEATURE_FILE, index=False)
    save_dataframe(feature_df, "model_features")
    record_pipeline_run(
        run_id=run_id,
        stage="feature_engineering",
        status="success",
        records_processed=len(feature_df),
        issue_count=len(issues),
        notes="Processed features saved to CSV and SQLite.",
    )
    logger.info("Feature engineering completed | run_id=%s | rows=%s", run_id, len(feature_df))

    print(f"Raw rows: {len(raw_df)}")
    print(f"Processed rows: {len(feature_df)}")
    print(f"Feature file saved to: {FEATURE_FILE}")
    print("Data quality report saved.")
    logger.info("Pipeline run completed | run_id=%s", run_id)


if __name__ == "__main__":
    main()
