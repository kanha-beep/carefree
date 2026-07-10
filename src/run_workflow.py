from __future__ import annotations

from src.reporting.generate_sql_report import generate_sql_kpi_report
from src.run_pipeline import main as run_pipeline
from src.train import main as run_training


def main() -> None:
    print("Starting CareFlow AI workflow...")
    run_pipeline()
    run_training()
    kpi_report = generate_sql_kpi_report()
    print("\nSQL KPI report generated:")
    print(kpi_report.to_string(index=False))


if __name__ == "__main__":
    main()
