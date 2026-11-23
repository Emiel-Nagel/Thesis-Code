"""
This script pushes the outputs to mlflow, after having performed the analyses.
"""
import mlflow
from pathlib import Path

from thesis_code import runs

if __name__ == "__main__":
    mlflow.set_tracking_uri("sqlite:///MLflow_server/mlflow.db")

    run_id, run_dir = runs.get_args({"run_id": str, "run_dir": Path})
    mlflow.log_artifacts(str(run_dir), run_id=run_id)

    mlflow.MlflowClient().set_terminated(run_id, status="FINISHED")