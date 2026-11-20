import mlflow

mlflow.set_tracking_uri("sqlite:///MLflow_server/mlflow.db")

# Print connection information
print(f"MLflow Tracking URI: {mlflow.get_tracking_uri()}")
print(f"Active Experiment: {mlflow.get_experiment_by_name('SRNN-training Experiment 1')}")