import mlflow, subprocess, sys

mlflow.set_tracking_uri("sqlite:///MLflow_server/mlflow.db")

# Print connection information
print(f"MLflow Tracking URI: {mlflow.get_tracking_uri()}")
print(f"Active Experiment: {mlflow.get_experiment_by_name('SRNN-training Experiment 1')}")

server = subprocess.Popen(
    [
        sys.executable, "-m", "mlflow", "server",
        "--backend-store-uri", "sqlite:///MLflow_server/mlflow.db",
        "--port", "5000",
    ],
    stdout=subprocess.DEVNULL,     # or open("server.log", "w") to keep the output
    stderr=subprocess.STDOUT,
)

# update to be able to close the server as well