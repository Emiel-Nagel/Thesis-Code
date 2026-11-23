"""
This script reads an experiment in experiments/scheduled/ and retrieves its id from mlflow.
For each run in the experiment folder, it creates a new parent run, of which each of the 5 copies later inherit their child runs from.
"""

from mlflow import MlflowClient
from pathlib import Path

from thesis_code import runs
from thesis_code import datasets

PROJECT_DIR = Path(__file__).resolve().parent.parent        # change if this file lives elsewhere
SCHEDULED_DIR = PROJECT_DIR / "experiments" / "scheduled"

if __name__ == "__main__":
    client = MlflowClient()
    output_dir = runs.create_output_dir()
    print(f"OUTPUT_DIR {output_dir}")

    # set up the main output folder
    for experiment_folder in SCHEDULED_DIR.iterdir():
        if not experiment_folder.is_dir():
            continue

        experiment = client.get_experiment_by_name(experiment_folder.name)
        if experiment is None:
            print(f"No experiment with name '{experiment_folder.name}' exists!")
            continue
        experiment_dir = runs.create_experiment_dir(output_dir, experiment_name=experiment.name)

        # set up the individual runs
        for run_file in sorted(experiment_folder.glob("*.py")):
            parent_run = client.create_run(experiment.experiment_id, run_name=run_file.stem)
            print(f"EXPERIMENT_ID {experiment.experiment_id}")
            print(f"RUN_ID {parent_run.info.run_id}")

            cfg = runs.load_config(run_file)
            run_dir = runs.create_run_dir(output_dir, cfg)

            print(f"RUN {run_file} {str(run_dir)}")

            datasets.download_SHD_dataset(
                n_timepoints=cfg.training.dataset.n_timepoints,
                re_download=False,
            )