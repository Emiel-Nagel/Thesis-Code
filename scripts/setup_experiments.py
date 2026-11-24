from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent        # change if this file lives elsewhere
SCHEDULED_DIR = PROJECT_DIR / "experiments" / "scheduled"

if __name__ == "__main__":
    from thesis_code import runs

    output_dir = runs.create_output_dir()
    print(f"OUTPUT_DIR {output_dir}")

    for experiment in sorted(SCHEDULED_DIR.glob("*.py")):
        task = runs.load_task(experiment)
        run_name = runs.get_run_name(experiment)
        run_dir = runs.create_run_dir(output_dir, run_name)
        print(f"RUN {experiment} {str(run_dir)}")

    task.download_dataset()