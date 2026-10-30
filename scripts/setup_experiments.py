from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent        # change if this file lives elsewhere
SCHEDULED_DIR = PROJECT_DIR / "experiments" / "scheduled"

if __name__ == "__main__":
    from thesis_code import runs
    from thesis_code import datasets

    output_dir = runs.create_output_dir()
    print(f"OUTPUT_DIR {output_dir}")

    for experiment in sorted(SCHEDULED_DIR.glob("*.py")):
        cfg = runs.load_config(experiment)
        run_dir = runs.create_run_dir(output_dir, cfg)
        print(f"RUN {experiment} {str(run_dir)}")

    datasets.download_SHD_dataset(
        time_window=cfg.training.dt*1e6,
        re_download=cfg.training.dataset.re_download,
    )