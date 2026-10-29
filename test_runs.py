from thesis_code.runs import iter_scheduled_runs

if __name__ == "__main__":
    for config_path, cfg, run_name, output_dir in iter_scheduled_runs():
        print(config_path, cfg, run_name, output_dir)