if __name__ == "__main__":
    import argparse
    import sys
    from pathlib import Path

    from thesis_code import runs

    p = argparse.ArgumentParser()
    p.add_argument("--output_dir", type=Path, required=True)

    args = p.parse_args()
    output_dir = Path(args.output_dir)

    for run_dir in sorted(p for p in output_dir.iterdir() if p.is_dir()):
        try:
            branch_name = f"{output_dir.name}-{run_dir.name}"
            runs.push_output(run_dir, branch_name)
        except Exception as e:
            print(e, file=sys.stderr)
        runs.remove_output_dir(run_dir)
    runs.remove_output_dir(output_dir)