#!/usr/bin/env python3
"""
Local twin of run_scheduled.sh, for testing your setup / experiment / push
scripts without slurm.

For every scheduled/*.py file it:
  1) runs the setup script  -> gets cfg + output_dir
  2) runs N_COPIES copies (task ids 1..N_COPIES), at most MAX_PARALLEL at once,
     as:  python <file> <cfg> <output_dir> <task_id>
  3) runs the push script, and deletes output_dir only if the push succeeded

Usage:  python run_scheduled.py
"""
import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

# ---------------- Config (mirrors the bash script) ----------------
PROJECT_DIR = Path(__file__).resolve().parent        # change if this file lives elsewhere
SCHEDULED_DIR = PROJECT_DIR / "experiments" / "scheduled"
SETUP_SCRIPT = PROJECT_DIR / "scripts" / "setup_experiments.py"       # python setup_output.py <experiment.py>
PUSH_SCRIPT = PROJECT_DIR / "scripts" / "push_output.py"         # python push_output.py <output_dir>

N_COPIES = 5                                         # task ids 1..N_COPIES
MAX_PARALLEL = 5                                     # same as the %2 in --array
PUSH_EVEN_IF_COPIES_FAILED = True
LOG_DIR = PROJECT_DIR / "local_logs"                 # stdout/stderr of each copy
# ------------------------------------------------------------------

PYTHON = sys.executable  # use the same interpreter/environment as this script

def _run_script(script: Path, *args) -> subprocess.CompletedProcess:
    proc = subprocess.run(
        [PYTHON, str(script), *map(str, args)],
        capture_output=True, text=True, cwd=PROJECT_DIR)
    if proc.stdout:
        print(proc.stdout, end="")
    if proc.returncode != 0:
        print(proc.stderr, file=sys.stderr)
        return None
    return proc

def main() -> int:
    # step 1: run setup_output.py
    setup = _run_script(SETUP_SCRIPT)
    if setup is None:
        return 1

    m = re.search(r"^OUTPUT_DIR (\S+)\s*$", setup.stdout, flags=re.M)
    run_list = [(Path(c), Path(r)) for c, r in
                re.findall(r"^RUN (\S+) (\S+)\s*$", setup.stdout, flags=re.M)]
    if not m or not run_list:
        print("Setup printed no OUTPUT_DIR/RUN lines")
        return 1
    output_dir = m.group(1)
    print(run_list)

    # step 2: run 5 copies of each of the runs
    print("Started running scheduled experiments.")
    for experiment, run_dir in run_list:
        print(f"Running {experiment.name} -> {run_dir}")
        with ThreadPoolExecutor(max_workers=MAX_PARALLEL) as pool:
            futures = [pool.submit(_run_script, experiment,
                "--output_dir", run_dir, "--seed", i)
                for i in range(1, N_COPIES + 1)]
        failed = [i for i, f in enumerate(futures, 1) if f.result() is None]
        if failed:
            print(f"WARNING: seeds {failed} of {experiment.name} failed")

    # step 3: push the outputs to a new branch
    _run_script(PUSH_SCRIPT, "--output_dir", output_dir)
    return 0

if __name__ == "__main__":
    sys.exit(main())