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
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

# ---------------- Config (mirrors the bash script) ----------------
PROJECT_DIR = Path(__file__).resolve().parent        # change if this file lives elsewhere
SCHEDULED_DIR = PROJECT_DIR / "scheduled"
SETUP_SCRIPT = PROJECT_DIR / "setup_output.py"       # python setup_output.py <experiment.py>
PUSH_SCRIPT = PROJECT_DIR / "push_output.py"         # python push_output.py <output_dir>

N_COPIES = 5                                         # task ids 1..N_COPIES
MAX_PARALLEL = 2                                     # same as the %2 in --array
PUSH_EVEN_IF_COPIES_FAILED = True
LOG_DIR = PROJECT_DIR / "local_logs"                 # stdout/stderr of each copy
# ------------------------------------------------------------------

PYTHON = sys.executable  # use the same interpreter/environment as this script

def run_setup(experiment: Path):
    """Run the setup script; return (cfg, output_dir) or None on failure.

    The setup script must print a line:  SETUP_RESULT <cfg> <output_dir>
    """
    proc = subprocess.run(
        [PYTHON, str(SETUP_SCRIPT), str(experiment)],
        capture_output=True, text=True, cwd=PROJECT_DIR,
    )
    if proc.stdout:
        print(proc.stdout, end="")
    if proc.returncode != 0:
        print(proc.stderr, file=sys.stderr)
        return None
    matches = re.findall(r"^SETUP_RESULT (\S+) (\S+)\s*$", proc.stdout, flags=re.M)
    if not matches:
        return None
    return matches[-1]  # last SETUP_RESULT line, like `tail -n 1`

def run_copy(experiment: Path, cfg: str, output_dir: str, task_id: int) -> int:
    """Run one copy of the experiment; stdout/stderr go to local_logs/."""
    LOG_DIR.mkdir(exist_ok=True)
    out = LOG_DIR / f"{experiment.stem}_{task_id}.out"
    err = LOG_DIR / f"{experiment.stem}_{task_id}.err"
    print(f"Running: python {experiment.name} {cfg} {output_dir} {task_id}")
    with open(out, "w") as fo, open(err, "w") as fe:
        proc = subprocess.run(
            [PYTHON, str(experiment), cfg, output_dir, str(task_id)],
            stdout=fo, stderr=fe, cwd=PROJECT_DIR,
        )
    print(f"  task {task_id} finished with exit code {proc.returncode}")
    return proc.returncode


def main() -> int:
    files = sorted(SCHEDULED_DIR.glob("*.py"))
    if not files:
        print(f"No .py files found in {SCHEDULED_DIR}")
        return 1

    print("Started running scheduled experiments.")
    for experiment in files:
        print("=" * 50)
        print(f"Experiment: {experiment.stem}")
        print("=" * 50)

        # ---- 1) Setup ----
        result = run_setup(experiment)
        if result is None:
            print(f"Setup failed for {experiment.stem} (no SETUP_RESULT line). Skipping.")
            continue
        cfg, output_dir = result
        print(f"cfg={cfg}")
        print(f"output_dir={output_dir}")

        # ---- 2) Copies, max MAX_PARALLEL at a time ----
        with ThreadPoolExecutor(max_workers=MAX_PARALLEL) as pool:
            futures = [
                pool.submit(run_copy, experiment, cfg, output_dir, i)
                for i in range(1, N_COPIES + 1)
            ]
            codes = [f.result() for f in futures]
        if any(codes):
            print(f"WARNING: at least one copy of {experiment.stem} failed: {codes}")
            if not PUSH_EVEN_IF_COPIES_FAILED:
                print(f"Keeping {output_dir} locally, skipping push.")
                continue

        # ---- 3) Push, delete local output_dir only if it worked ----
        push = subprocess.run([PYTHON, str(PUSH_SCRIPT), output_dir], cwd=PROJECT_DIR)
        if push.returncode == 0:
            print(f"Push succeeded, deleting {output_dir}")
            if output_dir and Path(output_dir).is_dir():
                shutil.rmtree(output_dir)
        else:
            print(f"Push FAILED, keeping {output_dir}")

    print("All tasks finished.")
    return 0


if __name__ == "__main__":
    sys.exit(main())