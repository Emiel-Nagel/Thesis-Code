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
import os
import threading

# ---------------- Config (mirrors the bash script) ----------------
PROJECT_DIR = Path(__file__).resolve().parent        # change if this file lives elsewhere
SCHEDULED_DIR = PROJECT_DIR / "experiments" / "scheduled"
SETUP_SCRIPT = PROJECT_DIR / "scripts" / "setup_experiments.py"       # python setup_output.py <experiment.py>
PUSH_SCRIPT = PROJECT_DIR / "scripts" / "push_output.py"         # python push_output.py <output_dir>

N_COPIES = 1                                         # task ids 1..N_COPIES
MAX_PARALLEL = 1                                     # same as the %2 in --array
PUSH_EVEN_IF_COPIES_FAILED = True
LOG_DIR = PROJECT_DIR / "local_logs"                 # stdout/stderr of each copy
# ------------------------------------------------------------------

PYTHON = sys.executable  # use the same interpreter/environment as this script

# def _run_script(script: Path, *args) -> subprocess.CompletedProcess:
#     proc = subprocess.run(
#         [PYTHON, str(script), *map(str, args)],
#         capture_output=True, text=True, cwd=PROJECT_DIR)
#     if proc.stderr:
#         print(proc.stderr, file=sys.stderr, end="")
#     if proc.stdout:
#         print(proc.stdout, end="")
#     if proc.returncode != 0:
#         print(proc.stderr, file=sys.stderr)
#         return None
#     return proc

_print_lock = threading.Lock()

def _run_script(script: Path, *args) -> subprocess.CompletedProcess | None:
    argv = [PYTHON, "-u", str(script), *map(str, args)]
    env = {**os.environ, "PYTHONUNBUFFERED": "1"}
    proc = subprocess.Popen(argv, cwd=PROJECT_DIR, env=env,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            text=True, bufsize=1)
    out, err = [], []

    def pump(pipe, sink, dest):
        for line in pipe:
            sink.append(line)
            with _print_lock:
                print(line, end="", file=dest, flush=True)

    threads = [threading.Thread(target=pump, args=(proc.stdout, out, sys.stdout)),
               threading.Thread(target=pump, args=(proc.stderr, err, sys.stderr))]
    for t in threads: t.start()
    proc.wait()
    for t in threads: t.join()

    if proc.returncode != 0:
        return None
    return subprocess.CompletedProcess(argv, 0, "".join(out), "".join(err))

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
    jobs = [(experiment, run_dir, seed)
        for experiment, run_dir in run_list
        for seed in range(1, N_COPIES + 1)]
    print(f"Running {len(run_list)} experiments x {N_COPIES} seeds = {len(jobs)} jobs "
          f"(max {MAX_PARALLEL} at once)")

    with ThreadPoolExecutor(max_workers=MAX_PARALLEL) as pool:
        futures = {
            pool.submit(_run_script, experiment, "--output_dir", run_dir, "--seed", seed):
            (experiment, seed)
            for experiment, run_dir, seed in jobs
        }

    # the `with` block waits for everything, so all results are ready here
    failed = {}
    for f, (experiment, seed) in futures.items():
        if f.result() is None:
            failed.setdefault(experiment.name, []).append(seed)
    for name, seeds in failed.items():
        print(f"WARNING: seeds {seeds} of {name} failed")

    # step 3: push the outputs to a new branch
    _run_script(PUSH_SCRIPT, "--output_dir", output_dir)
    return 0

if __name__ == "__main__":
    sys.exit(main())