#!/bin/bash
#SBATCH --job-name=Thesis-Emiel-driver
#SBATCH --partition=neurophys
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=2000
#SBATCH --time=02:00:00
#SBATCH --output=slurm-driver-%j.out
#SBATCH --error=slurm-driver-%j.err
#
# Slurm twin of run_scheduled.py. This script only SUBMITS jobs and then exits:
#
#   1) runs scripts/setup_experiments.py, which prints
#          OUTPUT_DIR <output_dir>
#          RUN <experiment.py> <run_dir>          (one line per scheduled experiment)
#   2) submits ONE array job covering every (experiment, seed) pair, with at most
#      MAX_PARALLEL tasks running at once in total (= max GPUs in use)
#   3) submits one push job that starts after the array job has ended
#
# Submit with:   sbatch run_scheduled.sh
# or run it directly on a login node:   bash run_scheduled.sh

# ---------------- Config ----------------
PROJECT_DIR="/vol/battaglialab/imaging1/vinck_students/emiel/thesis-code"
ENV_PATH="/vol/battaglialab/imaging1/vinck_students/emiel/miniconda3/envs/.venv"

SETUP_SCRIPT="$PROJECT_DIR/scripts/setup_experiments.py"   # python setup_experiments.py
PUSH_SCRIPT="$PROJECT_DIR/scripts/push_output.py"          # python push_output.py --output_dir <output_dir>
WARMUP_SCRIPT=""                                           # optional, e.g. "$PROJECT_DIR/scripts/warm_cache.py"; empty = skip

N_COPIES=5                                   # seeds 1 .. N_COPIES
MAX_PARALLEL=2                               # max tasks (= GPUs) running at once IN TOTAL (empty = no limit)
WORKER_TIME="06:00:00"                       # time limit per seed
WORKER_PARTITION="neurophys,battaglialab,neuroinf"
# ----------------------------------------

source ~/.bashrc
conda activate "$ENV_PATH"                   # jobs inherit this environment (sbatch exports it by default)
export MKL_THREADING_LAYER=GNU
export PYTHONPATH="$PROJECT_DIR:$PYTHONPATH"
cd "$PROJECT_DIR" || { echo "Cannot cd to $PROJECT_DIR"; exit 1; }

# ---- 1) Setup (runs now, before anything is submitted) -------------------
SETUP_OUT=$(python "$SETUP_SCRIPT")
setup_status=$?
echo "$SETUP_OUT"
if [[ $setup_status -ne 0 ]]; then
    echo "Setup failed (exit $setup_status)."
    exit 1
fi

OUTPUT_DIR=$(sed -n 's/^OUTPUT_DIR //p' <<< "$SETUP_OUT" | tail -n 1)
mapfile -t RUN_LINES < <(sed -n 's/^RUN //p' <<< "$SETUP_OUT")
if [[ -z "$OUTPUT_DIR" || ${#RUN_LINES[@]} -eq 0 ]]; then
    echo "Setup printed no OUTPUT_DIR/RUN lines"
    exit 1
fi

# ---- 2) One array job over every (experiment, seed) pair ------------------
# The list of "<experiment.py> <run_dir>" lines is handed to the tasks through the
# environment (sbatch exports the whole environment by default).
export RUNS_STR
RUNS_STR=$(printf '%s\n' "${RUN_LINES[@]}")

N_EXP=${#RUN_LINES[@]}
TOTAL=$((N_EXP * N_COPIES))

# \$ keeps the variable literal here, so it is expanded inside the task, not now
WORKER_CMD="idx=\$((SLURM_ARRAY_TASK_ID)); "
WORKER_CMD+="read -r EXPERIMENT RUN_DIR <<< \"\$(sed -n \"\$((idx / $N_COPIES + 1))p\" <<< \"\$RUNS_STR\")\"; "
WORKER_CMD+="SEED=\$((idx % $N_COPIES + 1)); "
WORKER_CMD+="export OMP_NUM_THREADS=\$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=\$SLURM_CPUS_PER_TASK; "
WORKER_CMD+="echo \"Task \$idx: \$EXPERIMENT seed \$SEED -> GPU \$CUDA_VISIBLE_DEVICES\"; "
WORKER_CMD+="python -u \"\$EXPERIMENT\" --output_dir \"\$RUN_DIR\" --seed \$SEED"

echo "Submitting $N_EXP experiments x $N_COPIES seeds = $TOTAL tasks (max ${MAX_PARALLEL:-unlimited} at once)"
JOB_ID=$(sbatch --parsable \
    --job-name=thesis-runs \
    --partition="$WORKER_PARTITION" \
    --exclude=cn40 \
    --gres=gpu:1 \
    --cpus-per-task=4 \
    --mem-per-cpu=6000 \
    --time="$WORKER_TIME" \
    --array=0-$((TOTAL - 1))${MAX_PARALLEL:+%$MAX_PARALLEL} \
    --output="slurm-%A_%a.out" \
    --error="slurm-%A_%a.err" \
    --wrap="$WORKER_CMD")
if [[ $? -ne 0 || -z "$JOB_ID" ]]; then
    echo "Could not submit the array job."
    exit 1
fi
echo "Submitted array job $JOB_ID"

# ---- 3) Push job: starts once every array job has ended (success or not) --
sbatch --dependency=afterany:"$JOB_ID" \
    --job-name=push-outputs \
    --partition="$WORKER_PARTITION" \
    --cpus-per-task=1 \
    --mem-per-cpu=2000 \
    --time=02:00:00 \
    --output="slurm-push-%j.out" \
    --error="slurm-push-%j.err" \
    --wrap="python \"$PUSH_SCRIPT\" --output_dir \"$OUTPUT_DIR\""

echo "All jobs submitted. The push runs after job $JOB_ID"