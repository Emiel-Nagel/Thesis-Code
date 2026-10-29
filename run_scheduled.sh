#!/bin/bash
#SBATCH --job-name=Thesis-Emiel-driver
#SBATCH --partition=neurophys
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=2000
#SBATCH --time=0
#SBATCH --output=slurm-driver-%j.out
#SBATCH --error=slurm-driver-%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=username@donders.ru.nl
#
# One file, two modes:
#   - DRIVER mode (default): loops over scheduled/*.py. For every file it
#       1) runs the setup script  -> gets cfg + output_dir
#       2) submits 5 copies (array 1-5) and waits until all have finished
#       3) runs the push script, and deletes output_dir if the push succeeded
#   - WORKER mode (MODE=worker, set by the driver): runs ONE copy of the
#       python file with the arguments: <cfg> <output_dir> <task_id>
#
# Submit with:   sbatch run_scheduled.sh
# (or, if your cluster does not allow sbatch from inside a job, run it from a
#  login node:   nohup bash run_scheduled.sh > driver.log 2>&1 &  )
 
# ---------------- Config ----------------
PROJECT_DIR="/vol/battaglialab/imaging1/vinck_students/emiel/thesis-code"
ENV_PATH="/vol/battaglialab/imaging1/vinck_students/emiel/miniconda3/envs/.venv"
 
SELF="$PROJECT_DIR/run_scheduled.sh"         # full path to THIS file (slurm copies the script, so $0 is unreliable)
SCHEDULED_DIR="$PROJECT_DIR/scheduled"       # folder with the python files to run
SETUP_SCRIPT="$PROJECT_DIR/setup_output.py"  # called as: python setup_output.py <experiment.py>
PUSH_SCRIPT="$PROJECT_DIR/push_output.py"    # called as: python push_output.py <output_dir>
 
N_COPIES=5                                   # copies get task ids 1..N_COPIES
MAX_PARALLEL=2                               # max copies (= GPUs) running at the same time
WORKER_TIME="06:00:00"                       # time limit per copy
WORKER_PARTITION="neurophys,battaglialab,neuroinf"
PUSH_EVEN_IF_COPIES_FAILED=true              # false = keep local output_dir when any copy failed
# ----------------------------------------
 
source ~/.bashrc
conda activate "$ENV_PATH"
export MKL_THREADING_LAYER=GNU
cd "$PROJECT_DIR" || { echo "Cannot cd to $PROJECT_DIR"; exit 1; }
 
# =====================================================================
# WORKER MODE: one copy of one experiment
# =====================================================================
if [[ "$MODE" == "worker" ]]; then
    echo "Task $SLURM_ARRAY_TASK_ID -> GPU $CUDA_VISIBLE_DEVICES"
    echo "Running: python $EXPERIMENT $CFG $OUTPUT_DIR $SLURM_ARRAY_TASK_ID"
    python "$EXPERIMENT" "$CFG" "$OUTPUT_DIR" "$SLURM_ARRAY_TASK_ID"
    exit $?
fi
 
# =====================================================================
# DRIVER MODE
# =====================================================================
echo "Started running scheduled experiments."
shopt -s nullglob
files=("$SCHEDULED_DIR"/*.py)
if [[ ${#files[@]} -eq 0 ]]; then
    echo "No .py files found in $SCHEDULED_DIR"
    exit 1
fi
 
for EXPERIMENT in "${files[@]}"; do
    name=$(basename "$EXPERIMENT" .py)
    echo "=================================================="
    echo "Experiment: $name"
    echo "=================================================="
 
    # ---- 1) Setup: returns cfg and output_dir -----------------------
    # The setup script must print one line:  SETUP_RESULT <cfg> <output_dir>
    read -r _ CFG OUTPUT_DIR < <(python "$SETUP_SCRIPT" "$EXPERIMENT" | grep '^SETUP_RESULT ' | tail -n 1)
    if [[ -z "$CFG" || -z "$OUTPUT_DIR" ]]; then
        echo "Setup failed for $name (no SETUP_RESULT line). Skipping."
        continue
    fi
    echo "cfg=$CFG"
    echo "output_dir=$OUTPUT_DIR"
 
    # ---- 2) Run the copies as an array job and wait for all of them --
    sbatch --wait \
        --job-name="${name:0:20}" \
        --partition="$WORKER_PARTITION" \
        --exclude=cn40 \
        --gres=gpu:1 \
        --cpus-per-task=4 \
        --mem-per-cpu=6000 \
        --time="$WORKER_TIME" \
        --array=1-"$N_COPIES"%"$MAX_PARALLEL" \
        --output="slurm-%A_%a.out" \
        --error="slurm-%A_%a.err" \
        --export=ALL,MODE=worker,EXPERIMENT="$EXPERIMENT",CFG="$CFG",OUTPUT_DIR="$OUTPUT_DIR" \
        "$SELF"
    array_status=$?
    if [[ $array_status -ne 0 ]]; then
        echo "WARNING: at least one copy of $name failed (exit $array_status)."
        if [[ "$PUSH_EVEN_IF_COPIES_FAILED" != true ]]; then
            echo "Keeping $OUTPUT_DIR locally, skipping push."
            continue
        fi
    fi
 
    # ---- 3) Push to GitHub, delete local output_dir only if it worked -
    if python "$PUSH_SCRIPT" "$OUTPUT_DIR"; then
        echo "Push succeeded, deleting $OUTPUT_DIR"
        [[ -n "$OUTPUT_DIR" && -d "$OUTPUT_DIR" ]] && rm -rf -- "$OUTPUT_DIR"
    else
        echo "Push FAILED, keeping $OUTPUT_DIR"
    fi
done
 
echo "All tasks finished."