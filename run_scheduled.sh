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
#   2) submits every experiment as its own array job (seeds 1..N_COPIES) without waiting,
#      so all experiments sit in the queue at once and run as soon as GPUs are free
#   3) submits one push job that starts after ALL array jobs have ended
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
MAX_PARALLEL=5                               # max seeds running at once PER experiment (empty = no limit)
WORKER_TIME="06:00:00"                       # time limit per seed
WORKER_PARTITION="neurophys,battaglialab,neuroinf"
MAIL_USER="username@donders.ru.nl"           # gets an e-mail when the push job ends or fails
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

# Optional: build the dataset cache once so parallel seeds do not race to create it
if [[ -n "$WARMUP_SCRIPT" ]]; then
    echo "Warming up cache..."
    python "$WARMUP_SCRIPT" || { echo "Cache warm-up failed"; exit 1; }
fi

# ---- 2) Submit every experiment as an array job, without waiting ---------
echo "Started submitting scheduled experiments."
JOB_IDS=()
for line in "${RUN_LINES[@]}"; do
    read -r EXPERIMENT RUN_DIR <<< "$line"
    name=$(basename "$EXPERIMENT" .py)

    # \$ keeps the variable literal here, so it is expanded inside the job, not now
    WORKER_CMD="export OMP_NUM_THREADS=\$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=\$SLURM_CPUS_PER_TASK; "
    WORKER_CMD+="echo \"Seed \$SLURM_ARRAY_TASK_ID -> GPU \$CUDA_VISIBLE_DEVICES\"; "
    WORKER_CMD+="python -u \"$EXPERIMENT\" --output_dir \"$RUN_DIR\" --seed \$SLURM_ARRAY_TASK_ID"

    jid=$(sbatch --parsable \
        --job-name="${name:0:20}" \
        --partition="$WORKER_PARTITION" \
        --exclude=cn40 \
        --gres=gpu:1 \
        --cpus-per-task=4 \
        --mem-per-cpu=6000 \
        --time="$WORKER_TIME" \
        --array=1-"$N_COPIES"${MAX_PARALLEL:+%$MAX_PARALLEL} \
        --output="slurm-%A_%a.out" \
        --error="slurm-%A_%a.err" \
        --wrap="$WORKER_CMD")
    if [[ $? -ne 0 || -z "$jid" ]]; then
        echo "WARNING: could not submit $name, skipping it"
        continue
    fi
    echo "Submitted $name -> $RUN_DIR (job $jid)"
    JOB_IDS+=("$jid")
done

if [[ ${#JOB_IDS[@]} -eq 0 ]]; then
    echo "Nothing was submitted."
    exit 1
fi

# ---- 3) Push job: starts once every array job has ended (success or not) --
DEPS=$(IFS=:; echo "${JOB_IDS[*]}")
sbatch --dependency=afterany:"$DEPS" \
    --job-name=push-outputs \
    --partition="$WORKER_PARTITION" \
    --cpus-per-task=1 \
    --mem-per-cpu=2000 \
    --time=02:00:00 \
    --output="slurm-push-%j.out" \
    --error="slurm-push-%j.err" \
    --mail-type=END,FAIL \
    --mail-user="$MAIL_USER" \
    --wrap="python \"$PUSH_SCRIPT\" --output_dir \"$OUTPUT_DIR\""

echo "All jobs submitted. The push runs after jobs: $DEPS"