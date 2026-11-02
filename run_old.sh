#!/bin/bash
#SBATCH --job-name=Thesis-Emiel # don't put spaces in the name as for .sh files spaces are the seperations bet#SBATCH --partition=partitionname # If you don't know ask :)
#SBATCH --gres=gpu:1
#SBATCH --partition=neurophys
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=6000 # increase if you need more but shouldn't need it
#SBATCH --time=0 # This is set for maximum time limit for the partition. For the neurophys partition maximum time lim#SBATCH --output=slurm-%j.out # this and the bottom file are the terminal outputs and the error outputs files so if y#SBATCH --error=slurm-%j.err # If you are sure your .sh file will run then you can delete the.err file
#SBATCH --mail-type=END,FAIL #Sends you an e-mail if it either ends or fails to the user mail below
#SBATCH --mail-user=username@donders.ru.nl #This and the above line can be deleted if you don't want the notification# ---- Config ----
#SBATCH --output=slurm-%j.out
#SBATCH --output=slurm-%j.err
PROJECT_DIR="/vol/battaglialab/imaging1/vinck_students/emiel/thesis-code"
#
ENV_PATH="/vol/battaglialab/imaging1/vinck_students/emiel/miniconda3/envs/.venv"
# Directory containing protocol file
PROTOCOL_DIR="/vol/battaglialab/imaging1/vinck_students/emiel/thesis-code/runs"
# Protocol file
PROTOCOL="$PROTOCOL_DIR/run.txt"
source ~/.bashrc
conda activate $ENV_PATH
cd $PROJECT_DIR
# ---- Sequential execution from tasklist ----
echo "Started running tasks."
while read -r line || [[ -n "$line" ]]
do
        echo "Read line: '$line'"
    # skip empty lines or comments
    [[ -z "$line" || "$line" =~ ^# ]] && continue
    echo "Running: python $line"
    python "$line"
done < "$PROTOCOL"
echo "All tasks finished."