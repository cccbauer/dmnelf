#!/bin/bash
#SBATCH --job-name=restfc
#SBATCH --partition=short
#SBATCH --time=01:30:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=2
#SBATCH --output=/home/cccbauer/restfc_out/logs/%x_%A_%a.out
# Usage (submit driver builds the list + array):
#   COHORT=$1  SUBLIST=$2  (one subject id per line, no "sub-" prefix)
COHORT=$1
SUBLIST=$2
SUB=$(sed -n "$((SLURM_ARRAY_TASK_ID+1))p" "$SUBLIST")
echo "[$(date)] cohort=$COHORT sub=$SUB task=$SLURM_ARRAY_TASK_ID"
/home/cccbauer/.conda/envs/eeg_preproc/bin/python /home/cccbauer/restfc_extract.py \
    --cohort "$COHORT" --subject "$SUB" --out /home/cccbauer/restfc_out
echo "[$(date)] done $SUB"
