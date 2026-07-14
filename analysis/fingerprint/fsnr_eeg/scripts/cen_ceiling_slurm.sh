#!/bin/bash
#SBATCH --job-name=cenrel
#SBATCH --partition=short
#SBATCH --time=00:40:00
#SBATCH --mem=12G
#SBATCH --cpus-per-task=2
#SBATCH --output=/home/cccbauer/cenrel_out/logs/%x_%A_%a.out
COHORT=$1
SUBLIST=$2
SUB=$(sed -n "$((SLURM_ARRAY_TASK_ID+1))p" "$SUBLIST")
echo "[$(date)] cohort=$COHORT sub=$SUB"
/home/cccbauer/.conda/envs/eeg_preproc/bin/python -u /home/cccbauer/cen_ceiling_extract.py \
    --cohort "$COHORT" --subject "$SUB" --out /home/cccbauer/cenrel_out
echo "[$(date)] done $SUB"
