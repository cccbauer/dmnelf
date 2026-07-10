#!/bin/bash
#SBATCH --job-name=foxseed
#SBATCH --partition=short
#SBATCH --time=01:30:00
#SBATCH --mem=20G
#SBATCH --cpus-per-task=2
#SBATCH --output=/home/cccbauer/foxseed_out/logs/%x_%A_%a.out
COHORT=$1
SUBLIST=$2
SUB=$(sed -n "$((SLURM_ARRAY_TASK_ID+1))p" "$SUBLIST")
echo "[$(date)] cohort=$COHORT sub=$SUB"
/home/cccbauer/.conda/envs/eeg_preproc/bin/python /home/cccbauer/fox_seed.py \
    --cohort "$COHORT" --subject "$SUB" --out /home/cccbauer/foxseed_out
echo "[$(date)] done $SUB"
