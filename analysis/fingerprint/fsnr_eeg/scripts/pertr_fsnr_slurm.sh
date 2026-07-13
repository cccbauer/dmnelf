#!/bin/bash
#SBATCH --job-name=pertrfsnr
#SBATCH --partition=short
#SBATCH --time=02:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=2
#SBATCH --output=/home/cccbauer/pertr_out/logs/%x_%A_%a.out
COHORT=$1
SUBLIST=$2
SUB=$(sed -n "$((SLURM_ARRAY_TASK_ID+1))p" "$SUBLIST")
echo "[$(date)] cohort=$COHORT sub=$SUB"
/home/cccbauer/.conda/envs/eeg_preproc/bin/python -u /home/cccbauer/pertr_fsnr_extract.py \
    --cohort "$COHORT" --subjects "$SUB" --out /home/cccbauer/pertr_out
echo "[$(date)] done $SUB"
