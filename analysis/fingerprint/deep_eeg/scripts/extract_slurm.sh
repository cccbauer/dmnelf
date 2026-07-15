#!/bin/bash
#SBATCH --job-name=deepwin
#SBATCH --partition=short
#SBATCH --time=00:30:00
#SBATCH --mem=12G
#SBATCH --cpus-per-task=2
#SBATCH --output=/home/cccbauer/deep_win/logs/%x_%A_%a.out
# args: COHORT SUBLIST OUT [HI]
SUB=$(sed -n "$((SLURM_ARRAY_TASK_ID+1))p" "$2")
/home/cccbauer/.conda/envs/eeg_preproc/bin/python -u /home/cccbauer/extract_windows.py \
    --cohort "$1" --subject "$SUB" --clean-dir /home/cccbauer/cenrel_out --out "$3" --hi "${4:-40}"
