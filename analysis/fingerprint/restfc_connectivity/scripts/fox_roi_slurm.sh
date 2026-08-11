#!/bin/bash
#SBATCH --job-name=foxroi
#SBATCH --partition=short
#SBATCH --time=01:30:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=2
#SBATCH --output=/home/cccbauer/foxseed_out/logs/foxroi_%A_%a.out
C=$1
/home/cccbauer/.conda/envs/eeg_preproc/bin/python -u /home/cccbauer/fox_seed_roi.py --cohort $C --out /home/cccbauer/foxseed_out
