#!/bin/bash
#SBATCH --job-name=eeg_specparam
#SBATCH --output=/projects/swglab/data/DMNELF/analysis/fingerprint/fsnr_eeg/logs/specparam_%a.out
#SBATCH --error=/projects/swglab/data/DMNELF/analysis/fingerprint/fsnr_eeg/logs/specparam_%a.err
#SBATCH --partition=short
#SBATCH --time=00:40:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --array=0-16

SUBS=(dmnelf001 dmnelf004 dmnelf005 dmnelf006 dmnelf007 dmnelf008 dmnelf009 dmnelf010 dmnelf011 dmnelf012 dmnelf013 dmnelf014 dmnelf015 dmnelf016 dmnelf1001 dmnelf1002 dmnelf1003)
SUB=${SUBS[$SLURM_ARRAY_TASK_ID]}
BASE=/projects/swglab/data/DMNELF/analysis/fingerprint/fsnr_eeg
PY=/home/cccbauer/.conda/envs/eeg_preproc/bin/python
mkdir -p "$BASE/logs"
cd "$BASE/scripts"
$PY eeg_fsnr_specparam.py --subjects "$SUB"
