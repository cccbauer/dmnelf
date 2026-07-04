#!/bin/bash
#SBATCH --job-name=rtbpd_eeg_nf2
#SBATCH --output=/projects/swglab/data/rtBPD/analysis/fingerprint/rtbpd_replication/logs/eeg_nf2_%a.out
#SBATCH --error=/projects/swglab/data/rtBPD/analysis/fingerprint/rtbpd_replication/logs/eeg_nf2_%a.err
#SBATCH --partition=short
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --array=0-11
PY=/home/cccbauer/.conda/envs/eeg_preproc/bin/python
cd /projects/swglab/data/rtBPD/analysis/fingerprint/rtbpd_replication/scripts
SUBS=(sub-rtbpd002 sub-rtbpd003 sub-rtbpd009 sub-rtbpd010 sub-rtbpd011 sub-rtbpd015 sub-rtbpd018 sub-rtbpd022 sub-rtbpd026 sub-rtbpd027 sub-rtbpd028 sub-rtbpd030)
SUB=${SUBS[$SLURM_ARRAY_TASK_ID]}
${PY} eeg_preproc_rtbpd.py --subject ${SUB} --raw-session ses-nf2 --out-session ses-nf2 --tasks feedback --sfreq 500
