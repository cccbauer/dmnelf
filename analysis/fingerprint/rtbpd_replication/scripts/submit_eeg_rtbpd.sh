#!/bin/bash
#SBATCH --job-name=rtbpd_eeg
#SBATCH --output=/projects/swglab/data/rtBPD/analysis/fingerprint/rtbpd_replication/logs/eeg_%a.out
#SBATCH --error=/projects/swglab/data/rtBPD/analysis/fingerprint/rtbpd_replication/logs/eeg_%a.err
#SBATCH --partition=short
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --array=0-17

# EEG preprocessing (EDF -> ICA-cleaned 500 Hz FIF) for the 18 new rtBPD subjects.
# Raw EEG is under ses-nf1; write outputs to ses-nf to unify with the pilot.
PY=/home/cccbauer/.conda/envs/eeg_preproc/bin/python
BASE=/projects/swglab/data/rtBPD/analysis/fingerprint/rtbpd_replication
cd $BASE/scripts
SUBS=(sub-rtbpd009 sub-rtbpd010 sub-rtbpd011 sub-rtbpd012 sub-rtbpd013 sub-rtbpd015 \
      sub-rtbpd018 sub-rtbpd020 sub-rtbpd021 sub-rtbpd022 sub-rtbpd024 sub-rtbpd026 \
      sub-rtbpd027 sub-rtbpd028 sub-rtbpd030 sub-rtbpd034 sub-rtbpd038 sub-rtbpd040)
SUB=${SUBS[$SLURM_ARRAY_TASK_ID]}
echo "[$SUB] EEG preproc start $(date)"
${PY} eeg_preproc_rtbpd.py --subject ${SUB} --raw-session ses-nf1 --out-session ses-nf \
      --tasks feedback --sfreq 500
echo "[$SUB] done $(date)"
