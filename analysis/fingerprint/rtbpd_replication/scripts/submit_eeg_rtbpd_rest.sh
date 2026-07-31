#!/bin/bash
#SBATCH --job-name=rtbpd_eeg_rest
#SBATCH --account=suewhit
#SBATCH --output=/projects/swglab/data/rtBPD/analysis/fingerprint/rtbpd_replication/logs/eeg_rest_%a.out
#SBATCH --error=/projects/swglab/data/rtBPD/analysis/fingerprint/rtbpd_replication/logs/eeg_rest_%a.err
#SBATCH --partition=short
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --array=0-14

# EEG preprocessing (EDF -> ICA-cleaned 500 Hz FIF) for task-rest, runs 01-04,
# for the 15 rtBPD nf1 subjects confirmed (Phase 0 preflight) to have a
# COMPLETE 4-run rest set. Excluded (partial/missing rest runs, no nf2
# fallback per user decision): rtbpd004, rtbpd022, rtbpd026, rtbpd027, rtbpd028,
# rtbpd034.
#
# Session remap: pilots (rtbpd002/003) raw EEG is under ses-nf; the other 13
# subjects' raw EEG is under ses-nf1. All outputs are written to ses-nf so
# rest derivatives land under the same session label as the already-processed
# task-feedback derivatives.
PY=/home/cccbauer/.conda/envs/eeg_preproc/bin/python
BASE=/projects/swglab/data/rtBPD/analysis/fingerprint/rtbpd_replication
cd $BASE/scripts

SUBS=(sub-rtbpd002 sub-rtbpd003 \
      sub-rtbpd009 sub-rtbpd010 sub-rtbpd011 sub-rtbpd012 sub-rtbpd013 sub-rtbpd015 \
      sub-rtbpd018 sub-rtbpd020 sub-rtbpd021 sub-rtbpd024 sub-rtbpd030 sub-rtbpd038 \
      sub-rtbpd040)
RAW_SESSIONS=(ses-nf ses-nf \
      ses-nf1 ses-nf1 ses-nf1 ses-nf1 ses-nf1 ses-nf1 \
      ses-nf1 ses-nf1 ses-nf1 ses-nf1 ses-nf1 ses-nf1 \
      ses-nf1)

SUB=${SUBS[$SLURM_ARRAY_TASK_ID]}
RAW_SES=${RAW_SESSIONS[$SLURM_ARRAY_TASK_ID]}

echo "[$SUB] EEG rest preproc start $(date)  raw-session=${RAW_SES}  out-session=ses-nf"
${PY} eeg_preproc_rtbpd.py --subject ${SUB} --raw-session ${RAW_SES} --out-session ses-nf \
      --tasks rest --sfreq 500
echo "[$SUB] done $(date)"
