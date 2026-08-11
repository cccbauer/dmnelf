#!/bin/bash
#SBATCH --job-name=rtbpd_eeg_rest_exp
#SBATCH --account=suewhit
#SBATCH --output=/projects/swglab/data/rtBPD/analysis/fingerprint/rtbpd_replication/connectivity_prepost/logs/eeg_rest_exp_%a.out
#SBATCH --error=/projects/swglab/data/rtBPD/analysis/fingerprint/rtbpd_replication/connectivity_prepost/logs/eeg_rest_exp_%a.err
#SBATCH --partition=short
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --array=0-3

# Cohort-expansion preprocessing (task-rest only) for the 4 subjects with
# usable (>=3/4 runs) rest EEG identified 2026-08-11: rtbpd022 (missing
# run-03), rtbpd026 (missing run-01), rtbpd039 (missing run-01), rtbpd045
# (complete 4/4). All raw under ses-nf1 -> written to ses-nf to match the
# existing 15-subject derivatives naming.
PY=/home/cccbauer/.conda/envs/eeg_preproc/bin/python
BASE=/projects/swglab/data/rtBPD/analysis/fingerprint/rtbpd_replication
cd $BASE/scripts

SUBS=(sub-rtbpd022 sub-rtbpd026 sub-rtbpd039 sub-rtbpd045)
SUB=${SUBS[$SLURM_ARRAY_TASK_ID]}

echo "[$SUB] EEG rest preproc start $(date)  raw-session=ses-nf1  out-session=ses-nf"
${PY} eeg_preproc_rtbpd.py --subject ${SUB} --raw-session ses-nf1 --out-session ses-nf \
      --tasks rest --sfreq 500
echo "[$SUB] done $(date)"
