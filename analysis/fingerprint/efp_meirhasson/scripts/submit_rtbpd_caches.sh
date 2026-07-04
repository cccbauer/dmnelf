#!/bin/bash
#SBATCH --job-name=rtbpd_efp
#SBATCH --output=/projects/swglab/data/DMNELF/analysis/fingerprint/efp_meirhasson/logs/rtbpd_cache_%a.out
#SBATCH --error=/projects/swglab/data/DMNELF/analysis/fingerprint/efp_meirhasson/logs/rtbpd_cache_%a.err
#SBATCH --partition=short
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --array=0-20

# EFP feature caches (Stockwell bands + targets incl VIS) for rtBPD subjects.
# Requires the rtBPD visual-sphere npz (submit_rtbpd_sphere.sh) to exist first.
PY=/home/cccbauer/.conda/envs/eeg_preproc/bin/python
BASE=/projects/swglab/data/DMNELF/analysis/fingerprint/efp_meirhasson
cd $BASE/scripts
SUBS=(rtbpd002 rtbpd003 rtbpd004 rtbpd009 rtbpd010 rtbpd011 rtbpd012 rtbpd013 \
      rtbpd015 rtbpd018 rtbpd020 rtbpd021 rtbpd022 rtbpd024 rtbpd026 rtbpd027 \
      rtbpd028 rtbpd030 rtbpd034 rtbpd038 rtbpd040)
SUB=${SUBS[$SLURM_ARRAY_TASK_ID]}
${PY} efp_features.py --subjects ${SUB} --config $BASE/config_rtbpd.yaml \
      --cache $BASE/results/features_cache_rtbpd
