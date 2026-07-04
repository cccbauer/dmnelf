#!/bin/bash
#SBATCH --job-name=rtbpd_feat
#SBATCH --output=/projects/swglab/data/rtBPD/analysis/fingerprint/rtbpd_replication/logs/feat_%a.out
#SBATCH --error=/projects/swglab/data/rtBPD/analysis/fingerprint/rtbpd_replication/logs/feat_%a.err
#SBATCH --partition=short
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --array=0-17

# cyclic_features (DiFuMo-64 + personal DMN/CEN) for the 18 new rtBPD subjects.
# Requires the personalized masks from submit_masks_rtbpd.sh to exist first.
PY=/home/cccbauer/.conda/envs/eeg_preproc/bin/python
BASE=/projects/swglab/data/rtBPD/analysis/fingerprint/rtbpd_replication
cd $BASE/scripts
SUBS=(rtbpd009 rtbpd010 rtbpd011 rtbpd012 rtbpd013 rtbpd015 rtbpd018 rtbpd020 \
      rtbpd021 rtbpd022 rtbpd024 rtbpd026 rtbpd027 rtbpd028 rtbpd030 rtbpd034 \
      rtbpd038 rtbpd040)
SUB=${SUBS[$SLURM_ARRAY_TASK_ID]}
echo "[$SUB] feature extraction start $(date)"
${PY} extract_features_rtbpd.py --subject ${SUB} --config extract_config_rtbpd.yaml
echo "[$SUB] done $(date)"
