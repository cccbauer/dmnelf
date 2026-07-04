#!/bin/bash
#SBATCH --job-name=rtbpd_mask
#SBATCH --output=/projects/swglab/data/rtBPD/analysis/fingerprint/rtbpd_replication/logs/mask_%a.out
#SBATCH --error=/projects/swglab/data/rtBPD/analysis/fingerprint/rtbpd_replication/logs/mask_%a.err
#SBATCH --partition=short
#SBATCH --time=03:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --array=0-17

# Personalized DMN/CEN mask extraction (CanICA via pineuro) for the 18 new rtBPD
# subjects (002/003/004 already have masks).
PY=/home/cccbauer/.conda/envs/pineuro/bin/python
BASE=/projects/swglab/data/rtBPD/analysis/fingerprint/rtbpd_replication
cd $BASE/scripts
SUBS=(rtbpd009 rtbpd010 rtbpd011 rtbpd012 rtbpd013 rtbpd015 rtbpd018 rtbpd020 \
      rtbpd021 rtbpd022 rtbpd024 rtbpd026 rtbpd027 rtbpd028 rtbpd030 rtbpd034 \
      rtbpd038 rtbpd040)
SUB=${SUBS[$SLURM_ARRAY_TASK_ID]}
echo "[$SUB] mask extraction start $(date)"
${PY} mask_extraction_rtbpd.py --subject ${SUB} --config mask_config_rtbpd.yaml
echo "[$SUB] done $(date)"
