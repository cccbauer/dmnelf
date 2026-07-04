#!/bin/bash
#SBATCH --job-name=rtbpd_efp_nf2
#SBATCH --output=/projects/swglab/data/DMNELF/analysis/fingerprint/efp_meirhasson/logs/rtbpd_cache_nf2_%a.out
#SBATCH --error=/projects/swglab/data/DMNELF/analysis/fingerprint/efp_meirhasson/logs/rtbpd_cache_nf2_%a.err
#SBATCH --partition=short
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --array=0-11
PY=/home/cccbauer/.conda/envs/eeg_preproc/bin/python
cd /projects/swglab/data/DMNELF/analysis/fingerprint/efp_meirhasson/scripts
SUBS=(rtbpd002 rtbpd003 rtbpd009 rtbpd010 rtbpd011 rtbpd015 rtbpd018 rtbpd022 rtbpd026 rtbpd027 rtbpd028 rtbpd030)
SUB=${SUBS[$SLURM_ARRAY_TASK_ID]}
${PY} efp_features.py --subjects ${SUB} --config /projects/swglab/data/DMNELF/analysis/fingerprint/efp_meirhasson/config_rtbpd_nf2.yaml --cache /projects/swglab/data/DMNELF/analysis/fingerprint/efp_meirhasson/results/features_cache_rtbpd_nf2
