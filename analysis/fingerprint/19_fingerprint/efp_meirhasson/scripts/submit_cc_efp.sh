#!/bin/bash
#SBATCH --job-name=cc_efp
#SBATCH --output=/projects/swglab/data/DMNELF/analysis/fingerprint/efp_meirhasson/logs/cc_efp.out
#SBATCH --error=/projects/swglab/data/DMNELF/analysis/fingerprint/efp_meirhasson/logs/cc_efp.err
#SBATCH --partition=short
#SBATCH --time=00:40:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
cd /projects/swglab/data/DMNELF/analysis/fingerprint/efp_meirhasson/scripts
/home/cccbauer/.conda/envs/eeg_preproc/bin/python cross_cohort_efp.py \
  --rtbpd-config /projects/swglab/data/DMNELF/analysis/fingerprint/efp_meirhasson/config_rtbpd.yaml \
  --dmnelf-cache /projects/swglab/data/DMNELF/analysis/fingerprint/efp_meirhasson/results/features_cache \
  --rtbpd-cache /projects/swglab/data/DMNELF/analysis/fingerprint/efp_meirhasson/results/features_cache_rtbpd --res tr
