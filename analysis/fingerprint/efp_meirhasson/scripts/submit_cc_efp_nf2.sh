#!/bin/bash
#SBATCH --job-name=cc_efp_nf2
#SBATCH --output=/projects/swglab/data/DMNELF/analysis/fingerprint/efp_meirhasson/logs/cc_efp_nf2.out
#SBATCH --error=/projects/swglab/data/DMNELF/analysis/fingerprint/efp_meirhasson/logs/cc_efp_nf2.err
#SBATCH --partition=short
#SBATCH --time=00:40:00
#SBATCH --mem=48G
#SBATCH --cpus-per-task=4
cd /projects/swglab/data/DMNELF/analysis/fingerprint/efp_meirhasson/scripts
/home/cccbauer/.conda/envs/eeg_preproc/bin/python cross_cohort_efp.py \
  --rtbpd-config /projects/swglab/data/DMNELF/analysis/fingerprint/efp_meirhasson/config_rtbpd_nf2.yaml \
  --dmnelf-cache /projects/swglab/data/DMNELF/analysis/fingerprint/efp_meirhasson/results/features_cache \
  --rtbpd-cache /projects/swglab/data/DMNELF/analysis/fingerprint/efp_meirhasson/results/features_cache_rtbpd_nf2 --res tr --tag _nf2
