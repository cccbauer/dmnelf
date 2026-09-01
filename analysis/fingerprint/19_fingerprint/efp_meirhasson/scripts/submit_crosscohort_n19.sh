#!/bin/bash
#SBATCH --job-name=cc_efp_n19
#SBATCH --output=/projects/swglab/data/DMNELF/analysis/fingerprint/19_fingerprint/efp_meirhasson/logs/cc_efp_n19_%j.out
#SBATCH --error=/projects/swglab/data/DMNELF/analysis/fingerprint/19_fingerprint/efp_meirhasson/logs/cc_efp_n19_%j.err
#SBATCH --partition=short
#SBATCH --time=00:40:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#
# Re-run the DMNELF -> rtBPD cross-cohort validation on the n=19 DMNELF cohort (the existing
# cross_cohort_efp_summary_tr.csv under efp_meirhasson/ is n=17, predates dmnelf002/003).
# This script's PROJ resolves to 19_fingerprint/efp_meirhasson (n=19 cache) by default;
# --rtbpd-cache still points at the shared rtBPD cache (unaffected by the DMNELF expansion).

set -euo pipefail
mkdir -p /projects/swglab/data/DMNELF/analysis/fingerprint/19_fingerprint/efp_meirhasson/logs
cd /projects/swglab/data/DMNELF/analysis/fingerprint/19_fingerprint/efp_meirhasson/scripts
P=/home/cccbauer/.conda/envs/eeg_preproc/bin/python

echo "################ nf1"
$P cross_cohort_efp.py \
  --rtbpd-config /projects/swglab/data/DMNELF/analysis/fingerprint/19_fingerprint/efp_meirhasson/config_rtbpd.yaml \
  --rtbpd-cache /projects/swglab/data/DMNELF/analysis/fingerprint/efp_meirhasson/results/features_cache_rtbpd \
  --res tr

echo ""
echo "################ nf2"
$P cross_cohort_efp.py \
  --rtbpd-config /projects/swglab/data/DMNELF/analysis/fingerprint/19_fingerprint/efp_meirhasson/config_rtbpd_nf2.yaml \
  --rtbpd-cache /projects/swglab/data/DMNELF/analysis/fingerprint/efp_meirhasson/results/features_cache_rtbpd_nf2 \
  --res tr --tag _nf2

echo "DONE"
