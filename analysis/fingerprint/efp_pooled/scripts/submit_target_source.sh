#!/bin/bash
#SBATCH --job-name=tgt_src
#SBATCH --output=/projects/swglab/data/DMNELF/analysis/fingerprint/efp_pooled/results/logs/tgtsrc_%j.out
#SBATCH --error=/projects/swglab/data/DMNELF/analysis/fingerprint/efp_pooled/results/logs/tgtsrc_%j.err
#SBATCH --partition=short
#SBATCH --time=05:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#
# Harness validation + the real comparison. Using the RESEARCH PIPELINE's own tgt_tr targets
# (which the +0.157 LOSO PDA benchmark used), rerun both:
#   1. single-electrode  -> should reproduce ~+0.157 PDA at Pz if the harness is sound
#   2. montage ablation  -> then the Pz / dimensionality question is apples-to-apples
set -euo pipefail
P=/home/cccbauer/.conda/envs/eeg_preproc/bin/python
D=/projects/swglab/data/DMNELF/analysis/fingerprint/efp_pooled
cd $D/scripts
echo "################ single-electrode, cache targets"
$P montage_pz_ablation.py --single --target-source cache --out $D/results/single_electrode_cachetgt_loso.csv
echo ""
echo "################ montage ablation, cache targets"
$P montage_pz_ablation.py --target-source cache --out $D/results/montage_pz_cachetgt_loso.csv
echo DONE
