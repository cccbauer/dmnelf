#!/bin/bash
#SBATCH --job-name=efp_train
#SBATCH --output=/projects/swglab/data/DMNELF/analysis/fingerprint/efp_pooled/results/logs/train_%j.out
#SBATCH --error=/projects/swglab/data/DMNELF/analysis/fingerprint/efp_pooled/results/logs/train_%j.err
#SBATCH --partition=short
#SBATCH --time=03:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#
# Phase 2, first pass: the two arms that isolate POOLING from the four defect fixes.
#   arm A  pooled (28 subj)  epoc12 clean ridge
#   arm B  DMNELF-only (19)  epoc12 clean ridge   <- same fixes, no pooling
# Both are then scored on the locked held-out set by eval_holdout.py.

set -euo pipefail
P=/home/cccbauer/.conda/envs/eeg_preproc/bin/python
D=/projects/swglab/data/DMNELF/analysis/fingerprint/efp_pooled
mkdir -p $D/results/logs $D/models
cd $D/scripts

echo "################ ARM A: pooled, epoc12, clean, ridge"
$P train_pooled.py --montage epoc12 --targets clean --estimator ridge \
   --out $D/models/pooled_epoc12_clean_ridge.npz

echo ""
echo "################ ARM B: DMNELF-only ablation, epoc12, clean, ridge"
$P train_pooled.py --montage epoc12 --targets clean --estimator ridge --dmnelf-only \
   --out $D/models/dmnelfonly_epoc12_clean_ridge.npz

echo ""
echo "DONE"
