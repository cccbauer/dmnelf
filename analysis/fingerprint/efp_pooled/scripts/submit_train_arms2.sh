#!/bin/bash
#SBATCH --job-name=efp_arms2
#SBATCH --output=/projects/swglab/data/DMNELF/analysis/fingerprint/efp_pooled/results/logs/arms2_%j.out
#SBATCH --error=/projects/swglab/data/DMNELF/analysis/fingerprint/efp_pooled/results/logs/arms2_%j.err
#SBATCH --partition=short
#SBATCH --time=03:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#
# Four single-variable-changed arms vs the dmnelfonly19-with-fixes baseline
# (models/dmnelfonly_epoc12_clean_ridge.npz -- CEN grouped-CV +0.042, DMN +0.032; locked test
# CEN r=+0.019 p=.34, DMN r=+0.049 p=.026). All --dmnelf-only so pooling is not a confound here
# (already tested separately in Phase 2). Selection is by grouped-CV only -- the locked test set
# has already been spent on 3 arms; only the single best of these 4 gets a final locked-set score.
#   arm C  montage=epoc_afproxy  ridge       (EPOC12 + Fp1/Fp2)
#   arm D  montage=cap31         ridge       (full research montage ceiling)
#   arm E  montage=epoc12        elasticnet
#   arm F  montage=epoc12        pls         (joint CEN+DMN fit, new estimator)

set -euo pipefail
P=/home/cccbauer/.conda/envs/eeg_preproc/bin/python
D=/projects/swglab/data/DMNELF/analysis/fingerprint/efp_pooled
mkdir -p $D/results/logs $D/models
cd $D/scripts

echo "################ ARM C: dmnelf-only, epoc_afproxy, clean, ridge"
$P train_pooled.py --montage epoc_afproxy --targets clean --estimator ridge --dmnelf-only \
   --out $D/models/dmnelfonly_epoc_afproxy_clean_ridge.npz

echo ""
echo "################ ARM D: dmnelf-only, cap31, clean, ridge"
$P train_pooled.py --montage cap31 --targets clean --estimator ridge --dmnelf-only \
   --out $D/models/dmnelfonly_cap31_clean_ridge.npz

echo ""
echo "################ ARM E: dmnelf-only, epoc12, clean, elasticnet"
$P train_pooled.py --montage epoc12 --targets clean --estimator elasticnet --dmnelf-only \
   --out $D/models/dmnelfonly_epoc12_clean_elasticnet.npz

echo ""
echo "################ ARM F: dmnelf-only, epoc12, clean, pls (joint CEN+DMN)"
$P train_pooled.py --montage epoc12 --targets clean --estimator pls --dmnelf-only \
   --out $D/models/dmnelfonly_epoc12_clean_pls.npz

echo ""
echo "DONE"
