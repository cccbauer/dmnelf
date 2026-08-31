#!/bin/bash
#SBATCH --job-name=efp_eval
#SBATCH --output=/projects/swglab/data/DMNELF/analysis/fingerprint/efp_pooled/results/logs/eval_%j.out
#SBATCH --error=/projects/swglab/data/DMNELF/analysis/fingerprint/efp_pooled/results/logs/eval_%j.err
#SBATCH --partition=short
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#
# Score the shipped efp_epoc_model.npz on each cohort. The login node OOM-kills even
# numpy-only jobs, so everything runs through SLURM.
#
# The shipped model was trained on DMNELF only, so:
#   dmnelf    -> IN-SAMPLE (sanity check; must roughly match the build log)
#   rtbpd     -> HELD OUT  (the number that has never existed)
#   rtbpd_nf2 -> HELD OUT  (second session, also never seen)

set -euo pipefail
P=/home/cccbauer/.conda/envs/eeg_preproc/bin/python
D=/projects/swglab/data/DMNELF/analysis/fingerprint/efp_pooled
R=$D/results
mkdir -p $R/logs
cd $D/scripts

for COH in dmnelf rtbpd rtbpd_nf2; do
  echo ""
  echo "############################################################"
  echo "### cohort: $COH"
  echo "############################################################"
  $P eval_holdout.py --cohort $COH \
     --label "shipped efp_epoc_model.npz (n=19 DMNELF-trained)" \
     --out $R/shipped_on_${COH}.csv
done
echo ""
echo "DONE"
