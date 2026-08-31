#!/bin/bash
#SBATCH --job-name=rtbpd_sphere
#SBATCH --output=/projects/swglab/data/DMNELF/analysis/fingerprint/efp_meirhasson/logs/rtbpd_sphere.out
#SBATCH --error=/projects/swglab/data/DMNELF/analysis/fingerprint/efp_meirhasson/logs/rtbpd_sphere.err
#SBATCH --partition=short
#SBATCH --time=00:50:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G

# Calcarine 6mm sphere (VIS target) from rtBPD BOLD.
PY=/home/cccbauer/.conda/envs/eeg_preproc/bin/python
BASE=/projects/swglab/data/DMNELF/analysis/fingerprint/efp_meirhasson
cd $BASE/scripts
${PY} extract_visual_sphere.py --group --config $BASE/config_rtbpd.yaml
