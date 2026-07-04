#!/bin/bash
#SBATCH --job-name=rtbpd_sph_nf2
#SBATCH --output=/projects/swglab/data/DMNELF/analysis/fingerprint/efp_meirhasson/logs/rtbpd_sphere_nf2.out
#SBATCH --error=/projects/swglab/data/DMNELF/analysis/fingerprint/efp_meirhasson/logs/rtbpd_sphere_nf2.err
#SBATCH --partition=short
#SBATCH --time=00:50:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
PY=/home/cccbauer/.conda/envs/eeg_preproc/bin/python
cd /projects/swglab/data/DMNELF/analysis/fingerprint/efp_meirhasson/scripts
${PY} extract_visual_sphere.py --group --config /projects/swglab/data/DMNELF/analysis/fingerprint/efp_meirhasson/config_rtbpd_nf2.yaml
