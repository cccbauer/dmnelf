#!/bin/bash
#SBATCH --job-name=is_loroSB
#SBATCH --account=suewhit
#SBATCH --partition=sharing
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=/projects/swglab/data/DMNELF/analysis/fingerprint/infraslow_loro_pda/logs/loro_%j.out
#SBATCH --error=/projects/swglab/data/DMNELF/analysis/fingerprint/infraslow_loro_pda/logs/loro_%j.err
cd /projects/swglab/data/DMNELF/analysis/fingerprint/infraslow_loro_pda
/home/cccbauer/.conda/envs/eeg_preproc/bin/python scripts/decode_loro.py --config config.yaml --smooth-both
echo DONE_LORO_SMOOTHBOTH
