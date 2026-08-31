#!/bin/bash
#SBATCH --job-name=is_swap
#SBATCH --account=suewhit
#SBATCH --partition=sharing
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=/projects/swglab/data/DMNELF/analysis/fingerprint/cyclic_transcoder/logs/is_swap_%j.out
#SBATCH --error=/projects/swglab/data/DMNELF/analysis/fingerprint/cyclic_transcoder/logs/is_swap_%j.err
/home/cccbauer/.conda/envs/eeg_preproc/bin/python /projects/swglab/data/DMNELF/analysis/fingerprint/cyclic_transcoder/scripts/swap_eeg_infraslow.py   --src /projects/swglab/data/DMNELF/derivatives/cyclic_features --dst /projects/swglab/data/DMNELF/derivatives/cyclic_features_infraslow   --eeg-root /projects/swglab/data/DMNELF/derivatives/eeg_preprocessed --desc preproc500HzISp01 --samples-per-tr 600
echo DONE_SWAP
