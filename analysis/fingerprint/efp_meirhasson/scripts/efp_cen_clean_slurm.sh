#!/bin/bash
#SBATCH --job-name=efpcen
#SBATCH --partition=short
#SBATCH --time=00:30:00
#SBATCH --mem=12G
#SBATCH --cpus-per-task=2
#SBATCH --output=/home/cccbauer/efp_cen_out/logs/%x_%A_%a.out
B=/projects/swglab/data/DMNELF/analysis/fingerprint/efp_meirhasson
SUB=$(sed -n "$((SLURM_ARRAY_TASK_ID+1))p" "$1")
echo "[$(date)] sub=$SUB"
cd $B/scripts
/home/cccbauer/.conda/envs/eeg_preproc/bin/python -u efp_cen_clean.py --subject "$SUB" \
    --cenmean-dir /home/cccbauer/cenrel_out --cache $B/results/features_cache --out /home/cccbauer/efp_cen_out
echo "[$(date)] done $SUB"
