#!/bin/bash
#SBATCH --job-name=efpcenrt
#SBATCH --partition=short
#SBATCH --time=00:30:00
#SBATCH --mem=12G
#SBATCH --cpus-per-task=2
#SBATCH --output=/home/cccbauer/efp_cen_rt/logs/%x_%A_%a.out
# args: SUBLIST CACHE CONFIG OUT
B=/projects/swglab/data/DMNELF/analysis/fingerprint/efp_meirhasson
SUB=$(sed -n "$((SLURM_ARRAY_TASK_ID+1))p" "$1")
echo "[$(date)] sub=$SUB cache=$2"
cd $B/scripts
/home/cccbauer/.conda/envs/eeg_preproc/bin/python -u efp_cen_clean.py \
    --subject "$SUB" --cache "$2" --config "$3" --out "$4"
echo "[$(date)] done $SUB"
