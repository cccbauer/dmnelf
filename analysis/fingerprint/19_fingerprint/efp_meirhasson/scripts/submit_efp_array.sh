#!/bin/bash
#SBATCH --job-name=efp19
#SBATCH --output=/projects/swglab/data/DMNELF/analysis/fingerprint/19_fingerprint/efp_meirhasson/logs/efp_%a.out
#SBATCH --error=/projects/swglab/data/DMNELF/analysis/fingerprint/19_fingerprint/efp_meirhasson/logs/efp_%a.err
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --partition=short
#SBATCH --array=0-18

PYTHON=/home/cccbauer/.conda/envs/eeg_preproc/bin/python
BASE=/projects/swglab/data/DMNELF/analysis/fingerprint/19_fingerprint/efp_meirhasson
SCRIPT_DIR=$BASE/scripts
CACHE=$BASE/results/features_cache

SUBJECTS=(dmnelf001 dmnelf002 dmnelf003 dmnelf004 dmnelf005 dmnelf006 dmnelf007 dmnelf008 \
          dmnelf009 dmnelf010 dmnelf011 dmnelf012 dmnelf013 dmnelf014 \
          dmnelf015 dmnelf016 dmnelf1001 dmnelf1002 dmnelf1003)
SUB=${SUBJECTS[$SLURM_ARRAY_TASK_ID]}

cd $SCRIPT_DIR
${PYTHON} efp_decode.py --subjects $SUB --outdir full --cache $CACHE
