#!/usr/bin/env bash
set -euo pipefail

CLUSTER="cccbauer@explorer.northeastern.edu"
PROJ="/projects/swglab/data/DMNELF/analysis/fingerprint/eeg_bold_coupling"
SCRIPTS_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJ_DIR="$(dirname "$SCRIPTS_DIR")"

echo "=== Deploying eeg_bold_coupling to cluster ==="

# ── 1. Sync files ──
echo "Syncing scripts + config..."
scp "$PROJ_DIR/config.yaml" "${CLUSTER}:${PROJ}/config.yaml"
scp "$SCRIPTS_DIR/bandpower.py" "${CLUSTER}:${PROJ}/scripts/bandpower.py"
scp "$SCRIPTS_DIR/multivariate_decode_pda.py" "${CLUSTER}:${PROJ}/scripts/multivariate_decode_pda.py"
scp "$SCRIPTS_DIR/plot_topomaps.py" "${CLUSTER}:${PROJ}/scripts/plot_topomaps.py"

# ── 2. Create sbatch job scripts on cluster ──
echo "Creating job scripts on cluster..."
ssh "$CLUSTER" << 'REMOTE'
set -e
PROJ="/projects/swglab/data/DMNELF/analysis/fingerprint/eeg_bold_coupling"
cd "$PROJ"
mkdir -p logs results/multivariate results/multivariate_loro

# --- Job A: Full run (all targets, ridge+elasticnet, 10K nulls, 5-fold) ---
cat > scripts/decode_full_job.sh << 'SLURM'
#!/bin/bash
#SBATCH --job-name=decode_full
#SBATCH --output=logs/decode_full_%a.out
#SBATCH --error=logs/decode_full_%a.err
#SBATCH --array=0-15
#SBATCH --partition=short
#SBATCH --mem=8G
#SBATCH --time=02:00:00

SUBJECTS=(dmnelf001 dmnelf004 dmnelf005 dmnelf006 dmnelf007 dmnelf008 dmnelf009 dmnelf010 dmnelf011 dmnelf012 dmnelf013 dmnelf014 dmnelf015 dmnelf1001 dmnelf1002 dmnelf1003)
SUB=${SUBJECTS[$SLURM_ARRAY_TASK_ID]}

cd /projects/swglab/data/DMNELF/analysis/fingerprint/eeg_bold_coupling/scripts
/home/cccbauer/.conda/envs/eeg_preproc/bin/python multivariate_decode_pda.py \
    --subjects $SUB \
    --n_shuffles 10000
SLURM

# --- Job B: LORO (all targets, ridge+elasticnet, 10K nulls) ---
cat > scripts/decode_loro_job.sh << 'SLURM'
#!/bin/bash
#SBATCH --job-name=decode_loro
#SBATCH --output=logs/decode_loro_%a.out
#SBATCH --error=logs/decode_loro_%a.err
#SBATCH --array=0-15
#SBATCH --partition=short
#SBATCH --mem=8G
#SBATCH --time=02:00:00

SUBJECTS=(dmnelf001 dmnelf004 dmnelf005 dmnelf006 dmnelf007 dmnelf008 dmnelf009 dmnelf010 dmnelf011 dmnelf012 dmnelf013 dmnelf014 dmnelf015 dmnelf1001 dmnelf1002 dmnelf1003)
SUB=${SUBJECTS[$SLURM_ARRAY_TASK_ID]}

cd /projects/swglab/data/DMNELF/analysis/fingerprint/eeg_bold_coupling/scripts
/home/cccbauer/.conda/envs/eeg_preproc/bin/python multivariate_decode_pda.py \
    --loro \
    --subjects $SUB \
    --n_shuffles 10000
SLURM

# --- Job C: Group tests + topomaps (runs after A and B finish) ---
cat > scripts/decode_group_job.sh << 'SLURM'
#!/bin/bash
#SBATCH --job-name=decode_group
#SBATCH --output=logs/decode_group.out
#SBATCH --error=logs/decode_group.err
#SBATCH --partition=short
#SBATCH --mem=4G
#SBATCH --time=00:30:00

cd /projects/swglab/data/DMNELF/analysis/fingerprint/eeg_bold_coupling/scripts
PY=/home/cccbauer/.conda/envs/eeg_preproc/bin/python

echo "=== 5-fold group test ==="
$PY multivariate_decode_pda.py --group

echo ""
echo "=== LORO group test ==="
$PY multivariate_decode_pda.py --loro --group

echo ""
echo "=== Topomaps ==="
$PY plot_topomaps.py --target GSR_CEN
$PY plot_topomaps.py --target GSR_CEN --loro
$PY plot_topomaps.py --target PDA
$PY plot_topomaps.py --target RAW_DMN
SLURM

chmod +x scripts/decode_full_job.sh scripts/decode_loro_job.sh scripts/decode_group_job.sh
echo "Job scripts created."
REMOTE

# ── 3. Submit jobs ──
echo ""
echo "Submitting jobs..."
ssh "$CLUSTER" << 'SUBMIT'
cd /projects/swglab/data/DMNELF/analysis/fingerprint/eeg_bold_coupling

JOB_A=$(sbatch --parsable scripts/decode_full_job.sh)
echo "5-fold job array: $JOB_A"

JOB_B=$(sbatch --parsable scripts/decode_loro_job.sh)
echo "LORO job array:   $JOB_B"

JOB_C=$(sbatch --parsable --dependency=afterok:${JOB_A}:${JOB_B} scripts/decode_group_job.sh)
echo "Group test job:   $JOB_C (depends on $JOB_A and $JOB_B)"

echo ""
echo "Monitor: squeue -u cccbauer"
SUBMIT

echo ""
echo "=== Deploy complete ==="
echo "32 subject jobs (16 x 5-fold + 16 x LORO) + 1 group job submitted."
echo "Each subject: all 4 targets, ridge + elasticnet, 10K circular-shift nulls."
echo ""
echo "After completion, pull results:"
echo "  scp -r ${CLUSTER}:${PROJ}/results/multivariate/ $PROJ_DIR/results/multivariate_cluster/"
echo "  scp -r ${CLUSTER}:${PROJ}/results/multivariate_loro/ $PROJ_DIR/results/multivariate_loro_cluster/"
