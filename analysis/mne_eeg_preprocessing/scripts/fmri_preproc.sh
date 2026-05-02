#!/bin/bash
# fmri_preproc.sh
# Master SLURM script for fMRI preprocessing pipeline
# Runs: DiFuMo extraction, microstate fitting, feature extraction, etc.
#
# This script should be submitted to SLURM and runs all fMRI preprocessing steps
# in sequence with proper dependencies.
#
# Usage:
#   sbatch fmri_preproc.sh
#   sbatch fmri_preproc.sh --subject sub-dmnelf012
#   sbatch fmri_preproc.sh --all --overwrite

#SBATCH --job-name=fmri_preproc_pipeline
#SBATCH --output=/projects/swglab/data/DMNELF/analysis/MNE/jupyter/microstate_pda_v3/logs/fmri_preproc_pipeline_%j.out
#SBATCH --error=/projects/swglab/data/DMNELF/analysis/MNE/jupyter/microstate_pda_v3/logs/fmri_preproc_pipeline_%j.err
#SBATCH --partition=short
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --account=suewhit

set -e  # Exit on error

PYTHON="/home/cccbauer/.conda/envs/eeg_preproc/bin/python"
CLUSTER_BASE="/projects/swglab/data/DMNELF/analysis/MNE/jupyter/microstate_pda_v3"
LOGS_DIR="$CLUSTER_BASE/logs"

# Create logs directory if needed
mkdir -p "$LOGS_DIR"

echo "=============================================="
echo "fMRI Preprocessing Pipeline"
echo "=============================================="
echo "Start time: $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Host: $(hostname)"
echo ""

# Parse arguments (passed from deploy script)
ARGS=""
if [ $# -gt 0 ]; then
    ARGS="$@"
fi

# ── STEP 0: Extract DiFuMo-64 Timeseries ──────────────────
echo "[STEP 0] Extracting DiFuMo-64 parcel timeseries..."
echo "Command: $PYTHON $CLUSTER_BASE/deploy_scripts/00_extract_difumo.py $ARGS"
$PYTHON "$CLUSTER_BASE/deploy_scripts/00_extract_difumo.py" $ARGS
STATUS=$?
if [ $STATUS -ne 0 ]; then
    echo "ERROR: DiFuMo extraction failed with status $STATUS"
    exit $STATUS
fi
echo "✓ DiFuMo extraction complete"
echo ""

# ── STEP 0B: Extract Personal Masks ──────────────────────
echo "[STEP 0B] Extracting personal brain masks..."
echo "Command: $PYTHON $CLUSTER_BASE/deploy_scripts/00b_extract_personal_masks.py $ARGS"
$PYTHON "$CLUSTER_BASE/deploy_scripts/00b_extract_personal_masks.py" $ARGS
STATUS=$?
if [ $STATUS -ne 0 ]; then
    echo "ERROR: Personal mask extraction failed with status $STATUS"
    exit $STATUS
fi
echo "✓ Personal mask extraction complete"
echo ""

# ── STEP 0C: Add Personal Parcels ────────────────────────
echo "[STEP 0C] Adding personal PDA parcels..."
echo "Command: $PYTHON $CLUSTER_BASE/deploy_scripts/00c_add_personal_parcels.py $ARGS"
$PYTHON "$CLUSTER_BASE/deploy_scripts/00c_add_personal_parcels.py" $ARGS
STATUS=$?
if [ $STATUS -ne 0 ]; then
    echo "ERROR: Personal parcel addition failed with status $STATUS"
    exit $STATUS
fi
echo "✓ Personal parcel addition complete"
echo ""

# ── STEP 0D: Extract Personal PDA ────────────────────────
echo "[STEP 0D] Extracting personal PDA features..."
echo "Command: $PYTHON $CLUSTER_BASE/deploy_scripts/00d_extract_personal_pda.py $ARGS"
$PYTHON "$CLUSTER_BASE/deploy_scripts/00d_extract_personal_pda.py" $ARGS
STATUS=$?
if [ $STATUS -ne 0 ]; then
    echo "ERROR: Personal PDA extraction failed with status $STATUS"
    exit $STATUS
fi
echo "✓ Personal PDA extraction complete"
echo ""

# ── STEP 1: Fit Microstate Maps ──────────────────────────
echo "[STEP 1] Fitting microstate maps..."
echo "Command: $PYTHON $CLUSTER_BASE/deploy_scripts/01_fit_microstates.py $ARGS"
$PYTHON "$CLUSTER_BASE/deploy_scripts/01_fit_microstates.py" $ARGS
STATUS=$?
if [ $STATUS -ne 0 ]; then
    echo "ERROR: Microstate fitting failed with status $STATUS"
    exit $STATUS
fi
echo "✓ Microstate fitting complete"
echo ""

# ── STEP 2: Extract TESS Features ────────────────────────
echo "[STEP 2] Computing TESS features..."
echo "Command: $PYTHON $CLUSTER_BASE/deploy_scripts/02_tess_features.py $ARGS"
$PYTHON "$CLUSTER_BASE/deploy_scripts/02_tess_features.py" $ARGS
STATUS=$?
if [ $STATUS -ne 0 ]; then
    echo "ERROR: TESS feature extraction failed with status $STATUS"
    exit $STATUS
fi
echo "✓ TESS feature extraction complete"
echo ""

# ── STEP 3: Compute PDA ──────────────────────────────────
echo "[STEP 3] Computing Pattern Distinctiveness Analysis..."
echo "Command: $PYTHON $CLUSTER_BASE/deploy_scripts/03_compute_pda.py $ARGS"
$PYTHON "$CLUSTER_BASE/deploy_scripts/03_compute_pda.py" $ARGS
STATUS=$?
if [ $STATUS -ne 0 ]; then
    echo "ERROR: PDA computation failed with status $STATUS"
    exit $STATUS
fi
echo "✓ PDA computation complete"
echo ""

# ── Optional: Generate Plots ─────────────────────────────
if [ -f "$CLUSTER_BASE/deploy_scripts/05_plot_microstate_pda_epochs.py" ]; then
    echo "[OPTIONAL] Generating plots..."
    echo "Command: $PYTHON $CLUSTER_BASE/deploy_scripts/05_plot_microstate_pda_epochs.py $ARGS"
    $PYTHON "$CLUSTER_BASE/deploy_scripts/05_plot_microstate_pda_epochs.py" $ARGS
    echo "✓ Plots generated"
    echo ""
fi

echo "=============================================="
echo "✓ fMRI preprocessing pipeline complete!"
echo "=============================================="
echo "End time: $(date)"
echo ""
echo "Outputs saved to:"
echo "  - DiFuMo timeseries: /projects/swglab/data/DMNELF/analysis/MNE/jupyter/neurobolt/difumo_timeseries/"
echo "  - Microstate maps: /projects/swglab/data/DMNELF/analysis/MNE/jupyter/neurobolt/microstate_maps/"
echo "  - PDA features: /projects/swglab/data/DMNELF/analysis/MNE/jupyter/neurobolt/pda_features/"
echo ""
