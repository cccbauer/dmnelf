#!/usr/bin/env bash
set -euo pipefail

# deploy_validation.sh — Full pipeline for dmnelf016 cross-subject validation
# Runs: organize EEG → preprocess EEG → extract masks → extract features → cross-subject predict

CLUSTER="cccbauer@explorer.northeastern.edu"
DMNELF="/projects/swglab/data/DMNELF"
PROJ="${DMNELF}/analysis/fingerprint/eeg_bold_coupling"
PY="/home/cccbauer/.conda/envs/eeg_preproc/bin/python"
SUB="sub-dmnelf016"
SUB_SHORT="dmnelf016"

VALIDATION_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJ_DIR="$(dirname "$VALIDATION_DIR")"

echo "=========================================="
echo "  Validation Pipeline for ${SUB}"
echo "=========================================="

# ── 1. Sync scripts to cluster ──
echo ""
echo ">>> [1/6] Syncing scripts to cluster..."
scp "$PROJ_DIR/config.yaml" "${CLUSTER}:${PROJ}/config.yaml"
scp "$PROJ_DIR/scripts/bandpower.py" "${CLUSTER}:${PROJ}/scripts/bandpower.py"
scp "$PROJ_DIR/scripts/multivariate_decode_pda.py" "${CLUSTER}:${PROJ}/scripts/multivariate_decode_pda.py"
scp "$VALIDATION_DIR/cross_subject_predict.py" "${CLUSTER}:${PROJ}/validation/cross_subject_predict.py"

# ── 2. Organize raw EEG (BVA export → rawdata_eeg BIDS) ──
echo ""
echo ">>> [2/6] Organizing raw EEG..."
ssh "$CLUSTER" << REMOTE
set -e
# Check if already done
if ls ${DMNELF}/rawdata_eeg/${SUB}/ses-dmnelf/eeg/*feedback*edf 2>/dev/null | head -1 > /dev/null; then
    echo "  rawdata_eeg already exists for ${SUB}, skipping."
else
    echo "  Running organize_raw_eeg.py..."
    ${PY} ${DMNELF}/analysis/MNE/jupyter/microstate_pda_v3/scripts/organize_raw_eeg.py \
        --subject ${SUB} --move
    echo "  Done."
fi
# Verify
echo "  Checking:"
ls ${DMNELF}/rawdata_eeg/${SUB}/ses-dmnelf/eeg/*feedback*edf 2>/dev/null | wc -l | xargs -I{} echo "    feedback EDFs: {}"
REMOTE

# ── 3. EEG preprocessing (500 Hz) ──
echo ""
echo ">>> [3/6] Submitting EEG preprocessing job..."
ssh "$CLUSTER" << 'PREPROC'
set -e
DMNELF="/projects/swglab/data/DMNELF"
SUB="sub-dmnelf016"
PY="/home/cccbauer/.conda/envs/eeg_preproc/bin/python"

# Check if already done
if ls ${DMNELF}/derivatives/eeg_preprocessed/${SUB}/ses-dmnelf/eeg/*feedback*preproc500Hz*fif 2>/dev/null | head -1 > /dev/null; then
    echo "  EEG already preprocessed for ${SUB}, skipping."
    echo "SKIP_EEG"
else
    cat > /tmp/eeg_preproc_016.sh << 'SLURM'
#!/bin/bash
#SBATCH --job-name=eeg_016
#SBATCH --output=/projects/swglab/data/DMNELF/analysis/fingerprint/eeg_bold_coupling/validation/logs/eeg_preproc_016.out
#SBATCH --error=/projects/swglab/data/DMNELF/analysis/fingerprint/eeg_bold_coupling/validation/logs/eeg_preproc_016.err
#SBATCH --partition=short
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --time=08:00:00

PY=/home/cccbauer/.conda/envs/eeg_preproc/bin/python
SCRIPT=/projects/swglab/data/DMNELF/analysis/MNE/jupyter/microstate_pda_v3/scripts/eeg_preproc.py

$PY $SCRIPT --subject sub-dmnelf016 --sfreq 500
SLURM
    mkdir -p /projects/swglab/data/DMNELF/analysis/fingerprint/eeg_bold_coupling/validation/logs
    JOB_EEG=$(sbatch --parsable /tmp/eeg_preproc_016.sh)
    echo "EEG_JOB=${JOB_EEG}"
fi
PREPROC

# ── 4. Extract personalized DMN/CEN masks ──
echo ""
echo ">>> [4/6] Submitting mask extraction job..."
ssh "$CLUSTER" << 'MASKS'
set -e
DMNELF="/projects/swglab/data/DMNELF"
SUB="sub-dmnelf016"
PY="/home/cccbauer/.conda/envs/eeg_preproc/bin/python"

if ls ${DMNELF}/derivatives/network_masks/${SUB}/*dmn_mask* 2>/dev/null | head -1 > /dev/null; then
    echo "  Masks already exist for ${SUB}, skipping."
    echo "SKIP_MASKS"
else
    cat > /tmp/mask_ext_016.sh << 'SLURM'
#!/bin/bash
#SBATCH --job-name=mask_016
#SBATCH --output=/projects/swglab/data/DMNELF/analysis/fingerprint/eeg_bold_coupling/validation/logs/mask_ext_016.out
#SBATCH --error=/projects/swglab/data/DMNELF/analysis/fingerprint/eeg_bold_coupling/validation/logs/mask_ext_016.err
#SBATCH --partition=short
#SBATCH --mem=16G
#SBATCH --time=01:00:00

PY=/home/cccbauer/.conda/envs/eeg_preproc/bin/python
cd /projects/swglab/data/DMNELF/analysis/fingerprint/cyclic_transcoder
$PY scripts/mask_extraction.py --subject dmnelf016 --config config.yaml
SLURM
    JOB_MASK=$(sbatch --parsable /tmp/mask_ext_016.sh)
    echo "MASK_JOB=${JOB_MASK}"
fi
MASKS

# ── 5. Extract cyclic_features .npz ──
echo ""
echo ">>> [5/6] Submitting feature extraction job..."
ssh "$CLUSTER" << 'FEATURES'
set -e
DMNELF="/projects/swglab/data/DMNELF"
SUB="sub-dmnelf016"
PY="/home/cccbauer/.conda/envs/eeg_preproc/bin/python"

if ls ${DMNELF}/derivatives/cyclic_features/${SUB}/*feedback*features.npz 2>/dev/null | wc -l | grep -q '^4$'; then
    echo "  Features already exist for ${SUB} (4 runs), skipping."
    echo "SKIP_FEATURES"
else
    # This depends on EEG preprocessing AND masks being done.
    # We submit with dependencies if those jobs were submitted above.
    # If they were skipped, submit without dependency.
    cat > /tmp/feat_ext_016.sh << 'SLURM'
#!/bin/bash
#SBATCH --job-name=feat_016
#SBATCH --output=/projects/swglab/data/DMNELF/analysis/fingerprint/eeg_bold_coupling/validation/logs/feat_ext_016.out
#SBATCH --error=/projects/swglab/data/DMNELF/analysis/fingerprint/eeg_bold_coupling/validation/logs/feat_ext_016.err
#SBATCH --partition=short
#SBATCH --mem=16G
#SBATCH --time=02:00:00

PY=/home/cccbauer/.conda/envs/eeg_preproc/bin/python
cd /projects/swglab/data/DMNELF/analysis/fingerprint/cyclic_transcoder
$PY data/extract_features.py --subject dmnelf016 --config config.yaml
SLURM
    JOB_FEAT=$(sbatch --parsable /tmp/feat_ext_016.sh)
    echo "FEAT_JOB=${JOB_FEAT}"
fi
FEATURES

# ── 6. Cross-subject prediction ──
echo ""
echo ">>> [6/6] Submitting cross-subject prediction job..."
ssh "$CLUSTER" << 'PREDICT'
set -e
PROJ="/projects/swglab/data/DMNELF/analysis/fingerprint/eeg_bold_coupling"
PY="/home/cccbauer/.conda/envs/eeg_preproc/bin/python"
mkdir -p ${PROJ}/validation ${PROJ}/results/validation

cat > /tmp/cross_predict_016.sh << 'SLURM'
#!/bin/bash
#SBATCH --job-name=xsub_016
#SBATCH --output=/projects/swglab/data/DMNELF/analysis/fingerprint/eeg_bold_coupling/validation/logs/cross_predict_016.out
#SBATCH --error=/projects/swglab/data/DMNELF/analysis/fingerprint/eeg_bold_coupling/validation/logs/cross_predict_016.err
#SBATCH --partition=short
#SBATCH --mem=16G
#SBATCH --time=04:00:00

PY=/home/cccbauer/.conda/envs/eeg_preproc/bin/python
cd /projects/swglab/data/DMNELF/analysis/fingerprint/eeg_bold_coupling/validation
$PY cross_subject_predict.py --test_subject dmnelf016 --n_shuffles 10000
SLURM
JOB_PRED=$(sbatch --parsable /tmp/cross_predict_016.sh)
echo "PREDICT_JOB=${JOB_PRED}"
PREDICT

echo ""
echo "=========================================="
echo "  All jobs submitted. Monitor with:"
echo "  ssh ${CLUSTER} 'squeue -u cccbauer'"
echo ""
echo "  After completion, pull results:"
echo "  rsync -avhP ${CLUSTER}:${PROJ}/results/validation/ ${PROJ_DIR}/results/validation/"
echo "=========================================="
