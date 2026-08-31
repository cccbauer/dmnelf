#!/bin/bash
# download_predictions.sh
# -----------------------
# Download all prediction .npz files from Explorer to local machine.
#
# Usage:
#   bash download_predictions.sh
#   bash download_predictions.sh --force  # overwrite local files

set -e

# ── Config ────────────────────────────────────────────────────────────────────
EXPLORER="cccbauer@explorer.northeastern.edu"
REMOTE_BASE="/projects/swglab/data/DMNELF/derivatives/cyclic_features"
LOCAL_DIR="$(cd "$(dirname "$0")" && pwd)/cyclic_features_local"

FORCE=false
for arg in "$@"; do
    case $arg in
        --force) FORCE=true ;;
    esac
done

echo "══════════════════════════════════════════════"
echo " Download Predictions from Explorer"
echo "══════════════════════════════════════════════"
echo ""
echo "Remote: $EXPLORER:$REMOTE_BASE"
echo "Local : $LOCAL_DIR"
echo ""

# ── Check if local dir exists ─────────────────────────────────────────────────
if [ -d "$LOCAL_DIR" ] && [ "$FORCE" != "true" ]; then
    echo "[WARNING] Local directory already exists: $LOCAL_DIR"
    echo "          Use --force to overwrite"
    exit 0
fi

# ── Create local directory ────────────────────────────────────────────────────
mkdir -p "$LOCAL_DIR"

# ── Download predictions ──────────────────────────────────────────────────────
echo "Downloading prediction files..."
rsync -avz --progress \
    --include="*/predictions/***" \
    --include="*/predictions/" \
    --include="sub-*/" \
    --exclude="*" \
    "$EXPLORER:$REMOTE_BASE/" \
    "$LOCAL_DIR/"

echo ""
echo "✓ Download complete"

# ── Count downloaded files ────────────────────────────────────────────────────
npz_count=$(find "$LOCAL_DIR" -name "*pda_prediction.npz" | wc -l)
echo ""
echo "Summary:"
echo "  Prediction files downloaded: $npz_count"
echo "  Directory: $LOCAL_DIR"
echo ""

# ── Next steps ────────────────────────────────────────────────────────────────
echo "Next steps:"
echo "  1. Update config.yaml to point to: $LOCAL_DIR"
echo "     features_dir: $LOCAL_DIR"
echo ""
echo "  2. Run evaluation:"
echo "     python evaluate_predictions.py --config config.yaml --plot"
echo ""
