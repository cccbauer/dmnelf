#!/bin/bash
# evaluate_locally.sh
# -------------------
# Download predictions from Explorer and run evaluation locally.
#
# Usage:
#   bash evaluate_locally.sh
#   bash evaluate_locally.sh --plot       # include plots
#   bash evaluate_locally.sh --force      # re-download

set -e

# ── Config ────────────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
CONFIG_FILE="$SCRIPT_DIR/config.yaml"
LOCAL_FEATURES_DIR="$SCRIPT_DIR/cyclic_features_local"
EVAL_SCRIPT="$SCRIPT_DIR/evaluate_predictions.py"

PLOT=false
FORCE=false

for arg in "$@"; do
    case $arg in
        --plot)   PLOT=true ;;
        --force)  FORCE=true ;;
    esac
done

echo "══════════════════════════════════════════════"
echo " Evaluate Predictions Locally"
echo "══════════════════════════════════════════════"
echo ""

# ── Step 1: Download predictions ──────────────────────────────────────────────
echo "[1/3] Downloading predictions from Explorer..."
if [ "$FORCE" = "true" ]; then
    bash "$SCRIPT_DIR/download_predictions.sh" --force
else
    bash "$SCRIPT_DIR/download_predictions.sh"
fi

# ── Step 2: Create temporary config with local paths ──────────────────────────
echo ""
echo "[2/3] Preparing configuration..."

TEMP_CONFIG="/tmp/config_local_$$.yaml"
cp "$CONFIG_FILE" "$TEMP_CONFIG"

# Update features_dir to local path
sed -i '' "s|features_dir:.*|features_dir: $LOCAL_FEATURES_DIR|g" "$TEMP_CONFIG"

echo "  Config: $TEMP_CONFIG"
echo "  Features dir: $LOCAL_FEATURES_DIR"

# ── Step 3: Run evaluation ────────────────────────────────────────────────────
echo ""
echo "[3/3] Running evaluation..."

EVAL_ARGS="--config $TEMP_CONFIG"
if [ "$PLOT" = "true" ]; then
    EVAL_ARGS="$EVAL_ARGS --plot"
fi

python "$EVAL_SCRIPT" $EVAL_ARGS

# ── Cleanup ───────────────────────────────────────────────────────────────────
rm "$TEMP_CONFIG"

echo ""
echo "═════════════════════════════════════════════════════"
echo "✓ EVALUATION COMPLETE"
echo ""
echo "Results:"
echo "  CSV: $(find $SCRIPT_DIR -name 'evaluation_results.csv' -type f)"
if [ "$PLOT" = "true" ]; then
    echo "  Plots: $SCRIPT_DIR/evaluation_plots/"
fi
echo ""
