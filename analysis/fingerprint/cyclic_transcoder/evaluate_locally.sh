#!/bin/bash
# evaluate_locally.sh
# -------------------
# Download predictions from Explorer and run evaluation locally.
#
# Usage:
#   bash evaluate_locally.sh
#   bash evaluate_locally.sh --plot       # include plots
#   bash evaluate_locally.sh --force      # re-download
#   bash evaluate_locally.sh --result-tag smooth_w11

set -e

# ── Config ────────────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
CONFIG_FILE="$SCRIPT_DIR/config.yaml"
LOCAL_FEATURES_DIR="$SCRIPT_DIR/cyclic_features_local"
EVAL_SCRIPT="$SCRIPT_DIR/evaluate_predictions.py"

PLOT=false
FORCE=false
SMOOTH_WINDOW=1
SMOOTH_BOTH=false
RESULT_TAG=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --plot)
            PLOT=true
            shift
            ;;
        --force)
            FORCE=true
            shift
            ;;
        --smooth-window)
            SMOOTH_WINDOW="$2"
            shift 2
            ;;
        --smooth-both)
            SMOOTH_BOTH=true
            shift
            ;;
        --result-tag)
            RESULT_TAG="$2"
            shift 2
            ;;
        *)
            echo "Unknown argument: $1"
            exit 1
            ;;
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
if [ "$SMOOTH_WINDOW" != "1" ]; then
    EVAL_ARGS="$EVAL_ARGS --smooth-window $SMOOTH_WINDOW"
fi
if [ "$SMOOTH_BOTH" = "true" ]; then
    EVAL_ARGS="$EVAL_ARGS --smooth-both"
fi
if [ -n "$RESULT_TAG" ]; then
    EVAL_ARGS="$EVAL_ARGS --result-tag $RESULT_TAG"
fi

python "$EVAL_SCRIPT" $EVAL_ARGS

# ── Cleanup ───────────────────────────────────────────────────────────────────
rm "$TEMP_CONFIG"

echo ""
echo "═════════════════════════════════════════════════════"
echo "✓ EVALUATION COMPLETE"
echo ""
echo "Results:"
if [ -n "$RESULT_TAG" ]; then
    echo "  CSV: $SCRIPT_DIR/evaluation_results_${RESULT_TAG}.csv"
else
    echo "  CSV: $SCRIPT_DIR/evaluation_results.csv"
fi
if [ "$PLOT" = "true" ]; then
    if [ -n "$RESULT_TAG" ]; then
        echo "  Plots: $SCRIPT_DIR/evaluation_plots_${RESULT_TAG}/"
    else
        echo "  Plots: $SCRIPT_DIR/evaluation_plots/"
    fi
fi
echo ""
