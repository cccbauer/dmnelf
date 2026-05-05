#!/bin/bash
# full_evaluation_workflow.sh
# ---------------------------
# Complete workflow: run evaluation on Explorer, download, and summarize
#
# Usage:
#   bash full_evaluation_workflow.sh           # evaluate + summarize
#   bash full_evaluation_workflow.sh --plot    # + generate plots

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PLOT=false

for arg in "$@"; do
    case $arg in
        --plot) PLOT=true ;;
    esac
done

echo "════════════════════════════════════════════════"
echo " Full Evaluation Workflow"
echo "════════════════════════════════════════════════"
echo ""

# Step 1: Run evaluation on Explorer
echo "[1/3] Running evaluation on Explorer..."
if [ "$PLOT" = "true" ]; then
    bash "$SCRIPT_DIR/evaluate_on_explorer.sh" --plot --download
else
    bash "$SCRIPT_DIR/evaluate_on_explorer.sh" --download
fi

# Step 2: Summarize results
echo ""
echo "[2/3] Generating summary report..."
cd "$SCRIPT_DIR"
python summarize_results.py --csv results/evaluation_results.csv --visualize --output-dir results

# Step 3: Display summary
echo ""
echo "[3/3] Summary complete!"
echo ""
echo "Results saved to: $SCRIPT_DIR/results/"
echo ""
echo "Files generated:"
echo "  • results/evaluation_results.csv          (per-subject metrics)"
echo "  • results/summary_correlations.png        (bar chart of performance)"
if [ "$PLOT" = "true" ]; then
    echo "  • results/evaluation_plots/               (detailed evaluation plots)"
fi
echo ""
echo "Open results:"
open results 2>/dev/null || echo "  results/"
echo ""
