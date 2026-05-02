#!/bin/bash
# monitor_fmri_pipeline.sh
# Check fMRI preprocessing jobs, outputs, and errors on cluster
# Run this on the cluster or locally: bash monitor_fmri_pipeline.sh

FMRI_DIR="/projects/swglab/data/DMNELF/analysis/fmri_preprocessing"
CLUSTER_BASE="/projects/swglab/data/DMNELF/analysis/MNE/jupyter/microstate_pda_v3"
LOGS_DIR="$CLUSTER_BASE/logs"
DERIVATIVES="/projects/swglab/data/DMNELF/derivatives"

echo "============================================================"
echo "fMRI Preprocessing Pipeline Monitor"
echo "============================================================"
echo ""

# Check SBATCH script
echo "📋 SBATCH Script:"
if [ -f "$FMRI_DIR/fmri_pipeline_job.sh" ]; then
    echo "  ✓ $FMRI_DIR/fmri_pipeline_job.sh"
    ls -lh "$FMRI_DIR/fmri_pipeline_job.sh"
else
    echo "  ✗ No SBATCH script found"
fi
echo ""

# Check recent job queue
echo "📊 Current SLURM Jobs:"
squeue -u cccbauer --format="%.18i %.9P %.8j %.8u %.2t %.10M %.6D %R" 2>/dev/null | grep -E "fmri|eeg" || echo "  (No active fMRI/EEG jobs)"
echo ""

# Check job logs
echo "📁 Recent Job Logs:"
if [ -d "$LOGS_DIR" ]; then
    echo "  Directory: $LOGS_DIR"
    echo ""
    # Show recent logs
    ls -lht "$LOGS_DIR"/fmri_pipeline*.out 2>/dev/null | head -5 || echo "  (No fMRI logs yet)"
    echo ""
    
    # Show last log
    latest_log=$(ls -t "$LOGS_DIR"/fmri_pipeline*.out 2>/dev/null | head -1)
    if [ -n "$latest_log" ]; then
        echo "🔍 Latest Log (last 30 lines):"
        echo "  File: $latest_log"
        echo "  ---"
        tail -30 "$latest_log"
    fi
else
    echo "  ✗ Logs directory not found"
fi
echo ""

# Check output directories
echo "📦 Output Directories:"
for subject in sub-dmnelf012 sub-dmnelf013; do
    echo ""
    echo "  $subject:"
    
    # Check if derivatives exist
    if [ -d "$DERIVATIVES" ]; then
        for dir in eeg_preprocessed fmri_microstates pda_features; do
            if [ -d "$DERIVATIVES/$dir/$subject" ]; then
                count=$(find "$DERIVATIVES/$dir/$subject" -type f | wc -l)
                size=$(du -sh "$DERIVATIVES/$dir/$subject" 2>/dev/null | cut -f1)
                echo "    ✓ $dir: $count files ($size)"
            fi
        done
    fi
done
echo ""

# Check for errors
echo "⚠️  Error Summary:"
if [ -d "$LOGS_DIR" ]; then
    error_count=$(grep -r "ERROR\|FAILED\|Traceback" "$LOGS_DIR"/fmri_pipeline*.out 2>/dev/null | wc -l)
    if [ "$error_count" -gt 0 ]; then
        echo "  Found $error_count error lines:"
        echo ""
        grep -r "ERROR\|FAILED\|Traceback" "$LOGS_DIR"/fmri_pipeline*.out 2>/dev/null | head -10
    else
        echo "  ✓ No errors found in logs"
    fi
else
    echo "  (Logs not accessible)"
fi
echo ""

echo "============================================================"
echo "Monitoring Complete"
echo "============================================================"
echo ""
echo "View full latest log:"
echo "  tail -f $LOGS_DIR/fmri_pipeline_*.out"
echo ""
echo "View specific subject log:"
echo "  grep -A 50 'sub-dmnelf012' $LOGS_DIR/fmri_pipeline_*.out"
echo ""
