#!/usr/bin/env python3
"""
check_fmri_status.py
Check fMRI preprocessing jobs, outputs, and errors from local machine
"""

import sys
from pathlib import Path
import subprocess

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from utils import run_ssh
from config import CLUSTER_SSH

CLUSTER_BASE = "/projects/swglab/data/DMNELF/analysis/MNE/jupyter/microstate_pda_v3"
FMRI_DIR = "/projects/swglab/data/DMNELF/analysis/fmri_preprocessing"
LOGS_DIR = f"{FMRI_DIR}/logs"
DERIVATIVES = "/projects/swglab/data/DMNELF/derivatives"

def check_slurm_queue():
    """Check active SLURM jobs"""
    print("\n📊 Current SLURM Jobs:")
    print("=" * 60)
    
    cmd = "squeue -u cccbauer --format='%.18i %.9P %.8j %.8u %.2t %.10M %.6D %R' | grep -E 'fmri|eeg' || echo '(No active fMRI/EEG jobs)'"
    result = run_ssh(cmd)
    
    if hasattr(result, 'stdout'):
        print(result.stdout)
    else:
        print(result)

def check_logs():
    """Check job logs"""
    print("\n📁 Recent Job Logs:")
    print("=" * 60)
    
    # List recent logs
    cmd = f"ls -lht {LOGS_DIR}/fmri_pipeline*.out 2>/dev/null | head -5 || echo '(No fMRI logs yet)'"
    result = run_ssh(cmd)
    
    if hasattr(result, 'stdout'):
        print(result.stdout)
    else:
        print(result)
    
    # Show latest log (last 30 lines)
    print("\n🔍 Latest Log (last 30 lines):")
    print("-" * 60)
    cmd = f"latest=$(ls -t {LOGS_DIR}/fmri_pipeline*.out 2>/dev/null | head -1); [ -n \"$latest\" ] && tail -30 \"$latest\" || echo '(No logs)'"
    result = run_ssh(cmd)
    
    if hasattr(result, 'stdout'):
        print(result.stdout)
    else:
        print(result)

def check_outputs():
    """Check output files"""
    print("\n📦 Output Files:")
    print("=" * 60)
    
    subjects = ["sub-dmnelf012", "sub-dmnelf013"]
    
    for subject in subjects:
        print(f"\n  {subject}:")
        
        # Check derivatives
        for dir_name in ["eeg_preprocessed", "fmri_microstates", "pda_features"]:
            cmd = f"[ -d {DERIVATIVES}/{dir_name}/{subject} ] && find {DERIVATIVES}/{dir_name}/{subject} -type f | wc -l && du -sh {DERIVATIVES}/{dir_name}/{subject} | cut -f1 || echo 'N/A'"
            result = run_ssh(cmd)
            
            if hasattr(result, 'stdout'):
                output = result.stdout.strip().split('\n')
                if len(output) >= 2 and output[0] != 'N/A':
                    count = output[0]
                    size = output[1] if len(output) > 1 else "?"
                    print(f"    • {dir_name}: {count} files ({size})")
            else:
                print(f"    • {dir_name}: (unable to check)")

def check_errors():
    """Check for errors in logs"""
    print("\n⚠️  Error Summary:")
    print("=" * 60)
    
    cmd = f"grep -r 'ERROR\\|FAILED\\|Traceback' {LOGS_DIR}/fmri_pipeline*.out 2>/dev/null | wc -l"
    result = run_ssh(cmd)
    
    if hasattr(result, 'stdout'):
        error_count = result.stdout.strip()
    else:
        error_count = "?"
    
    if error_count != "0" and error_count != "?":
        print(f"Found {error_count} error lines:")
        cmd = f"grep -r 'ERROR\\|FAILED\\|Traceback' {LOGS_DIR}/fmri_pipeline*.out 2>/dev/null | head -10"
        result = run_ssh(cmd)
        
        if hasattr(result, 'stdout'):
            print(result.stdout)
        else:
            print(result)
    else:
        print("✓ No errors found in logs (or logs not yet created)")

def main():
    print("=" * 60)
    print("fMRI Preprocessing Pipeline Status Check")
    print("=" * 60)
    
    try:
        check_slurm_queue()
        check_logs()
        check_outputs()
        check_errors()
        
        print("\n" + "=" * 60)
        print("Status Check Complete")
        print("=" * 60)
        print("\nTips:")
        print("  • Watch logs in real-time:")
        print("    ssh cccbauer@explorer.northeastern.edu 'tail -f " + LOGS_DIR + "/fmri_pipeline_*.out'")
        print("\n  • Check specific subject:")
        print("    ssh cccbauer@explorer.northeastern.edu 'grep -A 50 sub-dmnelf012 " + LOGS_DIR + "/fmri_pipeline_*.out'")
        print("\n  • Run this script again in 5 minutes to see progress")
        print("")
        
    except Exception as e:
        print(f"\n❌ Error checking status: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
