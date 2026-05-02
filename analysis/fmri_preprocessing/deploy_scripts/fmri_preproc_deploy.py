#!/usr/bin/env python3
"""
fmri_preproc_deploy.py
Deploy and run fMRI preprocessing pipeline on cluster (DiFuMo, microstate, TESS, PDA)

Orchestrator script that calls individual deployment scripts to generate cluster scripts,
then submits a single SBATCH job to run them in sequence.

Run locally from your machine:
    python fmri_preproc_deploy.py                          # all subjects
    python fmri_preproc_deploy.py --subject sub-dmnelf012
    python fmri_preproc_deploy.py --all --overwrite
"""

import argparse
import subprocess
import time
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils import run_ssh, scp_to
from config import CLUSTER_BASE, SLURM_ACCOUNT, SUBJECTS

CLUSTER_PYTHON = "/home/cccbauer/.conda/envs/eeg_preproc/bin/python"
DEPLOY_SCRIPTS_DIR = Path(__file__).parent
FMRI_DIR = "/projects/swglab/data/DMNELF/analysis/fmri_preprocessing"
FMRI_LOG_DIR = "/projects/swglab/data/DMNELF/analysis/fmri_preprocessing/logs"

def call_deployment_script(script_name, *args):
    """Call a deployment script locally"""
    script_path = DEPLOY_SCRIPTS_DIR / script_name
    cmd = [sys.executable, str(script_path)] + list(args)
    print(f"\n→ Calling {script_name} {' '.join(args)}")
    result = subprocess.run(cmd, capture_output=False)
    if result.returncode != 0:
        print(f"ERROR: {script_name} failed with exit code {result.returncode}")
        return False
    return True

def submit_fmri_pipeline(subjects=None, overwrite=False):
    """
    Deploy all fMRI pipeline scripts and submit orchestrating SBATCH job
    
    Parameters:
    -----------
    subjects : list or None
        List of subject IDs to process. If None, processes all SUBJECTS in config.
    overwrite : bool
        If True, reprocess even if outputs exist.
    """
    
    if subjects is None:
        subjects = SUBJECTS
    elif isinstance(subjects, str):
        subjects = [subjects]
    
    print("\n" + "=" * 60)
    print("fMRI Preprocessing Pipeline Deployment")
    print("=" * 60)
    print(f"Subjects: {', '.join(subjects)}")
    print(f"Overwrite: {overwrite}")
    print("=" * 60)
    
    # ── Step 1: Call deployment scripts to generate cluster scripts ────────
    print("\n[1/5] Deploying pipeline scripts...")
    
    # Each script generates and deploys its cluster version
    call_deployment_script("00_extract_difumo.py")
    call_deployment_script("01_fit_microstates.py")
    call_deployment_script("02_tess_features.py")
    call_deployment_script("03_compute_pda.py")
    
    # ── Step 2: Create output directories on cluster ─────────────────────
    print("\n[2/5] Creating cluster output directories...")
    run_ssh("mkdir -p /projects/swglab/data/DMNELF/analysis/fmri_preprocessing/scripts")
    run_ssh("mkdir -p " + FMRI_LOG_DIR)
    run_ssh("mkdir -p /projects/swglab/data/DMNELF/derivatives/fmri_microstates")
    run_ssh("mkdir -p /projects/swglab/data/DMNELF/derivatives/pda_features")
    
    # ── Step 3: Build orchestrating SBATCH script ──────────────────────────
    print("\n[3/5] Building orchestrating SBATCH job...")
    
    sbatch_lines = [
        "#!/bin/bash",
        "#SBATCH --job-name=fmri_pipeline_orchestrate",
        "#SBATCH --output=" + FMRI_LOG_DIR + "/orchestrate_%j.out",
        "#SBATCH --error=" + FMRI_LOG_DIR + "/orchestrate_%j.err",
        "#SBATCH --partition=short",
        "#SBATCH --time=48:00:00",
        "#SBATCH --cpus-per-task=4",
        "#SBATCH --mem=64G",
        "#SBATCH --account=" + SLURM_ACCOUNT,
        "",
        "set -e",
        "PYTHON=" + CLUSTER_PYTHON,
        "SCRIPTS_DIR=/projects/swglab/data/DMNELF/analysis/fmri_preprocessing/scripts",
        "",
        "echo '========================================'",
        "echo 'fMRI Preprocessing Pipeline Orchestrator'",
        "echo '========================================'",
        "echo \"Start: $(date)\"",
        "echo \"Job: $SLURM_JOB_ID\"",
        "echo ''",
        "",
        "# Run pipeline scripts in sequence",
        "echo '[0/4] Extracting DiFuMo-64 timeseries...'",
        "$PYTHON $SCRIPTS_DIR/00_extract_difumo_cluster.py",
        "",
        "echo '[1/4] Fitting microstate maps...'",
        "$PYTHON $SCRIPTS_DIR/01_fit_microstates_250_cluster.py",
        "",
        "echo '[2/4] Computing TESS features...'",
        "$PYTHON $SCRIPTS_DIR/02_tess_features_cluster.py",
        "",
        "echo '[3/4] Computing PDA...'",
        "$PYTHON $SCRIPTS_DIR/03_compute_pda_cluster.py",
        "",
        "echo ''",
        "echo '========================================'",
        "echo '✓ fMRI preprocessing complete!'",
        "echo '========================================'",
        "echo \"End: $(date)\"",
    ]
    
    sbatch_script = "\n".join(sbatch_lines)
    
    # Save SBATCH script to temp file
    import tempfile
    with tempfile.NamedTemporaryFile(mode='w', suffix='.sh', delete=False) as f:
        f.write(sbatch_script)
        local_sbatch_path = f.name
    
    # ── Step 4: Upload orchestrating SBATCH script ────────────────────────
    print("[4/5] Uploading orchestrating script...")
    remote_sbatch = "/projects/swglab/data/DMNELF/analysis/fmri_preprocessing/fmri_orchestrate_job.sh"
    scp_to(local_sbatch_path, remote_sbatch)
    run_ssh(f"chmod +x {remote_sbatch}")
    
    # ── Step 5: Submit SBATCH job ──────────────────────────────────────────
    print("[5/5] Submitting SBATCH job...")
    result = run_ssh(f"sbatch {remote_sbatch}")
    
    # Extract stdout if result is CompletedProcess
    if hasattr(result, 'stdout'):
        result_str = result.stdout.decode() if isinstance(result.stdout, bytes) else result.stdout
    else:
        result_str = str(result)
    
    print(result_str)
    
    # Extract job ID
    if "Submitted batch job" in result_str:
        job_id = result_str.split("Submitted batch job ")[-1].strip()
        print("\n" + "=" * 60)
        print(f"✓ Orchestrating job submitted successfully!")
        print(f"  Job ID: {job_id}")
        print(f"  Script: {remote_sbatch}")
        print("=" * 60)
        print("\nMonitor with:")
        print(f"  squeue -j {job_id}")
        print(f"  ssh cccbauer@explorer.northeastern.edu 'tail -f {FMRI_LOG_DIR}/orchestrate_{job_id}.out'")
        print(f"\nPipeline steps will be submitted as child jobs:")
        print(f"  - 00_extract_difumo_cluster.py")
        print(f"  - 01_fit_microstates_250_cluster.py")
        print(f"  - 02_tess_features_cluster.py")
        print(f"  - 03_compute_pda_cluster.py")
    else:
        print("WARNING: Could not parse job ID from response")
        print("Response:", result_str)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Deploy fMRI preprocessing pipeline to cluster"
    )
    parser.add_argument(
        "--subject",
        type=str,
        default=None,
        help="Process single subject (e.g., sub-dmnelf012)"
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Process all configured subjects"
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing outputs"
    )
    
    args = parser.parse_args()
    
    subjects = None
    if args.subject:
        subjects = [args.subject]
    elif args.all:
        subjects = SUBJECTS
    else:
        subjects = SUBJECTS
    
    submit_fmri_pipeline(subjects=subjects, overwrite=args.overwrite)
