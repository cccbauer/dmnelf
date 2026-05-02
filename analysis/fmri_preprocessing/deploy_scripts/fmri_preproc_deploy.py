#!/usr/bin/env python3
"""
fmri_preproc_deploy.py
Deploy and run fMRI preprocessing pipeline on cluster (DiFuMo, microstate, PDA, etc.)

Run locally from your machine to deploy to cluster and submit SLURM jobs.

Usage:
    python fmri_preproc_deploy.py                          # all subjects
    python fmri_preproc_deploy.py --subject sub-dmnelf012
    python fmri_preproc_deploy.py --all --overwrite
"""

import argparse
import time
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils import run_ssh, scp_to
from config import CLUSTER_BASE, SLURM_ACCOUNT, LOCAL_BASE, SUBJECTS

CLUSTER_PYTHON = "/home/cccbauer/.conda/envs/eeg_preproc/bin/python"
CLUSTER_SCRIPTS = CLUSTER_BASE + "/deploy_scripts"

def submit_fmri_pipeline(subjects=None, overwrite=False):
    """
    Submit fMRI preprocessing pipeline job to cluster
    
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
    print(f"Cluster: {CLUSTER_BASE}")
    print("=" * 60 + "\n")
    
    # Build command with arguments
    cmd = CLUSTER_PYTHON + " " + CLUSTER_SCRIPTS + "/01_fit_microstates.py"
    
    # Add subject arguments
    for subject in subjects:
        cmd += f" --subject {subject}"
    
    if overwrite:
        cmd += " --overwrite"
    
    # Build SBATCH script
    sbatch_lines = [
        "#!/bin/bash",
        "#SBATCH --job-name=fmri_preproc_pipeline",
        "#SBATCH --output=" + CLUSTER_BASE + "/logs/fmri_pipeline_%j.out",
        "#SBATCH --error=" + CLUSTER_BASE + "/logs/fmri_pipeline_%j.err",
        "#SBATCH --partition=short",
        "#SBATCH --time=24:00:00",
        "#SBATCH --cpus-per-task=4",
        "#SBATCH --mem=64G",
        "#SBATCH --account=" + SLURM_ACCOUNT,
        "",
        "# fMRI Preprocessing Pipeline",
        "# Steps: DiFuMo → Microstates → TESS → PDA",
        "",
        "set -e",
        "PYTHON=" + CLUSTER_PYTHON,
        "CLUSTER_BASE=" + CLUSTER_BASE,
        "",
        "mkdir -p $CLUSTER_BASE/logs",
        "cd $CLUSTER_BASE",
        "",
        "echo '========================================'",
        "echo 'fMRI Preprocessing Pipeline'",
        "echo '========================================'",
        "echo \"Start: $(date)\"",
        "echo \"Job: $SLURM_JOB_ID\"",
        "echo ''",
        "",
        "# Run pipeline steps in sequence",
        "for subject in " + " ".join(subjects) + "; do",
        "  echo \"Processing $subject...\"",
        "  ",
        "  # Step 0: DiFuMo extraction",
        "  echo \"  [0] Extracting DiFuMo-64 timeseries...\"",
        "  $PYTHON deploy_scripts/00_extract_difumo.py --subject $subject --all" + (" --overwrite" if overwrite else ""),
        "  ",
        "  # Step 1: Fit microstates",
        "  echo \"  [1] Fitting microstate maps...\"",
        "  $PYTHON deploy_scripts/01_fit_microstates.py --subject $subject --all" + (" --overwrite" if overwrite else ""),
        "  ",
        "  # Step 2: TESS features",
        "  echo \"  [2] Computing TESS features...\"",
        "  $PYTHON deploy_scripts/02_tess_features.py --subject $subject --all" + (" --overwrite" if overwrite else ""),
        "  ",
        "  # Step 3: PDA",
        "  echo \"  [3] Computing PDA...\"",
        "  $PYTHON deploy_scripts/03_compute_pda.py --subject $subject --all" + (" --overwrite" if overwrite else ""),
        "  ",
        "  echo \"  ✓ $subject complete\"",
        "done",
        "",
        "echo ''",
        "echo '========================================'",
        "echo '✓ fMRI preprocessing complete!'",
        "echo '========================================'",
        "echo \"End: $(date)\"",
    ]
    
    sbatch_script = "\n".join(sbatch_lines)
    
    # Save SBATCH script locally (to temp location)
    import tempfile
    with tempfile.NamedTemporaryFile(mode='w', suffix='.sh', delete=False) as f:
        f.write(sbatch_script)
        local_temp_path = f.name
    
    print(f"✓ Generated SBATCH script: {local_temp_path}\n")
    
    # Upload to cluster and submit
    print("Uploading to cluster...")
    remote_sbatch = "/projects/swglab/data/DMNELF/analysis/fmri_preprocessing/fmri_pipeline_job.sh"
    
    # Make sure deploy_scripts directory exists on cluster
    print("Creating cluster directories...")
    run_ssh("mkdir -p " + CLUSTER_BASE + "/logs")
    run_ssh("mkdir -p " + CLUSTER_BASE + "/deploy_scripts")
    run_ssh("mkdir -p /projects/swglab/data/DMNELF/analysis/fmri_preprocessing")
    
    print("Uploading SBATCH script...")
    scp_to(local_temp_path, remote_sbatch)
    
    # Make executable and submit
    print("Making script executable...")
    run_ssh(f"chmod +x {remote_sbatch}")
    
    print("Submitting to SLURM...")
    result = run_ssh(f"sbatch {remote_sbatch}")
    
    # Extract stdout if result is CompletedProcess
    if hasattr(result, 'stdout'):
        result_str = result.stdout
    else:
        result_str = str(result)
    
    print(result_str)
    
    # Extract job ID
    if "Submitted batch job" in result_str:
        job_id = result_str.split("Submitted batch job ")[-1].strip()
        print("\n" + "=" * 60)
        print(f"✓ Job submitted successfully!")
        print(f"  Job ID: {job_id}")
        print(f"  Script: {remote_sbatch}")
        print("=" * 60)
        print("\nMonitor with:")
        print(f"  squeue -j {job_id}")
        print(f"  ssh cccbauer@explorer.northeastern.edu 'tail -f {CLUSTER_BASE}/logs/fmri_pipeline_{job_id}.out'")
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
