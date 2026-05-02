#!/usr/bin/env python3
"""
run_pipeline.py
Run all pipeline steps in order, waiting for each SLURM job to finish
before launching the next.

Usage:
    python run_pipeline.py
    python run_pipeline.py --start 00b   # resume from a specific step
"""
import argparse
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

STEPS = [
    ("00",  "deploy_scripts/00_extract_difumo.py",          []),
    ("00b", "deploy_scripts/00b_extract_personal_masks.py", []),
    ("00c", "deploy_scripts/00c_add_personal_parcels.py",   []),
    ("00d", "deploy_scripts/00d_extract_personal_pda.py",   []),
    ("01",  "deploy_scripts/01_fit_microstates.py",         []),
    ("02",  "deploy_scripts/02_tess_features.py",           ["--overwrite"]),
    ("03",  "deploy_scripts/03_compute_pda.py",             ["--overwrite"]),
    ("04",  "deploy_scripts/04_train_decoder.py",           ["--overwrite"]),
    ("05",  "deploy_scripts/05_plot_microstate_pda_epochs.py", ["--overwrite"]),
    ("05b", "deploy_scripts/05b_stats_microstate_pda.py",      []),
    ("06",  "deploy_scripts/06_cen_pda_proxy.py",              []),
]

parser = argparse.ArgumentParser()
parser.add_argument("--start", default=None,
                    help="Step tag to start from (e.g. 00b, 01)")
args = parser.parse_args()

# Determine where to start
start_idx = 0
if args.start:
    tags = [s[0] for s in STEPS]
    if args.start not in tags:
        print(f"Unknown step '{args.start}'. Valid: {tags}")
        sys.exit(1)
    start_idx = tags.index(args.start)

steps_to_run = STEPS[start_idx:]
here = Path(__file__).parent

print("=" * 60)
print(f"Pipeline run  {datetime.now():%Y-%m-%d %H:%M}")
print(f"Steps: {[s[0] for s in steps_to_run]}")
print("=" * 60)

for tag, script, extra_args in steps_to_run:
    t0 = time.time()
    print(f"\n{'=' * 60}")
    print(f"[{tag}]  {script}  —  {datetime.now():%H:%M:%S}")
    print("=" * 60)

    env = os.environ.copy()
    env["PYTHONPATH"] = str(here) + os.pathsep + env.get("PYTHONPATH", "")
    cmd = [sys.executable, str(here / script)] + extra_args
    result = subprocess.run(cmd, cwd=str(here), env=env)

    elapsed = time.time() - t0
    if result.returncode != 0:
        print(f"\nERROR: {script} exited with code {result.returncode}. Stopping.")
        sys.exit(result.returncode)

    print(f"\n✓  [{tag}] done  ({elapsed/60:.1f} min)")

print(f"\n{'=' * 60}")
print(f"Pipeline complete  {datetime.now():%Y-%m-%d %H:%M}")
print("=" * 60)
