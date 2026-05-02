#!/usr/bin/env python3
"""
fetch_results.py
Download all pipeline outputs from the cluster to the local results/ folder.
Keeps results in git alongside the code.

Usage:
    python fetch_results.py            # fetch everything
    python fetch_results.py --step 05  # fetch only step 05 plots
    python fetch_results.py --step 06  # fetch only step 06 stats
    python fetch_results.py --dry-run  # show what would be downloaded
"""
import argparse
import subprocess
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils import run_ssh
from config import CLUSTER_BASE, CLUSTER_SSH

HERE    = Path(__file__).parent.parent   # repo root
RESULTS = HERE / "results"

# ── Cluster sources → local destinations ───────────────────────
SOURCES = {
    "05": {
        "desc": "Microstate PDA epoch plots (step 05)",
        "items": [
            {
                # Per-subject plots
                "remote": "/projects/swglab/data/DMNELF/derivatives/microstate_pda_plots/*/*.png",
                "local":  RESULTS / "microstate_pda_plots" / "individual",
                "flat":   True,   # scp -r flattens to one dir
            },
            {
                # Group plot
                "remote": "/projects/swglab/data/DMNELF/derivatives/microstate_pda_plots/group_*.png",
                "local":  RESULTS / "microstate_pda_plots",
                "flat":   False,
            },
        ],
    },
    "06": {
        "desc": "Microstate PDA statistics (step 06)",
        "items": [
            {
                "remote": CLUSTER_BASE + "/stats/*.csv",
                "local":  RESULTS / "stats",
                "flat":   False,
            },
            {
                "remote": CLUSTER_BASE + "/stats/*.png",
                "local":  RESULTS / "stats",
                "flat":   False,
            },
        ],
    },
    "08": {
        "desc": "Personalized decoder: rest→feedback (step 08)",
        "items": [
            {
                "remote": CLUSTER_BASE + "/results/personalized_decoder/*.csv",
                "local":  RESULTS / "personalized_decoder",
                "flat":   False,
            },
            {
                "remote": CLUSTER_BASE + "/results/personalized_decoder/*.png",
                "local":  RESULTS / "personalized_decoder",
                "flat":   False,
            },
            {
                "remote": CLUSTER_BASE + "/results/personalized_decoder/timeseries/*.png",
                "local":  RESULTS / "personalized_decoder" / "timeseries",
                "flat":   False,
            },
        ],
    },
    "07": {
        "desc": "Decomposed network proxy + transfer CV: rest→feedback (step 07)",
        "items": [
            {
                "remote": CLUSTER_BASE + "/results/network_proxy/*.csv",
                "local":  RESULTS / "network_proxy",
                "flat":   False,
            },
            {
                "remote": CLUSTER_BASE + "/results/network_proxy/*.png",
                "local":  RESULTS / "network_proxy",
                "flat":   False,
            },
        ],
    },
}


def scp_glob(remote_glob, local_dir, dry_run=False):
    local_dir.mkdir(parents=True, exist_ok=True)
    cmd = ["scp", "-q",
           f"{CLUSTER_SSH}:{remote_glob}",
           str(local_dir) + "/"]
    print(f"  scp {remote_glob}")
    print(f"   → {local_dir}/")
    if dry_run:
        return
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        # scp returns non-zero if no files match glob — check stderr
        if "No such file" in result.stderr or "no matches" in result.stderr.lower():
            print(f"    (no files matched)")
        else:
            print(f"    WARNING: {result.stderr.strip()}")
    else:
        n = len(list(local_dir.glob("*.png")) + list(local_dir.glob("*.csv")))
        print(f"    OK ({n} files in dest)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--step", choices=["05", "06", "07", "08"], default=None,
                        help="Fetch only this step (default: all)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print commands without executing")
    args = parser.parse_args()

    steps = [args.step] if args.step else list(SOURCES.keys())

    print()
    print("=" * 60)
    print(f"Fetching results from {CLUSTER_SSH}")
    print("=" * 60)

    for step in steps:
        info = SOURCES[step]
        print(f"\nStep {step}: {info['desc']}")
        for item in info["items"]:
            scp_glob(item["remote"], item["local"], dry_run=args.dry_run)

    if not args.dry_run:
        print()
        print("Results saved to: results/")
        print()
        print("To add to git:")
        print("  git add results/")
        print("  git commit -m 'update results'")
        print("  git push")


if __name__ == "__main__":
    main()
