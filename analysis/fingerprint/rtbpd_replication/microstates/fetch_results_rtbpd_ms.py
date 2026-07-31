#!/usr/bin/env python3
"""
fetch_results_rtbpd_ms.py
Download rtBPD microstate pipeline outputs from the cluster to the local
results/ folder. Pattern-matched to microstate_pda/deploy_scripts/fetch_
results.py's scp_glob helper, simplified to this pipeline's single flat
CLUSTER_BASE/{microstates,results}/ layout (no per-step subdirectories).

Usage:
    python fetch_results_rtbpd_ms.py            # fetch everything
    python fetch_results_rtbpd_ms.py --dry-run  # show what would be downloaded
"""
import argparse
import subprocess
from pathlib import Path

from config_rtbpd_ms import CLUSTER_BASE, CLUSTER_SSH, RESULTS_DIR

# ── Cluster sources → local destinations ───────────────────────
SOURCES = [
    {
        "desc":   "Templates, assignments, channel list (step 01)",
        "remote": CLUSTER_BASE + "/microstates/*",
        "local":  RESULTS_DIR / "microstates",
    },
    {
        "desc":   "Temporal parameters CSV (step 02) + stats/figures (step 03)",
        "remote": CLUSTER_BASE + "/results/*",
        "local":  RESULTS_DIR,
    },
]


def scp_glob(remote_glob, local_dir, dry_run=False):
    local_dir.mkdir(parents=True, exist_ok=True)
    cmd = ["scp", "-q",
           f"{CLUSTER_SSH}:{remote_glob}",
           str(local_dir) + "/"]
    print(f"  scp {remote_glob}")
    print(f"   -> {local_dir}/")
    if dry_run:
        return
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        if "No such file" in result.stderr or "no matches" in result.stderr.lower():
            print("    (no files matched)")
        else:
            print(f"    WARNING: {result.stderr.strip()}")
    else:
        n = len(list(local_dir.glob("*")))
        print(f"    OK ({n} files in dest)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true",
                        help="Print commands without executing")
    args = parser.parse_args()

    print()
    print("=" * 60)
    print(f"Fetching rtBPD microstate results from {CLUSTER_SSH}")
    print("=" * 60)

    for item in SOURCES:
        print(f"\n{item['desc']}")
        scp_glob(item["remote"], item["local"], dry_run=args.dry_run)

    if not args.dry_run:
        print()
        print("Results saved to: " + str(RESULTS_DIR))
        print()
        print("To add to git:")
        print("  git add rtbpd_replication/microstates/results/")
        print("  git commit -m 'rtBPD microstate pre/post results'")
        print("  git push")


if __name__ == "__main__":
    main()
