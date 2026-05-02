# eeg_preproc_deploy.py
# Run locally: python eeg_preproc_deploy.py
# Deploys eeg_preproc.py to cluster and submits two parallel SLURM jobs
# (250Hz and 500Hz) simultaneously.
#
# Usage:
#   python eeg_preproc_deploy.py                    # all subjects, 250+500Hz
#   python eeg_preproc_deploy.py --subject sub-dmnelf009
#   python eeg_preproc_deploy.py --overwrite

import argparse
import time
from pathlib import Path
from utils import run_ssh, scp_to
from config import CLUSTER_BASE, SLURM_ACCOUNT, LOCAL_BASE

CLUSTER_PYTHON = "/home/cccbauer/.conda/envs/eeg_preproc/bin/python"
CLUSTER_SCRIPT = CLUSTER_BASE + "/scripts/eeg_preproc.py"

def submit_job(cmd, job_name, sfreq):
    log_out = CLUSTER_BASE + "/logs/" + job_name + "_" + sfreq + "Hz_%j.out"
    log_err = CLUSTER_BASE + "/logs/" + job_name + "_" + sfreq + "Hz_%j.err"

    sbatch_lines = [
        "#!/bin/bash",
        "#SBATCH --job-name=" + job_name + "_" + sfreq + "Hz",
        "#SBATCH --output=" + log_out,
        "#SBATCH --error="  + log_err,
        "#SBATCH --partition=short",
        "#SBATCH --time=08:00:00",
        "#SBATCH --cpus-per-task=4",
        "#SBATCH --mem=32G",
        "#SBATCH --account=" + SLURM_ACCOUNT,
        "",
        cmd + " --sfreq " + sfreq,
    ]

    sbatch_name = "eeg_preproc_" + sfreq + "Hz.sh"
    sbatch_path = LOCAL_BASE / "scripts" / sbatch_name
    with open(sbatch_path, "w") as f:
        f.write("\n".join(sbatch_lines))

    scp_to(sbatch_path,
           CLUSTER_BASE + "/scripts/" + sbatch_name,
           verbose=False)
    print("Deployed: " + sbatch_name)

    result = run_ssh("sbatch " + CLUSTER_BASE + "/scripts/" + sbatch_name)
    job_id = ""
    for line in result.stdout.strip().split("\n"):
        if "Submitted" in line:
            job_id = line.strip().split()[-1]
            print(sfreq + "Hz Job ID: " + job_id)
    return job_id


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject",   type=str, default=None)
    parser.add_argument("--task",      type=str, default=None)
    parser.add_argument("--run",       type=str, default=None)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    # ── 1. Deploy eeg_preproc.py to cluster ────────────────
    local_script = Path(__file__).parent / "eeg_preproc.py"
    if not local_script.exists():
        print("ERROR: eeg_preproc.py not found at " + str(local_script))
        return

    print("Deploying eeg_preproc.py to cluster...")
    scp_to(local_script, CLUSTER_SCRIPT, verbose=False)
    print("Deployed: " + CLUSTER_SCRIPT)

    # ── 2. Build base run command ───────────────────────────
    cmd = CLUSTER_PYTHON + " " + CLUSTER_SCRIPT

    if args.subject and args.task and args.run:
        cmd      += " --subject " + args.subject
        cmd      += " --task "    + args.task
        cmd      += " --run "     + args.run
        job_name  = "eeg_" + args.subject.replace("sub-", "") + "_" + args.task + "_" + args.run
    elif args.subject:
        cmd      += " --subject " + args.subject
        job_name  = "eeg_" + args.subject.replace("sub-", "")
    else:
        cmd      += " --all"
        job_name  = "eeg_preproc_all"

    if args.overwrite:
        cmd += " --overwrite"

    print("Base command: " + cmd)

    # ── 3. Submit two parallel jobs ────────────────────────
    print("\nSubmitting parallel jobs (250Hz + 500Hz)...")
    job_ids = {}
    for sfreq in ["250", "500"]:
        job_id = submit_job(cmd, job_name, sfreq)
        if job_id:
            job_ids[sfreq] = job_id

    # ── 4. Monitor both jobs ───────────────────────────────
    if job_ids:
        print("\nMonitoring jobs: " + str(job_ids) + "  (Ctrl+C to stop)")
        print("-" * 55)
        try:
            pending = dict(job_ids)
            while pending:
                for sfreq, job_id in list(pending.items()):
                    r = run_ssh(
                        "squeue -j " + job_id
                        + " --format=%.8i_%.8T_%.10M 2>/dev/null",
                        verbose=False
                    )
                    status = r.stdout.strip()
                    if status and "JOBID" not in status.split("\n")[-1]:
                        print(sfreq + "Hz: " + status)
                    else:
                        print(sfreq + "Hz job " + job_id + " finished")
                        log = run_ssh(
                            "tail -10 " + CLUSTER_BASE
                            + "/logs/" + job_name + "_" + sfreq
                            + "Hz_" + job_id + ".out 2>/dev/null",
                            verbose=False
                        )
                        print(log.stdout)
                        del pending[sfreq]
                if pending:
                    time.sleep(20)
        except KeyboardInterrupt:
            print("\nStopped watching. Check manually:")
            for sfreq, job_id in job_ids.items():
                print("  " + sfreq + "Hz: squeue -j " + job_id)
                print("  tail -f " + CLUSTER_BASE
                      + "/logs/" + job_name + "_" + sfreq
                      + "Hz_" + job_id + ".out")

if __name__ == "__main__":
    main()