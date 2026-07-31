# utils_cluster_conn.py — SSH/SCP helpers for the PLV connectivity pipeline.
# Identical logic to the other rtBPD pipelines' utils_cluster*.py, pointed
# at config_connectivity.CLUSTER_BASE so this stays fully decoupled.

import subprocess
from pathlib import Path

from config_connectivity import CLUSTER_SSH, CLUSTER_BASE


def run_ssh(cmd, verbose=True):
    """Run a command on the cluster via SSH."""
    full = "/usr/bin/ssh " + CLUSTER_SSH + " 'bash -l -c \"" + cmd + "\"'"
    result = subprocess.run(
        full, shell=True, capture_output=True, text=True
    )
    if verbose:
        if result.stdout:
            print(result.stdout)
        if result.stderr:
            filtered = "\n".join([
                line for line in result.stderr.split("\n")
                if not any(x in line for x in [
                    "flatpak",
                    "libcrypto",
                    "OPENSSL",
                    "Loading matlab",
                    "Loading requirement",
                    "OpenJDK",
                ])
            ]).strip()
            if filtered:
                print(filtered)
    return result


def scp_to(local_path, remote_path, verbose=True):
    """Copy a local file to the cluster."""
    cmd = ("/usr/bin/scp '" + str(local_path)
           + "' " + CLUSTER_SSH + ":" + remote_path)
    result = subprocess.run(
        cmd, shell=True, capture_output=True, text=True
    )
    if verbose:
        if result.stdout: print(result.stdout)
        if result.stderr: print(result.stderr)
    verify = run_ssh("ls " + remote_path + " 2>/dev/null || echo MISSING",
                     verbose=False)
    if "MISSING" in verify.stdout:
        print("WARNING: SCP failed for " + str(local_path))
    elif verbose:
        print("Verified: " + remote_path)
    return result


def scp_from(remote_path, local_path, verbose=True):
    """Copy a file from the cluster to local."""
    cmd = ("/usr/bin/scp " + CLUSTER_SSH + ":" + remote_path
           + " " + str(local_path))
    result = subprocess.run(
        cmd, shell=True, capture_output=True, text=True
    )
    if verbose:
        if result.stdout: print(result.stdout)
        if result.stderr: print(result.stderr)
    return result


def make_cluster_dirs():
    """Create all required directories on the cluster."""
    dirs = [
        CLUSTER_BASE,
        CLUSTER_BASE + "/scripts",
        CLUSTER_BASE + "/results",
        CLUSTER_BASE + "/logs",
    ]
    cmd = "mkdir -p " + " ".join(dirs)
    return run_ssh(cmd)
