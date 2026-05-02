#!/usr/bin/env python3
"""
check_config.py
Verify pipeline configuration for the current machine.
Run with:  python check_config.py

Exits 0 if all checks pass, 1 if any fail.
"""
import os
import sys
from pathlib import Path

# ── Colour helpers ────────────────────────────────────────────
def green(s):  return f"\033[32m{s}\033[0m"
def red(s):    return f"\033[31m{s}\033[0m"
def yellow(s): return f"\033[33m{s}\033[0m"
def bold(s):   return f"\033[1m{s}\033[0m"

PASS = green("PASS")
FAIL = red("FAIL")
WARN = yellow("WARN")
INFO = "INFO"

results = []

def check(label, ok, detail="", level="required"):
    tag = PASS if ok else (WARN if level == "warn" else FAIL)
    results.append((label, ok, detail, level))
    icon = "✓" if ok else ("⚠" if level == "warn" else "✗")
    colour = green if ok else (yellow if level == "warn" else red)
    print(f"  {colour(icon)}  {label:<45}  {tag}  {detail}")

# ── Import config ─────────────────────────────────────────────
print()
print(bold("=" * 65))
print(bold("  DMNELF Microstate PDA — Config Check"))
print(bold("=" * 65))
print()

try:
    import config as cfg
    print(f"  config.py loaded OK")
except Exception as e:
    print(red(f"  FATAL: cannot import config.py: {e}"))
    sys.exit(1)

try:
    from utils import run_ssh
    print(f"  utils.py loaded OK")
except Exception as e:
    print(red(f"  FATAL: cannot import utils.py: {e}"))
    sys.exit(1)

print()

# ── 1. Machine identity ───────────────────────────────────────
print(bold("1. Machine identity"))
user = os.environ.get("USER", "")
check("USER env var", bool(user), user)

known = {"anitya": "anitya desktop", "cccbauer": "cccbauer laptop"}
if user in known:
    check(f"Known machine ({known[user]})", True, f"USER={user}")
else:
    check("Known machine", False,
          f"USER={user!r} not in config — add branch to config.py", level="warn")

print()

# ── 2. Local paths ────────────────────────────────────────────
print(bold("2. Local paths"))
lb = cfg.LOCAL_BASE
check("LOCAL_BASE exists", lb.exists(), str(lb))

for name in ("scripts", "logs", "models", "results", "figures", "microstates"):
    d = lb / name
    check(f"  {name}/", d.exists(), str(d))

print()

# ── 3. Python executable ──────────────────────────────────────
print(bold("3. Python executable"))
exe = sys.executable
is_conda = "anaconda" in exe or "conda" in exe or "miniforge" in exe
check("Running under conda", is_conda,
      exe, level="required" if not is_conda else "required")
if not is_conda:
    print(f"       {yellow('⚠  System Python detected. Activate conda before running pipeline scripts.')}")

try:
    import numpy, pandas, mne
    check("numpy / pandas / mne importable", True,
          f"numpy {numpy.__version__}, mne {mne.__version__}")
except ImportError as e:
    check("numpy / pandas / mne importable", False, str(e))

print()

# ── 4. Cluster config ─────────────────────────────────────────
print(bold("4. Cluster config"))
check("CLUSTER_SSH defined", bool(cfg.CLUSTER_SSH), cfg.CLUSTER_SSH)
check("CLUSTER_BASE defined", bool(cfg.CLUSTER_BASE), cfg.CLUSTER_BASE)
check("SLURM_ACCOUNT defined", bool(cfg.SLURM_ACCOUNT), cfg.SLURM_ACCOUNT)
check("PYTHON (cluster) defined", bool(cfg.PYTHON), cfg.PYTHON)

print()

# ── 5. SSH connectivity ───────────────────────────────────────
print(bold("5. SSH connectivity"))
print(f"  Connecting to {cfg.CLUSTER_SSH} ...")
try:
    r = run_ssh("echo ok", verbose=False)
    ssh_ok = r.returncode == 0 and "ok" in r.stdout
    check("SSH to cluster", ssh_ok,
          r.stdout.strip() if ssh_ok else r.stderr.strip()[:80])
except Exception as e:
    check("SSH to cluster", False, str(e)[:80])

print()

# ── Summary ───────────────────────────────────────────────────
print(bold("=" * 65))
failures = [r for r in results if not r[1] and r[3] == "required"]
warnings = [r for r in results if not r[1] and r[3] == "warn"]

if failures:
    print(red(f"  {len(failures)} check(s) FAILED — fix before running pipeline."))
    sys.exit(1)
elif warnings:
    print(yellow(f"  All required checks passed. {len(warnings)} warning(s)."))
    sys.exit(0)
else:
    print(green("  All checks passed. Ready to run pipeline from this machine."))
    sys.exit(0)
