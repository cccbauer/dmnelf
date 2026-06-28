#!/usr/bin/env bash
set -euo pipefail

# deploy_rtbpd.sh — Full preprocessing + prediction pipeline for rtBPD pilot
# Adapts DMNELF scripts for rtBPD naming conventions

CLUSTER="cccbauer@explorer.northeastern.edu"
RTBPD="/projects/swglab/data/rtBPD"
DMNELF="/projects/swglab/data/DMNELF"
PROJ="${RTBPD}/analysis/fingerprint/rtbpd_replication"
PY="/home/cccbauer/.conda/envs/eeg_preproc/bin/python"
PY_PINEURO="/home/cccbauer/.conda/envs/pineuro/bin/python"
LOGDIR="${PROJ}/logs"

LOCAL_DIR="$(cd "$(dirname "$0")" && pwd)/.."

SUBJECTS=(rtbpd002 rtbpd003 rtbpd004)

echo "=========================================="
echo "  rtBPD Replication Pipeline"
echo "=========================================="

# ── 0. Deploy scripts ──
echo ""
echo ">>> [0] Deploying scripts to cluster..."
ssh "$CLUSTER" "mkdir -p ${PROJ}/{scripts,validation,results,logs}"
scp "$LOCAL_DIR/config.yaml" "${CLUSTER}:${PROJ}/config.yaml"
scp "$LOCAL_DIR/scripts/cross_cohort_predict.py" "${CLUSTER}:${PROJ}/scripts/"

# Also ensure wavelet scripts are deployed
WAVELET_DIR="$(dirname $LOCAL_DIR)/wavelet_coupling"
scp "$WAVELET_DIR/scripts/bandpower_wavelet.py" "${CLUSTER}:${DMNELF}/analysis/fingerprint/wavelet_coupling/scripts/"

# ── 1. EEG preprocessing ──
echo ""
echo ">>> [1] EEG preprocessing..."
ssh "$CLUSTER" << 'REMOTE'
set -e
PY=/home/cccbauer/.conda/envs/eeg_preproc/bin/python
RTBPD=/projects/swglab/data/rtBPD
LOGDIR=${RTBPD}/analysis/fingerprint/rtbpd_replication/logs

# Create rtBPD-adapted eeg_preproc wrapper that overrides paths
cat > /tmp/eeg_preproc_rtbpd.py << 'PYEOF'
#!/usr/bin/env python3
"""Wrapper to run eeg_preproc.py with rtBPD paths."""
import sys, importlib.util

# Load the DMNELF eeg_preproc module
spec = importlib.util.spec_from_file_location(
    "eeg_preproc",
    "/projects/swglab/data/DMNELF/analysis/MNE/jupyter/microstate_pda_v3/scripts/eeg_preproc.py"
)
mod = importlib.util.module_from_spec(spec)

# Override paths before executing
import types
# We'll monkey-patch the module globals after loading
spec.loader.exec_module(mod)

# Override the paths
mod.EEG_RAW_ROOT = __import__('pathlib').Path("/projects/swglab/data/rtBPD/rawdata_eeg")
mod.EEG_OUT_ROOT = __import__('pathlib').Path("/projects/swglab/data/rtBPD/derivatives/eeg_preprocessed")

# Override TASKS to include 5 feedback runs
mod.TASKS = {
    "rest":      ["01", "02", "03", "04"],
    "feedback":  ["01", "02", "03", "04", "05"],
}

# Override session - rtBPD uses ses-nf not ses-dmnelf
# The preprocess_run function constructs paths using a hardcoded session
# We need to find and fix that

# Actually, let's just call the function directly
import argparse
ap = argparse.ArgumentParser()
ap.add_argument("--subject", required=True)
ap.add_argument("--sfreq", type=float, default=500)
args = ap.parse_args()

subject = args.subject
sfreq = args.sfreq

ok, fail = 0, 0
for task, runs in mod.TASKS.items():
    for run in runs:
        edf = mod.EEG_RAW_ROOT / subject / "ses-nf" / "eeg" / \
              f"{subject}_ses-nf_task-{task}_run-{run}_desc-bvaAC1kHz_eeg.edf"
        if not edf.exists():
            continue
        out_dir = mod.EEG_OUT_ROOT / subject / "ses-nf" / "eeg"
        out_fif = out_dir / f"{subject}_ses-nf_task-{task}_run-{run}_desc-preproc{int(sfreq)}Hz_eeg.fif"
        if out_fif.exists():
            print(f"  EXISTS: {out_fif.name}")
            ok += 1
            continue
        try:
            mod.preprocess_run(subject, task, run, sfreq_target=sfreq)
            ok += 1
        except Exception as e:
            print(f"  FAILED {task} run-{run}: {e}")
            fail += 1

print(f"\nDONE  OK={ok}  FAILED={fail}")
PYEOF

for SUB in sub-rtbpd002 sub-rtbpd003 sub-rtbpd004; do
    COUNT=$(ls ${RTBPD}/derivatives/eeg_preprocessed/${SUB}/ses-nf/eeg/*feedback*500Hz*fif 2>/dev/null | wc -l)
    if [ "$COUNT" -ge 4 ]; then
        echo "  ${SUB}: already has ${COUNT} feedback fifs"
    else
        sbatch --job-name=eeg_${SUB##*-} --partition=short --mem=32G --cpus-per-task=4 --time=08:00:00 \
          --output=${LOGDIR}/eeg_${SUB##*-}.out --error=${LOGDIR}/eeg_${SUB##*-}.err \
          --wrap="${PY} /tmp/eeg_preproc_rtbpd.py --subject ${SUB} --sfreq 500"
        echo "  ${SUB}: submitted"
    fi
done
REMOTE

echo ""
echo "=========================================="
echo "  EEG preprocessing submitted."
echo "  Monitor: ssh ${CLUSTER} 'squeue -u cccbauer'"
echo "  After EEG + masks complete, run feature extraction + prediction."
echo "=========================================="
