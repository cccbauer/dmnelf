# 02_tess_features.py
# Run locally: python 02_tess_features.py [--sfreq 250|500]
# Deploys cluster script, submits SLURM job.
#
# What it does on the cluster:
#   For each subject x task x run:
#   1. Load preprocessed EEG (250Hz or 500Hz FIF)
#   2. Load microstate templates fitted at same sfreq
#   3. TESS Stage 1: project onto 7 templates → T-hat (continuous)
#   4. Convolve each T-hat with canonical HRF (Glover 1999)
#   5. Downsample to TR (1.2s)
#   6. Append GFP and GMD (downsampled)
#   7. Save (n_vols, 9) feature matrix
#
# Output:
#   features/{subject}_task-{task}_run-{run}_{sfreq}Hz_features.npy

import argparse
import py_compile
import time
from pathlib import Path
from utils import run_ssh, scp_to, make_cluster_dirs
from config import (
    CLUSTER_BASE, SLURM_ACCOUNT, PYTHON,
    SUBJECTS_EEG_FMRI_ALL, EEG_ROOT,
    TR, N_MICROSTATES, LOCAL_BASE,
    MISSING_RUNS
)

parser = argparse.ArgumentParser()
parser.add_argument("--sfreq", type=int, default=250,
                    choices=[250, 500],
                    help="EEG sampling frequency (default: 250)")
parser.add_argument("--overwrite", action="store_true")
args   = parser.parse_args()
SFREQ     = args.sfreq
SFREQ_TAG = str(SFREQ) + "Hz"
TR_SAMPLES = int(round(SFREQ * TR))

# ── 1. Build cluster-side script ───────────────────────────
lines = [
    '#!/usr/bin/env python3',
    '"""',
    '02_tess_features_cluster.py',
    'Compute TESS T-hat features for all subjects x tasks x runs.',
    'Loads microstate templates fitted at the same sfreq.',
    '"""',
    'import sys',
    'sys.stdout.reconfigure(line_buffering=True)',
    'import numpy as np',
    'from pathlib import Path',
    'import mne',
    'from scipy.stats import gamma as gamma_dist',
    '',
    '# ── Constants ─────────────────────────────────────────',
    'EEG_ROOT       = Path("' + EEG_ROOT + '")',
    'CLUSTER_BASE   = Path("' + CLUSTER_BASE + '")',
    'TEMPLATES_PATH = CLUSTER_BASE / "microstates" / "templates_' + SFREQ_TAG + '.npy"',
    'OUT_DIR        = CLUSTER_BASE / "features"',
    'OUT_DIR.mkdir(parents=True, exist_ok=True)',
    '',
    'SUBJECTS      = ' + str(SUBJECTS_EEG_FMRI_ALL),
    'SFREQ         = ' + str(SFREQ),
    'SFREQ_TAG     = "' + SFREQ_TAG + '"',
    'TR            = ' + str(TR),
    'TR_SAMPLES    = ' + str(TR_SAMPLES),
    'N_MICROSTATES = ' + str(N_MICROSTATES),
    'MISSING_RUNS  = ' + str(MISSING_RUNS),
    'OVERWRITE     = ' + str(args.overwrite),
    '',
    'TASK_RUNS = {',
    '    "rest":      ["01", "02"],',
    '    "shortrest": ["01"],',
    '    "feedback":  ["01", "02", "03", "04"],',
    '}',
    '',
    '# ── Load templates ────────────────────────────────────',
    'print("=" * 55)',
    'print("Loading templates  " + SFREQ_TAG)',
    'print("=" * 55)',
    'if not TEMPLATES_PATH.exists():',
    '    print("ERROR: templates not found: " + str(TEMPLATES_PATH))',
    '    sys.exit(1)',
    'templates = np.load(str(TEMPLATES_PATH))',
    'print("Templates shape: " + str(templates.shape))',
    '',
    '# ── Helpers ───────────────────────────────────────────',
    'def load_eeg(fif_path):',
    '    raw = mne.io.read_raw_fif(str(fif_path), preload=True,',
    '                               verbose=False)',
    '    drop = [ch for ch in raw.ch_names',
    '            if any(x in ch.upper() for x in',
    '                   ("ECG","EKG","EMG","EOG","STIM","STATUS","TP9","TP10"))]',
    '    if drop:',
    '        raw.drop_channels(drop)',
    '    if raw.info["sfreq"] != SFREQ:',
    '        raw.resample(SFREQ, verbose=False)',
    '    data = (raw.get_data() * 1e6).astype(np.float32)',
    '    # Mean-center across channels at each time point',
    '    data -= data.mean(axis=1, keepdims=True)',
    '    return data',
    '',
    'def compute_gfp(eeg):',
    '    return eeg.std(axis=0).astype(np.float32)',
    '',
    'def compute_gmd(eeg):',
    '    diff = np.diff(eeg, axis=1)',
    '    gmd  = np.sqrt((diff ** 2).mean(axis=0)).astype(np.float32)',
    '    return np.concatenate([[gmd[0]], gmd]).astype(np.float32)',
    '',
    'def hrf_canonical(sfreq, duration=32.0):',
    '    """Canonical double-gamma HRF (Glover 1999) at EEG sfreq."""',
    '    dt    = 1.0 / sfreq',
    '    t     = np.arange(0.0, duration, dt)',
    '    peak  = gamma_dist.pdf(t, 5.0, scale=1.0)',
    '    under = gamma_dist.pdf(t, 15.0, scale=1.0)',
    '    hrf   = peak - under / 6.0',
    '    hrf  /= (hrf.max() + 1e-10)',
    '    return hrf.astype(np.float32)',
    '',
    'def convolve_hrf(signal, hrf):',
    '    conv = np.convolve(signal, hrf, mode="full")',
    '    return conv[:len(signal)].astype(np.float32)',
    '',
    'def downsample_to_tr(signal, tr_samples):',
    '    n_vols = len(signal) // tr_samples',
    '    out    = np.zeros(n_vols, dtype=np.float32)',
    '    for i in range(n_vols):',
    '        out[i] = signal[i*tr_samples:(i+1)*tr_samples].mean()',
    '    return out',
    '',
    'def compute_features(eeg, templates, tr_samples, sfreq):',
    '    """',
    '    Full TESS feature pipeline for one run.',
    '    Returns: (n_vols, n_maps + 2) float32',
    '    Columns: T-hat_A..T-hat_G, GFP, GMD',
    '    """',
    '    hrf    = hrf_canonical(sfreq)',
    '    n_maps = templates.shape[0]',
    '',
    '    # TESS Stage 1: spatial projection → T-hat (n_maps, n_samples)',
    '    # Mean-center both templates and EEG before projection',
    '    # (required for TESS — removes DC offset from dot product)',
    '    templates_mc = templates - templates.mean(axis=1, keepdims=True)',
    '    eeg_mc       = eeg - eeg.mean(axis=1, keepdims=True)',
    '    t_hat = (templates_mc @ eeg_mc).astype(np.float32)',
    '',
    '    # Downsample T-hat to TR (no HRF — predicting raw BOLD directly)',
    '    n_vols   = eeg.shape[1] // tr_samples',
    '    t_hat_tr = np.zeros((n_maps, n_vols), dtype=np.float32)',
    '    for m in range(n_maps):',
    '        t_hat_tr[m] = downsample_to_tr(t_hat[m], tr_samples)',
    '',
    '    # GFP and GMD → downsample',
    '    gfp_tr = downsample_to_tr(compute_gfp(eeg),  tr_samples)',
    '    gmd_tr = downsample_to_tr(compute_gmd(eeg),  tr_samples)',
    '',
    '    # Assemble (n_vols, 9)',
    '    n_vols = min(t_hat_tr.shape[1], len(gfp_tr), len(gmd_tr))',
    '    feats  = np.zeros((n_vols, n_maps + 2), dtype=np.float32)',
    '    feats[:, :n_maps]    = t_hat_tr[:, :n_vols].T',
    '    feats[:, n_maps]     = gfp_tr[:n_vols]',
    '    feats[:, n_maps + 1] = gmd_tr[:n_vols]',
    '    return feats',
    '',
    '# ── Main loop ─────────────────────────────────────────',
    'print()',
    'print("=" * 55)',
    'print("Computing TESS features  " + SFREQ_TAG)',
    'print("=" * 55)',
    '',
    'n_done    = 0',
    'n_skipped = 0',
    'n_missing = 0',
    '',
    'for subject in SUBJECTS:',
    '    for task, runs in TASK_RUNS.items():',
    '        for run in runs:',
    '',
    '            # Skip known missing raw data runs',
    '            if subject in MISSING_RUNS:',
    '                if (task, run) in MISSING_RUNS[subject]:',
    '                    continue',
    '',
    '            fif_name = (subject + "_ses-dmnelf_task-" + task',
    '                        + "_run-" + run',
    '                        + "_desc-preproc" + SFREQ_TAG + "_eeg.fif")',
    '            fif = (EEG_ROOT / subject / "ses-dmnelf"',
    '                   / "eeg" / fif_name)',
    '',
    '            out_name = (subject + "_task-" + task',
    '                        + "_run-" + run',
    '                        + "_" + SFREQ_TAG + "_nohrf_features.npy")',
    '            out_path = OUT_DIR / out_name',
    '',
    '            if out_path.exists() and not OVERWRITE:',
    '                print("  EXISTS (skip): " + out_name)',
    '                n_skipped += 1',
    '                continue', 
    '',
    '            if not fif.exists():',
    '                print("  MISSING: " + fif_name)',
    '                n_missing += 1',
    '                continue',
    '',
    '            print("  " + subject + "  " + task + "  run-" + run)',
    '            eeg   = load_eeg(fif)',
    '            feats = compute_features(eeg, templates,',
    '                                     TR_SAMPLES, SFREQ)',
    '            np.save(str(out_path), feats)',
    '            print("    shape=" + str(feats.shape))',
    '            n_done += 1',
    '',
    'print()',
    'print("=" * 55)',
    'print("DONE  " + SFREQ_TAG)',
    'print("  Computed: " + str(n_done))',
    'print("  Skipped:  " + str(n_skipped))',
    'print("  Missing:  " + str(n_missing))',
    'print("=" * 55)',
]

# ── 2. Save cluster script locally ─────────────────────────
script_name = "02_tess_features_" + SFREQ_TAG + "_cluster.py"
script_path = LOCAL_BASE / "scripts" / script_name
script_path.parent.mkdir(parents=True, exist_ok=True)

with open(script_path, "w") as f:
    f.write("\n".join(lines))

# ── 3. Syntax check ────────────────────────────────────────
print("Checking syntax...")
try:
    py_compile.compile(str(script_path), doraise=True)
    print("Syntax OK: " + script_name)
except py_compile.PyCompileError as e:
    print("SYNTAX ERROR: " + str(e))
    raise

# ── 4. Build SLURM script ──────────────────────────────────
job_name = "tess_" + SFREQ_TAG
sbatch_lines = [
    "#!/bin/bash",
    "#SBATCH --job-name=" + job_name,
    "#SBATCH --output=" + CLUSTER_BASE + "/logs/" + job_name + "_%j.out",
    "#SBATCH --error="  + CLUSTER_BASE + "/logs/" + job_name + "_%j.err",
    "#SBATCH --partition=short",
    "#SBATCH --time=04:00:00",
    "#SBATCH --cpus-per-task=4",
    "#SBATCH --mem=32G",
    "#SBATCH --account=" + SLURM_ACCOUNT,
    "",
    PYTHON + " " + CLUSTER_BASE + "/scripts/" + script_name
        + (" --overwrite" if args.overwrite else ""),
]

sbatch_name = "02_tess_features_" + SFREQ_TAG + ".sh"
sbatch_path = LOCAL_BASE / "scripts" / sbatch_name
with open(sbatch_path, "w") as f:
    f.write("\n".join(sbatch_lines))

# ── 5. Deploy ──────────────────────────────────────────────
print("\nDeploying...")
scp_to(script_path,
       CLUSTER_BASE + "/scripts/" + script_name,
       verbose=False)
scp_to(sbatch_path,
       CLUSTER_BASE + "/scripts/" + sbatch_name,
       verbose=False)
print("Deployed: " + script_name)
print("Deployed: " + sbatch_name)

# ── 6. Check templates exist before submitting ─────────────
templates_check = run_ssh(
    "ls " + CLUSTER_BASE + "/microstates/templates_"
    + SFREQ_TAG + ".npy 2>/dev/null || echo MISSING",
    verbose=False
)
if "MISSING" in templates_check.stdout:
    print("\nWARNING: templates_" + SFREQ_TAG + ".npy not found.")
    print("Run 01_fit_microstates.py --sfreq " + str(SFREQ) + " first.")
    print("Script deployed but not submitted.")
else:
    # ── 7. Submit ──────────────────────────────────────────
    print("\nSubmitting SLURM job...")
    result = run_ssh("sbatch " + CLUSTER_BASE + "/scripts/" + sbatch_name)
    job_id = ""
    for line in result.stdout.strip().split("\n"):
        if "Submitted" in line:
            job_id = line.strip().split()[-1]
            print("Job ID: " + job_id)

    # ── 8. Monitor ─────────────────────────────────────────
    if job_id:
        print("\nMonitoring job " + job_id + "  (Ctrl+C to stop)")
        print("-" * 55)
        try:
            while True:
                r = run_ssh(
                    "squeue -j " + job_id
                    + " --format=%.8i_%.8T_%.10M 2>/dev/null",
                    verbose=False
                )
                status = r.stdout.strip()
                if status and "JOBID" not in status.split("\n")[-1]:
                    print(status)
                else:
                    print("Job finished — checking log...")
                    log = run_ssh(
                        "tail -20 " + CLUSTER_BASE
                        + "/logs/" + job_name + "_" + job_id
                        + ".out 2>/dev/null",
                        verbose=False
                    )
                    print(log.stdout)
                    break
                time.sleep(15)
        except KeyboardInterrupt:
            print("\nStopped watching.")
            print("  tail -f " + CLUSTER_BASE
                  + "/logs/" + job_name + "_" + job_id + ".out")