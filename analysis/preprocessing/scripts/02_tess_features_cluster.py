#!/usr/bin/env python3
"""
02_tess_features_cluster.py
Compute TESS T-hat features for all subjects x tasks x runs.
Loads templates from 01_fit_microstates.py output.
"""
import sys
sys.stdout.reconfigure(line_buffering=True)
import numpy as np
from pathlib import Path
import mne
from scipy.signal import argrelmax
from scipy.stats import gamma as gamma_dist

# ── Constants ─────────────────────────────────────────
EEG_ROOT      = Path("/projects/swglab/data/DMNELF/analysis/MNE/bids/derivatives/preprocessed")
CLUSTER_BASE  = Path("/projects/swglab/data/DMNELF/analysis/MNE/jupyter/microstate_pda_v3")
TEMPLATES_PATH = CLUSTER_BASE / "microstates" / "templates.npy"
OUT_DIR       = CLUSTER_BASE / "features"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SUBJECTS      = ['sub-dmnelf001', 'sub-dmnelf004', 'sub-dmnelf005', 'sub-dmnelf006', 'sub-dmnelf007', 'sub-dmnelf008', 'sub-dmnelf010']
SFREQ         = 200
TR_SAMPLES    = 240
TR            = 1.2
N_MICROSTATES = 7

TASK_RUNS = {
    "rest":      ["01", "02"],
    "shortrest": ["01"],
    "feedback":  ["01", "02", "03", "04"],
}

# ── Load templates ────────────────────────────────────
print("=" * 55)
print("Loading templates")
print("=" * 55)
if not TEMPLATES_PATH.exists():
    print("ERROR: templates.npy not found at " + str(TEMPLATES_PATH))
    print("Run 01_fit_microstates.py first.")
    sys.exit(1)

templates = np.load(str(TEMPLATES_PATH))
print("Templates shape: " + str(templates.shape))
print("Expected:        (" + str(N_MICROSTATES) + ", n_channels)")

# ── Helpers ───────────────────────────────────────────
def load_eeg(fif_path):
    raw = mne.io.read_raw_fif(str(fif_path), preload=True,
                               verbose=False)
    drop = [ch for ch in raw.ch_names
            if any(x in ch.upper() for x in
                   ("ECG","EKG","EMG","EOG","STIM","STATUS","TP9","TP10"))]
    if drop:
        raw.drop_channels(drop)
    ref = float(np.abs(raw.get_data().mean(axis=0)).mean())
    if ref > 1e-7:
        raw.set_eeg_reference("average", projection=False,
                               verbose=False)
    if raw.info["sfreq"] != SFREQ:
        raw.resample(SFREQ, verbose=False)
    return (raw.get_data() * 1e6).astype(np.float32)

def compute_gfp(eeg):
    return eeg.std(axis=0).astype(np.float32)

def compute_gmd(eeg):
    norm = eeg / (eeg.std(axis=0, keepdims=True) + 1e-10)
    diff = np.diff(norm, axis=1)
    gmd  = np.sqrt((diff ** 2).mean(axis=0)).astype(np.float32)
    return np.concatenate([[0.0], gmd]).astype(np.float32)

def hrf_canonical(tr, duration=32.0):
    """Canonical double-gamma HRF (Glover 1999)."""
    t     = np.arange(0.0, duration, tr)
    peak  = gamma_dist.pdf(t, 5.0, scale=1.0)
    under = gamma_dist.pdf(t, 15.0, scale=1.0)
    hrf   = peak - under / 6.0
    hrf  /= hrf.max()
    return hrf.astype(np.float32)

def convolve_hrf(signal, hrf):
    conv = np.convolve(signal, hrf, mode="full")
    return conv[:len(signal)].astype(np.float32)

def downsample_to_tr(signal, tr_samples):
    n_vols = len(signal) // tr_samples
    out    = np.zeros(n_vols, dtype=np.float32)
    for i in range(n_vols):
        out[i] = signal[i * tr_samples:(i + 1) * tr_samples].mean()
    return out

def tess_project(eeg, templates):
    """
    TESS Stage 1 spatial GLM.
    Returns T-hat: (n_maps, n_samples)
    """
    return (templates @ eeg).astype(np.float32)

def compute_features(eeg, templates, tr_samples, tr):
    """
    Full feature pipeline for one run.
    Returns: (n_vols, n_maps + 2) float32
    Columns: T-hat_0..T-hat_6, GFP, GMD
    """
    hrf    = hrf_canonical(tr)
    n_maps = templates.shape[0]

    # T-hat at EEG rate → HRF convolve → downsample
    t_hat  = tess_project(eeg, templates)
    n_vols = len(eeg[0]) // tr_samples
    t_hat_tr = np.zeros((n_maps, n_vols), dtype=np.float32)
    for m in range(n_maps):
        conv         = convolve_hrf(t_hat[m], hrf)
        t_hat_tr[m]  = downsample_to_tr(conv, tr_samples)

    # GFP and GMD → downsample
    gfp    = compute_gfp(eeg)
    gmd    = compute_gmd(eeg)
    gfp_tr = downsample_to_tr(gfp, tr_samples)
    gmd_tr = downsample_to_tr(gmd, tr_samples)

    # Trim to min length (alignment safety)
    n_vols = min(t_hat_tr.shape[1], len(gfp_tr), len(gmd_tr))
    feats  = np.zeros((n_vols, n_maps + 2), dtype=np.float32)
    feats[:, :n_maps]    = t_hat_tr[:, :n_vols].T
    feats[:, n_maps]     = gfp_tr[:n_vols]
    feats[:, n_maps + 1] = gmd_tr[:n_vols]
    return feats

# ── Main loop ─────────────────────────────────────────
print()
print("=" * 55)
print("Computing TESS features")
print("=" * 55)

n_done    = 0
n_skipped = 0
n_missing = 0

for subject in SUBJECTS:
    for task, runs in TASK_RUNS.items():
        for run in runs:
            fif_name = (subject + "_ses-dmnelf_task-" + task
                        + "_run-" + run + "_desc-preproc_eeg.fif")
            fif = (EEG_ROOT / subject / "ses-dmnelf"
                   / "eeg" / fif_name)

            out_name = (subject + "_task-" + task
                        + "_run-" + run + "_features.npy")
            out_path = OUT_DIR / out_name

            if not fif.exists():
                print("  MISSING EEG: " + fif_name)
                n_missing += 1
                continue

            if out_path.exists():
                print("  EXISTS (skip): " + out_name)
                n_skipped += 1
                continue

            print("  Processing: " + subject
                  + "  task-" + task + "  run-" + run)

            eeg   = load_eeg(fif)
            feats = compute_features(eeg, templates,
                                     TR_SAMPLES, TR)

            np.save(str(out_path), feats)

            print("    EEG:      " + str(eeg.shape))
            print("    Features: " + str(feats.shape)
                  + "  expect (n_vols, " + str(N_MICROSTATES + 2) + ")")
            print("    Saved:    " + out_name)
            n_done += 1

print()
print("=" * 55)
print("Summary")
print("=" * 55)
print("  Computed: " + str(n_done))
print("  Skipped:  " + str(n_skipped) + " (already existed)")
print("  Missing:  " + str(n_missing) + " (EEG file not found)")
print()
print("DONE")