#!/usr/bin/env python3
"""
02_tess_features_cluster.py
Compute TESS T-hat features for all subjects x tasks x runs.
Loads microstate templates fitted at the same sfreq.
"""
import sys
sys.stdout.reconfigure(line_buffering=True)
import numpy as np
from pathlib import Path
import mne
from scipy.stats import gamma as gamma_dist

# ── Constants ─────────────────────────────────────────
EEG_ROOT       = Path("/projects/swglab/data/DMNELF/derivatives/eeg_preprocessed")
CLUSTER_BASE   = Path("/projects/swglab/data/DMNELF/analysis/MNE/jupyter/microstate_pda_v3")
TEMPLATES_PATH = CLUSTER_BASE / "microstates" / "templates_250Hz.npy"
OUT_DIR        = CLUSTER_BASE / "features"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SUBJECTS      = ['sub-dmnelf001', 'sub-dmnelf004', 'sub-dmnelf005', 'sub-dmnelf006', 'sub-dmnelf007', 'sub-dmnelf008', 'sub-dmnelf009', 'sub-dmnelf010', 'sub-dmnelf011', 'sub-dmnelf1001', 'sub-dmnelf1002', 'sub-dmnelf1003']
SFREQ         = 250
SFREQ_TAG     = "250Hz"
TR            = 1.2
TR_SAMPLES    = 300
N_MICROSTATES = 7
MISSING_RUNS  = {'sub-dmnelf1002': [('rest', '02')], 'sub-dmnelf1003': [('feedback', '04')]}
OVERWRITE     = True

TASK_RUNS = {
    "rest":      ["01", "02"],
    "shortrest": ["01"],
    "feedback":  ["01", "02", "03", "04"],
}

# ── Load templates ────────────────────────────────────
print("=" * 55)
print("Loading templates  " + SFREQ_TAG)
print("=" * 55)
if not TEMPLATES_PATH.exists():
    print("ERROR: templates not found: " + str(TEMPLATES_PATH))
    sys.exit(1)
templates = np.load(str(TEMPLATES_PATH))
print("Templates shape: " + str(templates.shape))

# ── Helpers ───────────────────────────────────────────
def load_eeg(fif_path):
    raw = mne.io.read_raw_fif(str(fif_path), preload=True,
                               verbose=False)
    drop = [ch for ch in raw.ch_names
            if any(x in ch.upper() for x in
                   ("ECG","EKG","EMG","EOG","STIM","STATUS","TP9","TP10"))]
    if drop:
        raw.drop_channels(drop)
    if raw.info["sfreq"] != SFREQ:
        raw.resample(SFREQ, verbose=False)
    data = (raw.get_data() * 1e6).astype(np.float32)
    # Mean-center across channels at each time point
    data -= data.mean(axis=1, keepdims=True)
    return data

def compute_gfp(eeg):
    return eeg.std(axis=0).astype(np.float32)

def compute_gmd(eeg):
    diff = np.diff(eeg, axis=1)
    gmd  = np.sqrt((diff ** 2).mean(axis=0)).astype(np.float32)
    return np.concatenate([[gmd[0]], gmd]).astype(np.float32)

def hrf_canonical(sfreq, duration=32.0):
    """Canonical double-gamma HRF (Glover 1999) at EEG sfreq."""
    dt    = 1.0 / sfreq
    t     = np.arange(0.0, duration, dt)
    peak  = gamma_dist.pdf(t, 5.0, scale=1.0)
    under = gamma_dist.pdf(t, 15.0, scale=1.0)
    hrf   = peak - under / 6.0
    hrf  /= (hrf.max() + 1e-10)
    return hrf.astype(np.float32)

def convolve_hrf(signal, hrf):
    conv = np.convolve(signal, hrf, mode="full")
    return conv[:len(signal)].astype(np.float32)

def downsample_to_tr(signal, tr_samples):
    n_vols = len(signal) // tr_samples
    out    = np.zeros(n_vols, dtype=np.float32)
    for i in range(n_vols):
        out[i] = signal[i*tr_samples:(i+1)*tr_samples].mean()
    return out

def compute_features(eeg, templates, tr_samples, sfreq):
    """
    Full TESS feature pipeline for one run.
    Returns: (n_vols, n_maps + 2) float32
    Columns: T-hat_A..T-hat_G, GFP, GMD
    """
    hrf    = hrf_canonical(sfreq)
    n_maps = templates.shape[0]

    # TESS Stage 1: spatial projection → T-hat (n_maps, n_samples)
    # Mean-center both templates and EEG before projection
    # (required for TESS — removes DC offset from dot product)
    templates_mc = templates - templates.mean(axis=1, keepdims=True)
    eeg_mc       = eeg - eeg.mean(axis=1, keepdims=True)
    t_hat = (templates_mc @ eeg_mc).astype(np.float32)

    # Downsample T-hat to TR (no HRF — predicting raw BOLD directly)
    n_vols   = eeg.shape[1] // tr_samples
    t_hat_tr = np.zeros((n_maps, n_vols), dtype=np.float32)
    for m in range(n_maps):
        t_hat_tr[m] = downsample_to_tr(t_hat[m], tr_samples)

    # GFP and GMD → downsample
    gfp_tr = downsample_to_tr(compute_gfp(eeg),  tr_samples)
    gmd_tr = downsample_to_tr(compute_gmd(eeg),  tr_samples)

    # Assemble (n_vols, 9)
    n_vols = min(t_hat_tr.shape[1], len(gfp_tr), len(gmd_tr))
    feats  = np.zeros((n_vols, n_maps + 2), dtype=np.float32)
    feats[:, :n_maps]    = t_hat_tr[:, :n_vols].T
    feats[:, n_maps]     = gfp_tr[:n_vols]
    feats[:, n_maps + 1] = gmd_tr[:n_vols]
    return feats

# ── Main loop ─────────────────────────────────────────
print()
print("=" * 55)
print("Computing TESS features  " + SFREQ_TAG)
print("=" * 55)

n_done    = 0
n_skipped = 0
n_missing = 0

for subject in SUBJECTS:
    for task, runs in TASK_RUNS.items():
        for run in runs:

            # Skip known missing raw data runs
            if subject in MISSING_RUNS:
                if (task, run) in MISSING_RUNS[subject]:
                    continue

            fif_name = (subject + "_ses-dmnelf_task-" + task
                        + "_run-" + run
                        + "_desc-preproc" + SFREQ_TAG + "_eeg.fif")
            fif = (EEG_ROOT / subject / "ses-dmnelf"
                   / "eeg" / fif_name)

            out_name = (subject + "_task-" + task
                        + "_run-" + run
                        + "_" + SFREQ_TAG + "_nohrf_features.npy")
            out_path = OUT_DIR / out_name

            if out_path.exists() and not OVERWRITE:
                print("  EXISTS (skip): " + out_name)
                n_skipped += 1
                continue

            if not fif.exists():
                print("  MISSING: " + fif_name)
                n_missing += 1
                continue

            print("  " + subject + "  " + task + "  run-" + run)
            eeg   = load_eeg(fif)
            feats = compute_features(eeg, templates,
                                     TR_SAMPLES, SFREQ)
            np.save(str(out_path), feats)
            print("    shape=" + str(feats.shape))
            n_done += 1

print()
print("=" * 55)
print("DONE  " + SFREQ_TAG)
print("  Computed: " + str(n_done))
print("  Skipped:  " + str(n_skipped))
print("  Missing:  " + str(n_missing))
print("=" * 55)