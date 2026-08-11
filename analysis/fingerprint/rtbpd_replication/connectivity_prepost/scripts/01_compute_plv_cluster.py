#!/usr/bin/env python3
"""
01_compute_plv_cluster.py
Whole-run PLV (phase-locking value) between curated DMN-relevant
channel pairs, averaged across pairs, for each of the 20 rtBPD nf1
subjects x 4 rest runs x 5 bands.

Cohort expanded 2026-08-11 from 15 -> 20: added sub-rtbpd001 (already had
complete 4/4 fMRI+EEG, simply missed by the original preflight), sub-rtbpd045
(complete 4/4 both modalities), and sub-rtbpd022/026/039 (4/4 fMRI but only
3/4 EEG rest runs each -- one pre or post run missing; the per-condition
mean in 02_stats_plv_cluster.py naturally falls back to the single
available run for these three, so no code change was needed there).
"""
import sys
sys.stdout.reconfigure(line_buffering=True)
import numpy as np
import csv
from pathlib import Path
from itertools import combinations
import mne
from scipy.signal import hilbert
mne.set_log_level("ERROR")

# -- Paths and constants ---------------------------------
EEG_ROOT      = Path("/projects/swglab/data/rtBPD/derivatives/eeg_preprocessed")
OUT_DIR       = Path("/projects/swglab/data/rtBPD/analysis/fingerprint/rtbpd_replication/connectivity_prepost")
RESULTS_DIR   = OUT_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
SUBJECTS      = ['sub-rtbpd002', 'sub-rtbpd003', 'sub-rtbpd009', 'sub-rtbpd010', 'sub-rtbpd011', 'sub-rtbpd012', 'sub-rtbpd013', 'sub-rtbpd015', 'sub-rtbpd018', 'sub-rtbpd020', 'sub-rtbpd021', 'sub-rtbpd024', 'sub-rtbpd030', 'sub-rtbpd038', 'sub-rtbpd040', 'sub-rtbpd001', 'sub-rtbpd022', 'sub-rtbpd026', 'sub-rtbpd039', 'sub-rtbpd045']
SESSION       = "ses-nf"
ALL_RUNS      = ['01', '02', '03', '04']
PRE_RUNS      = ['01', '02']
POST_RUNS     = ['03', '04']
SFREQ         = 500
EEG_DESC      = "preproc500Hz"
EXCLUDE_CHS   = ['TP9', 'TP10']
BANDS         = {'delta': (1.0, 4.0), 'theta': (4.0, 8.0), 'alpha': (8.0, 13.0), 'beta': (13.0, 30.0), 'gamma': (30.0, 40.0)}

# -- Helpers ----------------------------------------------
def load_eeg_raw(fif_path):
    """Same drop/resample logic as the other pipelines' load_eeg()."""
    raw = mne.io.read_raw_fif(str(fif_path), preload=True,
                               verbose=False)
    drop = [ch for ch in raw.ch_names
            if any(x in ch.upper() for x in
                   ("ECG","EKG","EMG","EOG","STIM","STATUS"))
            or ch in EXCLUDE_CHS]
    if drop:
        raw.drop_channels(drop)
    if raw.info["sfreq"] != SFREQ:
        raw.resample(SFREQ, verbose=False)
    return raw

def get_dmn_relevant_pairs(ch_names):
    """Select channel pairs relevant to DMN connectivity.
    DMN involves frontal-posterior and midline connectivity.
    Ported VERBATIM from wavelet_coupling/scripts/connectivity_features.py
    (frontal/posterior/midline/temporal channel-name sets unchanged).
    Note: TP9/TP10 are excluded from this montage (EXCLUDE_CHS), so
    the "temporal" set below naturally reduces to just T7/T8 here --
    the function's own `if t in ch_idx` guard already handles this,
    no code change needed.

    Frontal: Fp1, Fp2, F3, F4, Fz, FC1, FC2
    Posterior: P3, P4, Pz, O1, O2, Oz, POz
    Midline: Fz, Cz, Pz, Oz, POz
    """
    frontal = {"Fp1", "Fp2", "F3", "F4", "Fz", "FC1", "FC2", "F7", "F8"}
    posterior = {"P3", "P4", "Pz", "O1", "O2", "Oz", "POz", "P7", "P8"}
    midline = {"Fz", "Cz", "Pz", "Oz", "POz"}
    temporal = {"T7", "T8", "TP9", "TP10"}

    pairs = []
    ch_idx = {ch: i for i, ch in enumerate(ch_names)}

    # Frontal-posterior pairs (DMN long-range)
    for f in frontal:
        for p in posterior:
            if f in ch_idx and p in ch_idx:
                pairs.append((ch_idx[f], ch_idx[p]))

    # Midline pairs (DMN midline axis)
    midline_list = [ch for ch in midline if ch in ch_idx]
    for a, b in combinations([ch_idx[ch] for ch in midline_list], 2):
        pairs.append((a, b))

    # Temporal-frontal pairs
    for t in temporal:
        for f in frontal:
            if t in ch_idx and f in ch_idx:
                pairs.append((ch_idx[t], ch_idx[f]))

    return list(set(pairs))  # deduplicate

def compute_whole_run_plv(raw, bands, pairs):
    """Whole-run PLV per band, averaged across the curated pairs.
    Adapted from connectivity_features.py's compute_plv_run(): same
    bandpass -> Hilbert -> |mean(exp(i*phase_diff))| formula, but a
    SINGLE PLV value over the full run instead of per-TR blocks, and
    no HRF convolution (both TR-blocking and HRF are EEG-BOLD
    alignment machinery, irrelevant for a pure rest EEG-EEG measure).
    """
    picks = mne.pick_types(raw.info, eeg=True, exclude=[])
    data = raw.get_data(picks=picks)   # (n_ch, n_samples)
    sfreq = raw.info["sfreq"]

    band_plv = {}
    for bname, (lo, hi) in bands.items():
        filtered = mne.filter.filter_data(data, sfreq, lo, hi, verbose=False)
        analytic = hilbert(filtered, axis=-1)
        phase = np.angle(analytic)   # (n_ch, n_samples)

        pair_plvs = []
        for ch_a, ch_b in pairs:
            phase_diff = phase[ch_a] - phase[ch_b]
            plv = float(np.abs(np.mean(np.exp(1j * phase_diff))))
            pair_plvs.append(plv)
        band_plv[bname] = float(np.mean(pair_plvs))
    return band_plv

# -- Compute PLV for every run -------------------------------
print("=" * 55)
print("Whole-run PLV, " + str(len(SUBJECTS)) + " subjects x "
      + str(len(ALL_RUNS)) + " runs x " + str(len(BANDS)) + " bands")
print("Excluding: " + str(EXCLUDE_CHS)
      + "  (+ ECG/EKG/EMG/EOG/STIM/STATUS by substring)")
print("Bands: " + str(BANDS))
print("=" * 55)

rows = []
n_ok = 0
n_missing = 0
n_pairs_reported = None

for subject in SUBJECTS:
    for run in ALL_RUNS:
        condition = "pre" if run in PRE_RUNS else "post"
        fname = (subject + "_" + SESSION + "_task-rest"
                 + "_run-" + run
                 + "_desc-" + EEG_DESC + "_eeg.fif")
        fif = (EEG_ROOT / subject / SESSION / "eeg" / fname)
        if not fif.exists():
            print("  MISSING: " + fname)
            n_missing += 1
            continue

        raw = load_eeg_raw(fif)
        picks = mne.pick_types(raw.info, eeg=True, exclude=[])
        ch_names = [raw.ch_names[i] for i in picks]
        pairs = get_dmn_relevant_pairs(ch_names)
        if n_pairs_reported is None:
            n_pairs_reported = len(pairs)
            print("  DMN-relevant pairs: " + str(len(pairs))
                  + "  (from " + str(len(ch_names)) + " channels)")

        band_plv = compute_whole_run_plv(raw, BANDS, pairs)
        for band, plv in band_plv.items():
            rows.append(dict(
                subject=subject, run=run, condition=condition,
                band=band, plv=round(plv, 5),
            ))

        n_ok += 1
        print("  " + subject + "  run-" + run + "  (" + condition + ")"
              + "  n_samples=" + str(raw.n_times)
              + "  plv=" + str({k: round(v, 3) for k, v in band_plv.items()}))

print()
print("Runs processed: " + str(n_ok) + "  missing: " + str(n_missing))

# -- Write CSV ------------------------------------------------
out_csv = RESULTS_DIR / "plv_connectivity.csv"
fields = ["subject", "run", "condition", "band", "plv"]
with open(str(out_csv), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=fields)
    w.writeheader()
    w.writerows(rows)
print("Saved: " + str(out_csv) + "  (" + str(len(rows)) + " rows)")

print()
print("DONE")