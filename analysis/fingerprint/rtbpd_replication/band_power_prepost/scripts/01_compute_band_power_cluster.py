#!/usr/bin/env python3
"""
01_compute_band_power_cluster.py
Welch PSD -> 5 standard band powers (dB), averaged across 29 EEG
channels, for each of the 15 rtBPD nf1 subjects x 4 rest runs.
"""
import sys
sys.stdout.reconfigure(line_buffering=True)
import numpy as np
import csv
from pathlib import Path
import mne

# -- Paths and constants ---------------------------------
EEG_ROOT      = Path("/projects/swglab/data/rtBPD/derivatives/eeg_preprocessed")
OUT_DIR       = Path("/projects/swglab/data/rtBPD/analysis/fingerprint/rtbpd_replication/band_power_prepost")
RESULTS_DIR   = OUT_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
SUBJECTS      = ['sub-rtbpd002', 'sub-rtbpd003', 'sub-rtbpd009', 'sub-rtbpd010', 'sub-rtbpd011', 'sub-rtbpd012', 'sub-rtbpd013', 'sub-rtbpd015', 'sub-rtbpd018', 'sub-rtbpd020', 'sub-rtbpd021', 'sub-rtbpd024', 'sub-rtbpd030', 'sub-rtbpd038', 'sub-rtbpd040']
SESSION       = "ses-nf"
ALL_RUNS      = ['01', '02', '03', '04']
PRE_RUNS      = ['01', '02']
POST_RUNS     = ['03', '04']
SFREQ         = 500
EEG_DESC      = "preproc500Hz"
EXCLUDE_CHS   = ['TP9', 'TP10']
WELCH_WIN_SEC = 4.0
WELCH_OVERLAP = 0.5
BANDS         = {'delta': (1.0, 4.0), 'theta': (4.0, 8.0), 'alpha': (8.0, 13.0), 'beta': (13.0, 30.0), 'gamma': (30.0, 40.0)}

# -- Helpers ----------------------------------------------
def load_eeg_raw(fif_path):
    """Same drop/resample logic as the microstate pipelines' load_eeg(),
    but returns the MNE Raw object itself (compute_psd needs it),
    rather than a bare numpy array."""
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

# -- Compute band power for every run -------------------------
print("=" * 55)
print("Welch PSD -> band power (dB), " + str(len(SUBJECTS))
      + " subjects x " + str(len(ALL_RUNS)) + " runs")
print("Excluding: " + str(EXCLUDE_CHS)
      + "  (+ ECG/EKG/EMG/EOG/STIM/STATUS by substring)")
print("Welch window: " + str(WELCH_WIN_SEC) + "s  overlap="
      + str(int(WELCH_OVERLAP * 100)) + "%")
print("Bands: " + str(BANDS))
print("=" * 55)

n_per_seg = int(WELCH_WIN_SEC * SFREQ)
n_overlap = int(WELCH_OVERLAP * n_per_seg)

rows = []
n_ok = 0
n_missing = 0

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
        n_ch = len(mne.pick_types(raw.info, eeg=True))

        # Welch PSD per channel: (n_channels, n_freqs), linear power
        # (V^2/Hz on the microvolt-scaled data MNE stores internally
        # in volts -- absolute scale does not matter here since we
        # only ever compare pre vs post on the same scale).
        #
        # reject_by_annotation=False: some runs carry a small trailing
        # BAD_edge_end annotation from Phase 1 preprocessing (edge-ramp
        # artifact, typically <0.1% of samples). MNE's default
        # reject_by_annotation=True excises that span and Welch-PSDs
        # the tiny leftover tail, which can be shorter than the 4s
        # window and crash ("noverlap must be less than nperseg").
        # The microstate pipeline never applies annotation-based
        # rejection either -- it treats the full continuous recording
        # as valid -- so we match that same approach here.
        spectrum = raw.compute_psd(
            method="welch", picks="eeg",
            fmin=0.5, fmax=45.0,
            n_fft=n_per_seg, n_per_seg=n_per_seg, n_overlap=n_overlap,
            reject_by_annotation=False,
            verbose=False,
        )
        psd = spectrum.get_data()   # (n_ch, n_freqs), linear power
        freqs = spectrum.freqs

        for band, (lo, hi) in BANDS.items():
            fmask = (freqs >= lo) & (freqs < hi)
            # 1) mean over frequency bins within the band, per channel
            band_power_per_ch = psd[:, fmask].mean(axis=1)
            # 2) mean (linear) across the 29 channels
            band_power_avg = float(band_power_per_ch.mean())
            # 3) log-transform the channel-averaged linear power
            log_power_db = 10.0 * np.log10(band_power_avg + 1e-30)
            rows.append(dict(
                subject=subject, run=run, condition=condition,
                band=band, log_power_db=round(log_power_db, 4),
            ))

        n_ok += 1
        print("  " + subject + "  run-" + run + "  (" + condition + ")"
              + "  ch=" + str(n_ch)
              + "  freqs=" + str(len(freqs)))

print()
print("Runs processed: " + str(n_ok) + "  missing: " + str(n_missing))

# -- Write CSV ------------------------------------------------
out_csv = RESULTS_DIR / "band_power.csv"
fields = ["subject", "run", "condition", "band", "log_power_db"]
with open(str(out_csv), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=fields)
    w.writeheader()
    w.writerows(rows)
print("Saved: " + str(out_csv) + "  (" + str(len(rows)) + " rows)")

print()
print("DONE")