#!/usr/bin/env python3
"""
02_temporal_params_band_k4_cluster_fullspectrum.py
Back-fit continuous rest EEG (band=fullspectrum) onto the 4 classic
A/B/C/D templates fit by step 01 and compute classic microstate
temporal parameters (duration, occurrence, coverage, GEV) per
subject x run x microstate.
"""
import sys
sys.stdout.reconfigure(line_buffering=True)
import numpy as np
import json
import csv
from pathlib import Path
import mne

# -- Paths and constants ---------------------------------
EEG_ROOT      = Path("/projects/swglab/data/rtBPD/derivatives/eeg_preprocessed")
OUT_DIR       = Path("/projects/swglab/data/rtBPD/analysis/fingerprint/rtbpd_replication/microstates/band_k4")
MS_DIR        = OUT_DIR / "microstates"
RESULTS_DIR   = OUT_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
SUBJECTS      = ['sub-rtbpd002', 'sub-rtbpd003', 'sub-rtbpd009', 'sub-rtbpd010', 'sub-rtbpd011', 'sub-rtbpd012', 'sub-rtbpd013', 'sub-rtbpd015', 'sub-rtbpd018', 'sub-rtbpd020', 'sub-rtbpd021', 'sub-rtbpd024', 'sub-rtbpd030', 'sub-rtbpd038', 'sub-rtbpd040']
SESSION       = "ses-nf"
ALL_RUNS      = ['01', '02', '03', '04']
PRE_RUNS      = ['01', '02']
POST_RUNS     = ['03', '04']
SFREQ         = 500
SFREQ_TAG     = "500Hz"
EEG_DESC      = "preproc500Hz"
N_MICROSTATES = 4
EXCLUDE_CHS   = ['TP9', 'TP10']
BAND          = "fullspectrum"
THETA_LO      = 4.0
THETA_HI      = 8.0
TAG           = "fullspectrum_k4_500Hz"
MIN_SEG_SAMPLES = 3   # Pascual-Marqui (1995) minimum-duration criterion

# -- Load templates + labels + reference channel order ----
templates = np.load(str(MS_DIR / ("templates_" + TAG + ".npy")))
assign_path = MS_DIR / ("assignments_" + TAG + ".json")
if assign_path.exists():
    with open(str(assign_path)) as f:
        _asgn = json.load(f)
    MS_LABELS = [_asgn.get(str(i), "MS" + chr(65 + i))
                 for i in range(N_MICROSTATES)]
else:
    MS_LABELS = ["MS" + chr(65 + i) for i in range(N_MICROSTATES)]
print("Microstate labels:", MS_LABELS)

chan_path = MS_DIR / ("channels_" + TAG + ".json")
with open(str(chan_path)) as f:
    REF_CHANNELS = json.load(f)
print("Reference channels (" + str(len(REF_CHANNELS)) + "): "
      + str(REF_CHANNELS))

# -- Helpers ------------------------------------------------
def load_eeg(fif_path):
    """Same drop/resample/centering/reorder logic as the k=7
    pipeline's step-02 load_eeg(), plus the same optional band
    filter step 01 applied before GFP-peak extraction."""
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
    if BAND == "theta":
        raw.filter(THETA_LO, THETA_HI, picks="eeg",
                   method="fir", phase="zero", verbose=False)
    missing = [ch for ch in REF_CHANNELS if ch not in raw.ch_names]
    if missing:
        raise ValueError("Run missing template channels: " + str(missing))
    raw.reorder_channels(REF_CHANNELS)
    data = (raw.get_data() * 1e6).astype(np.float32)
    data -= data.mean(axis=1, keepdims=True)
    return data

def rle(labels):
    """Run-length encode an integer label sequence."""
    n = len(labels)
    if n == 0:
        return (np.array([], dtype=labels.dtype),
                np.array([], dtype=int), np.array([], dtype=int))
    change = np.where(np.diff(labels) != 0)[0] + 1
    starts = np.concatenate(([0], change))
    ends   = np.concatenate((change, [n]))
    return labels[starts], starts, ends - starts

def smooth_labels(labels, min_len=MIN_SEG_SAMPLES, max_passes=50):
    """Same majority-merge temporal-smoothing rule as the k=7
    pipeline (Pascual-Marqui 1995 minimum-duration criterion,
    simplified merge-to-longer-neighbor variant)."""
    labels = labels.copy()
    for _ in range(max_passes):
        labs, starts, lens = rle(labels)
        if len(labs) <= 1:
            break
        short = np.where(lens < min_len)[0]
        if len(short) == 0:
            break
        changed = False
        for si in short:
            prev_len = lens[si - 1] if si > 0 else -1
            next_len = lens[si + 1] if si < len(labs) - 1 else -1
            if prev_len < 0 and next_len < 0:
                continue
            new_lab = labs[si - 1] if prev_len >= next_len else labs[si + 1]
            labels[starts[si]:starts[si] + lens[si]] = new_lab
            changed = True
        if not changed:
            break
    return labels

# -- Back-fit every run onto the templates -------------------
print()
print("=" * 55)
print("Back-fitting continuous EEG (band=" + BAND + ") onto "
      + str(N_MICROSTATES) + " templates  (" + TAG + ")")
print("=" * 55)

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

        eeg = load_eeg(fif)
        n_samples = eeg.shape[1]

        gfp    = eeg.std(axis=0).astype(np.float32)
        gfp_sq = (gfp.astype(np.float64) ** 2)

        norms = np.linalg.norm(eeg, axis=0)
        norms_safe = np.where(norms > 1e-10, norms, 1.0).astype(np.float32)
        eeg_norm = eeg / norms_safe
        corr_matrix = np.abs(templates @ eeg_norm)   # (N_MICROSTATES, n_samples)
        labels_raw = corr_matrix.argmax(axis=0)

        labels_smooth = smooth_labels(labels_raw)

        labs_f, starts_f, lens_f = rle(labels_smooth)
        duration_s = n_samples / SFREQ

        for k in range(N_MICROSTATES):
            seg_mask = labs_f == k
            seg_lens = lens_f[seg_mask]
            n_segs_k = len(seg_lens)
            coverage_pct = 100.0 * float(seg_lens.sum()) / n_samples \
                           if n_samples > 0 else 0.0
            if n_segs_k > 0:
                duration_ms = float(seg_lens.mean()) / SFREQ * 1000.0
                occurrence_hz = n_segs_k / duration_s
            else:
                duration_ms = None
                occurrence_hz = 0.0
            samp_mask = labels_smooth == k
            if samp_mask.sum() > 0 and gfp_sq.sum() > 0:
                gev_frac = float((corr_matrix[k, samp_mask].astype(np.float64) ** 2
                                  * gfp_sq[samp_mask]).sum() / gfp_sq.sum())
            else:
                gev_frac = 0.0
            rows.append(dict(
                subject=subject, run=run, condition=condition,
                microstate=MS_LABELS[k],
                duration_ms=round(duration_ms, 3) if duration_ms is not None else "",
                occurrence_hz=round(occurrence_hz, 4),
                coverage_pct=round(coverage_pct, 3),
                gev_pct=round(gev_frac * 100.0, 3),
            ))

        cov_sum = sum(r["coverage_pct"] for r in rows[-N_MICROSTATES:])
        n_ok += 1
        print("  " + subject + "  run-" + run + "  (" + condition + ")"
              + "  samples=" + str(n_samples)
              + "  segs=" + str(len(labs_f))
              + "  coverage_sum=" + "{:.1f}".format(cov_sum) + "%")

print()
print("Runs processed: " + str(n_ok) + "  missing: " + str(n_missing))

# -- Write CSV ------------------------------------------------
out_csv = RESULTS_DIR / ("temporal_params_" + TAG + ".csv")
fields = ["subject", "run", "condition", "microstate",
          "duration_ms", "occurrence_hz", "coverage_pct", "gev_pct"]
with open(str(out_csv), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=fields)
    w.writeheader()
    w.writerows(rows)
print("Saved: " + str(out_csv) + "  (" + str(len(rows)) + " rows)")

print()
print("DONE  " + TAG)