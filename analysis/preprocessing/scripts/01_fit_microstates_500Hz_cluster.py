#!/usr/bin/env python3
"""
01_fit_microstates_cluster.py
Fit 7 microstate templates on pooled rest EEG.
Excludes TP9/TP10 (artifact-prone in simultaneous EEG-fMRI).
Matches fitted maps to canonical networks after fitting.
"""
import sys
sys.stdout.reconfigure(line_buffering=True)
import numpy as np
import json
import glob
from pathlib import Path
import mne
from scipy.signal import argrelmax

# ── Paths and constants ───────────────────────────────
EEG_ROOT      = Path("/projects/swglab/data/DMNELF/derivatives/eeg_preprocessed")
OUT_DIR       = Path("/projects/swglab/data/DMNELF/analysis/MNE/jupyter/microstate_pda_v3")
SUBJECTS      = ['sub-dmnelf001', 'sub-dmnelf004', 'sub-dmnelf005', 'sub-dmnelf006', 'sub-dmnelf007', 'sub-dmnelf008', 'sub-dmnelf009', 'sub-dmnelf010', 'sub-dmnelf011', 'sub-dmnelf1001', 'sub-dmnelf1002', 'sub-dmnelf1003']
SFREQ         = 500
SFREQ_TAG     = "500Hz"
N_MICROSTATES = 7
N_RESTARTS    = 20
MAX_ITER      = 1000
OUTLIER_SD    = 3.0
EXCLUDE_CHS   = ["TP9", "TP10"]

# ── Helpers ───────────────────────────────────────────
def load_eeg(fif_path):
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
    data = (raw.get_data() * 1e6).astype(np.float32)
    data -= data.mean(axis=1, keepdims=True)
    return data

def compute_gfp(eeg):
    return eeg.std(axis=0).astype(np.float32)

def get_gfp_peaks(gfp):
    peaks = argrelmax(gfp, order=1)[0]
    mu    = gfp[peaks].mean()
    sig   = gfp[peaks].std()
    peaks = peaks[gfp[peaks] < mu + OUTLIER_SD * sig]
    return peaks

def normalize_map(m):
    n = np.linalg.norm(m)
    return m / n if n > 1e-10 else m

# ── Load all rest GFP peaks ───────────────────────────
print("=" * 55)
print("1. Loading rest EEG GFP peaks  (" + SFREQ_TAG + ")")
print("   Excluding: " + str(EXCLUDE_CHS))
print("=" * 55)

all_peaks = []
n_loaded  = 0
n_missing = 0

for subject in SUBJECTS:
    for run in ["01", "02"]:
        fname = (subject + "_ses-dmnelf_task-rest"
                 + "_run-" + run
                 + "_desc-preproc" + SFREQ_TAG + "_eeg.fif")
        fif = (EEG_ROOT / subject / "ses-dmnelf" / "eeg" / fname)
        if not fif.exists():
            print("  MISSING: " + fname)
            n_missing += 1
            continue
        print("  Loading: " + subject + "  run-" + run)
        eeg  = load_eeg(fif)
        gfp  = compute_gfp(eeg)
        pksi = get_gfp_peaks(gfp)
        maps = eeg[:, pksi].T
        maps = np.array([normalize_map(m) for m in maps])
        all_peaks.append(maps)
        n_loaded += 1
        print("    ch=" + str(eeg.shape[0])
              + "  samples=" + str(eeg.shape[1])
              + "  peaks=" + str(len(pksi)))

if len(all_peaks) == 0:
    print("ERROR: no EEG files found")
    sys.exit(1)

all_peaks = np.concatenate(all_peaks, axis=0)
print()
print("Runs loaded:  " + str(n_loaded))
print("Runs missing: " + str(n_missing))
print("Total peaks:  " + str(len(all_peaks)))
print("Map shape:    " + str(all_peaks.shape))

# ── Polarity-invariant k-means ────────────────────────
print()
print("=" * 55)
print("2. Fitting k-means  k=" + str(N_MICROSTATES)
      + "  restarts=" + str(N_RESTARTS))
print("=" * 55)

best_gev       = -1.0
best_templates = None

for restart in range(N_RESTARTS):
    idx  = np.random.choice(len(all_peaks), N_MICROSTATES,
                             replace=False)
    maps = all_peaks[idx].copy()

    prev_labels = None
    for iteration in range(MAX_ITER):
        corrs  = np.abs(all_peaks @ maps.T)
        labels = corrs.argmax(axis=1)

        new_maps = np.zeros_like(maps)
        for k in range(N_MICROSTATES):
            members = all_peaks[labels == k].copy()
            if len(members) == 0:
                new_maps[k] = all_peaks[
                    np.random.randint(len(all_peaks))]
                continue
            ref = members[0]
            for i in range(len(members)):
                if np.dot(members[i], ref) < 0:
                    members[i] = -members[i]
            new_maps[k] = normalize_map(members.mean(axis=0))

        if prev_labels is not None:
            if np.all(labels == prev_labels):
                break
        prev_labels = labels.copy()
        maps = new_maps

    gfp_sq = (all_peaks ** 2).sum(axis=1)
    gev    = 0.0
    for k in range(N_MICROSTATES):
        corr_k = np.abs(all_peaks[labels == k] @ maps[k])
        if len(corr_k) > 0:
            gev += float((corr_k ** 2
                          * gfp_sq[labels == k]).sum()
                         / gfp_sq.sum())

    print("  restart " + str(restart + 1).zfill(2)
          + "  iter=" + str(iteration)
          + "  GEV=" + "{:.4f}".format(gev))

    if gev > best_gev:
        best_gev       = gev
        best_templates = maps.copy()

# ── Save templates ────────────────────────────────────
print()
print("=" * 55)
print("3. Saving templates")
print("=" * 55)

ms_dir = OUT_DIR / "microstates"
ms_dir.mkdir(parents=True, exist_ok=True)

out_templates = ms_dir / ("templates_" + SFREQ_TAG + ".npy")
out_gev       = ms_dir / ("gev_"       + SFREQ_TAG + ".npy")

np.save(str(out_templates), best_templates)
np.save(str(out_gev),       np.array([best_gev]))

print("Templates: " + str(best_templates.shape))
print("Best GEV:  " + "{:.4f}".format(best_gev))
print("Saved:     " + str(out_templates))

# ── QC per-map stats ──────────────────────────────────
print()
print("=" * 55)
print("4. Per-map stats")
print("=" * 55)

corrs  = np.abs(all_peaks @ best_templates.T)
labels = corrs.argmax(axis=1)
gfp_sq = (all_peaks ** 2).sum(axis=1)

for k in range(N_MICROSTATES):
    cov    = 100.0 * (labels == k).sum() / len(labels)
    corr_k = np.abs(all_peaks[labels == k] @ best_templates[k])
    gev_k  = float((corr_k ** 2
                    * gfp_sq[labels == k]).sum()
                   / gfp_sq.sum())
    print("  MS" + chr(65 + k)
          + "  coverage=" + "{:.1f}".format(cov) + "%"
          + "  GEV="      + "{:.4f}".format(gev_k))

# ── Canonical network matching ────────────────────────
print()
print("=" * 55)
print("5. Canonical network matching (Custo 2017)")
print("=" * 55)

# Get channel names from first available FIF
fif_pattern = (str(EEG_ROOT) + "/sub-dmnelf001/ses-dmnelf/eeg/"
               + "sub-dmnelf001_ses-dmnelf_task-rest_run-01"
               + "_desc-preproc" + SFREQ_TAG + "_eeg.fif")
fifs = glob.glob(fif_pattern)
if not fifs:
    print("  WARNING: could not find reference FIF for channel names")
else:
    raw_ref  = mne.io.read_raw_fif(fifs[0], preload=False,
                                    verbose=False)
    ch_names = [ch for ch in raw_ref.ch_names
                if "ECG" not in ch.upper()
                and ch not in EXCLUDE_CHS]
    print("  Channels: " + str(len(ch_names)))
    print("  " + str(ch_names))
    print()

    # Canonical signatures — key channels per network
    # Based on Custo 2017, Britz 2010, Michel & Koenig 2018
    canonical_signatures = {
        "DMN":  {"pos": ["Pz","POz","P3","P4","CP1","CP2"],
                 "neg": ["Fz","F3","F4","FC1","FC2","Fp1","Fp2"]},
        "CEN":  {"pos": ["F4","FC2","FC6","P4","CP2"],
                 "neg": ["F3","FC1","FC5","P3","CP1"]},
        "VIS":  {"pos": ["Oz","O1","O2","POz","Pz"],
                 "neg": ["Fz","Fp1","Fp2","F3","F4"]},
        "SOM":  {"pos": ["Cz","C3","C4","CP1","CP2"],
                 "neg": ["Oz","O1","O2","Fp1","Fp2"]},
        "SAL":  {"pos": ["Fz","FC1","FC2","Cz","F3","F4"],
                 "neg": ["Pz","POz","P3","P4","Oz"]},
        "AUD":  {"pos": ["T7","T8","C3","C4","F7","F8"],
                 "neg": ["Pz","POz","Fz","Fp1","Fp2"]},
        "DANT": {"pos": ["Pz","P3","P4","Cz","CP1","CP2"],
                 "neg": ["Fp1","Fp2","F7","F8","Fz"]},
    }

    def make_canonical_vec(sig, ch_names):
        v = np.zeros(len(ch_names))
        for ch in sig["pos"]:
            if ch in ch_names:
                v[ch_names.index(ch)] += 1.0
        for ch in sig["neg"]:
            if ch in ch_names:
                v[ch_names.index(ch)] -= 1.0
        n = np.linalg.norm(v)
        return v / n if n > 1e-10 else v

    canonical_vecs = {name: make_canonical_vec(sig, ch_names)
                      for name, sig in canonical_signatures.items()}
    canon_names    = list(canonical_vecs.keys())

    # Build correlation matrix (n_maps x n_canonical)
    corr_matrix = np.zeros((N_MICROSTATES, len(canon_names)))
    for k in range(N_MICROSTATES):
        t = best_templates[k]
        for j, cname in enumerate(canon_names):
            cvec = canonical_vecs[cname]
            # Polarity-invariant correlation
            corr_matrix[k, j] = abs(np.corrcoef(t, cvec)[0, 1])

    # Print full correlation matrix
    print("  Correlation matrix (rows=our maps, cols=canonical):")
    header = "       " + "".join(["{:>7}".format(n)
                                   for n in canon_names])
    print(header)
    for k in range(N_MICROSTATES):
        row = "  MS" + chr(65+k) + "  "
        row += "".join(["{:>7.3f}".format(corr_matrix[k,j])
                         for j in range(len(canon_names))])
        print(row)
    print()

    # Greedy assignment — best match first
    assigned_maps   = set()
    assigned_canons = set()
    assignment_list = []
    flat_idx = np.argsort(corr_matrix.ravel())[::-1]
    for idx in flat_idx:
        k = idx // len(canon_names)
        j = idx  % len(canon_names)
        if k not in assigned_maps and j not in assigned_canons:
            assigned_maps.add(k)
            assigned_canons.add(j)
            assignment_list.append((k, canon_names[j],
                                    float(corr_matrix[k, j])))
    assignment_list.sort(key=lambda x: x[0])

    print("  Final assignments:")
    label_map = {}
    for k, canon, corr in assignment_list:
        label_map[k] = canon
        print("  MS" + chr(65+k)
              + " -> " + canon
              + "  r=" + "{:.3f}".format(corr))

    # Save assignments JSON
    assign_path = ms_dir / ("assignments_" + SFREQ_TAG + ".json")
    with open(str(assign_path), "w") as f:
        json.dump({str(k): v for k, v in label_map.items()},
                  f, indent=2)
    print()
    print("  Saved: " + str(assign_path))

print()
print("DONE  " + SFREQ_TAG)