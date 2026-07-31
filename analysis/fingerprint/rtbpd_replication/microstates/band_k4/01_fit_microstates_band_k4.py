# 01_fit_microstates_band_k4.py
# Run locally: python 01_fit_microstates_band_k4.py --band fullspectrum
#              python 01_fit_microstates_band_k4.py --band theta
# Deploys cluster script, submits SLURM job, monitors.
#
# READ-ONLY reuse of the Phase 1 preprocessed rest FIFs (no new cluster
# preprocessing). Adapted from ../01_fit_microstates_rtbpd.py:
#   - k=4 (classic Koenig et al. 2002 A/B/C/D taxonomy) instead of k=7
#   - canonical network-matching dict replaced with A/B/C/D signatures
#   - "theta" band applies a zero-phase FIR 4-8 Hz bandpass (MNE
#     raw.filter(4., 8., phase="zero")) to the continuous EEG BEFORE GFP
#     peak extraction; "fullspectrum" applies no extra filter (same
#     broadband signal as the k=7 pipeline).
#   - Same GFP-peak extraction, same polarity-invariant k-means, same 15
#     subjects / 4 pooled runs, same EXCLUDE_CHS as the k=7 pipeline.

import argparse
import py_compile
import time
from pathlib import Path
from utils_cluster_band import run_ssh, scp_to, make_cluster_dirs
from config_band_k4 import (
    CLUSTER_BASE, SLURM_ACCOUNT, PYTHON,
    SUBJECTS, EEG_ROOT, SESSION, ALL_RUNS,
    SFREQ, SFREQ_TAG, EEG_DESC,
    N_MICROSTATES, N_KMEANS_RESTARTS,
    KMEANS_MAX_ITER, GFP_OUTLIER_SD, EXCLUDE_CHS,
    BANDS, THETA_BAND, SCRIPTS_DIR,
)

parser = argparse.ArgumentParser()
parser.add_argument("--band", required=True, choices=BANDS)
parser.add_argument("--time-budget", default="08:00:00",
                    help="SLURM --time (default: 08:00:00)")
args = parser.parse_args()
BAND = args.band
TIME_BUDGET = args.time_budget
TAG = BAND + "_k4_" + SFREQ_TAG

# ── 1. Build cluster-side script ───────────────────────────
lines = [
    '#!/usr/bin/env python3',
    '"""',
    '01_fit_microstates_band_k4_cluster_' + BAND + '.py',
    'Fit 4 microstate templates (classic A/B/C/D taxonomy, Koenig et al.',
    '2002 / Michel & Koenig 2018) on pooled rtBPD rest EEG, band = ' + BAND + '.',
    'Reuses the Phase 1 preprocessed FIFs read-only -- no new preprocessing.',
    '"""',
    'import sys',
    'sys.stdout.reconfigure(line_buffering=True)',
    'import numpy as np',
    'import json',
    'from pathlib import Path',
    'import mne',
    'from scipy.signal import argrelmax',
    '',
    '# -- Paths and constants ---------------------------------',
    'EEG_ROOT      = Path("' + EEG_ROOT + '")',
    'OUT_DIR       = Path("' + CLUSTER_BASE + '")',
    'SUBJECTS      = ' + repr(SUBJECTS),
    'SESSION       = "' + SESSION + '"',
    'ALL_RUNS      = ' + repr(ALL_RUNS),
    'SFREQ         = ' + repr(SFREQ),
    'SFREQ_TAG     = "' + SFREQ_TAG + '"',
    'EEG_DESC      = "' + EEG_DESC + '"',
    'N_MICROSTATES = ' + repr(N_MICROSTATES),
    'N_RESTARTS    = ' + repr(N_KMEANS_RESTARTS),
    'MAX_ITER      = ' + repr(KMEANS_MAX_ITER),
    'OUTLIER_SD    = ' + repr(GFP_OUTLIER_SD),
    'EXCLUDE_CHS   = ' + repr(EXCLUDE_CHS),
    'BAND          = "' + BAND + '"',
    'THETA_LO      = ' + repr(THETA_BAND[0]),
    'THETA_HI      = ' + repr(THETA_BAND[1]),
    'TAG           = "' + TAG + '"',
    '',
    '# -- Helpers ----------------------------------------------',
    'def load_eeg(fif_path):',
    '    """Same drop/resample/centering logic as the k=7 pipeline\'s',
    '    load_eeg(), plus an optional zero-phase FIR 4-8 Hz bandpass',
    '    (BAND=="theta") applied to the continuous signal BEFORE any',
    '    GFP-peak extraction / k-means. BAND=="fullspectrum" applies no',
    '    extra filter (identical broadband signal to the k=7 pipeline)."""',
    '    raw = mne.io.read_raw_fif(str(fif_path), preload=True,',
    '                               verbose=False)',
    '    drop = [ch for ch in raw.ch_names',
    '            if any(x in ch.upper() for x in',
    '                   ("ECG","EKG","EMG","EOG","STIM","STATUS"))',
    '            or ch in EXCLUDE_CHS]',
    '    if drop:',
    '        raw.drop_channels(drop)',
    '    if raw.info["sfreq"] != SFREQ:',
    '        raw.resample(SFREQ, verbose=False)',
    '    if BAND == "theta":',
    '        raw.filter(THETA_LO, THETA_HI, picks="eeg",',
    '                   method="fir", phase="zero", verbose=False)',
    '    data = (raw.get_data() * 1e6).astype(np.float32)',
    '    data -= data.mean(axis=1, keepdims=True)',
    '    return data, list(raw.ch_names)',
    '',
    'def compute_gfp(eeg):',
    '    return eeg.std(axis=0).astype(np.float32)',
    '',
    'def get_gfp_peaks(gfp):',
    '    peaks = argrelmax(gfp, order=1)[0]',
    '    mu    = gfp[peaks].mean()',
    '    sig   = gfp[peaks].std()',
    '    peaks = peaks[gfp[peaks] < mu + OUTLIER_SD * sig]',
    '    return peaks',
    '',
    'def normalize_map(m):',
    '    n = np.linalg.norm(m)',
    '    return m / n if n > 1e-10 else m',
    '',
    '# -- Load all rest GFP peaks (pooled across all 4 runs) ----',
    'print("=" * 55)',
    'print("1. Loading rest EEG GFP peaks  band=" + BAND + "  (" + SFREQ_TAG + ")")',
    'print("   Runs pooled: " + str(ALL_RUNS) + "  (pre+post)")',
    'print("   Excluding: " + str(EXCLUDE_CHS)',
    '      + "  (+ ECG/EKG/EMG/EOG/STIM/STATUS by substring)")',
    'if BAND == "theta":',
    '    print("   Bandpass: " + str(THETA_LO) + "-" + str(THETA_HI) + " Hz (zero-phase FIR)")',
    'print("=" * 55)',
    '',
    'all_peaks = []',
    'ref_ch_names = None',
    'n_loaded  = 0',
    'n_missing = 0',
    '',
    'for subject in SUBJECTS:',
    '    for run in ALL_RUNS:',
    '        fname = (subject + "_" + SESSION + "_task-rest"',
    '                 + "_run-" + run',
    '                 + "_desc-" + EEG_DESC + "_eeg.fif")',
    '        fif = (EEG_ROOT / subject / SESSION / "eeg" / fname)',
    '        if not fif.exists():',
    '            print("  MISSING: " + fname)',
    '            n_missing += 1',
    '            continue',
    '        print("  Loading: " + subject + "  run-" + run)',
    '        eeg, ch_names = load_eeg(fif)',
    '        if ref_ch_names is None:',
    '            ref_ch_names = ch_names',
    '        gfp  = compute_gfp(eeg)',
    '        pksi = get_gfp_peaks(gfp)',
    '        maps = eeg[:, pksi].T',
    '        maps = np.array([normalize_map(m) for m in maps])',
    '        all_peaks.append(maps)',
    '        n_loaded += 1',
    '        print("    ch=" + str(eeg.shape[0])',
    '              + "  samples=" + str(eeg.shape[1])',
    '              + "  peaks=" + str(len(pksi)))',
    '',
    'if len(all_peaks) == 0:',
    '    print("ERROR: no EEG files found")',
    '    sys.exit(1)',
    '',
    'all_peaks = np.concatenate(all_peaks, axis=0)',
    'print()',
    'print("Runs loaded:  " + str(n_loaded)',
    '      + "  (expected " + str(len(SUBJECTS) * len(ALL_RUNS)) + ")")',
    'print("Runs missing: " + str(n_missing))',
    'print("Total peaks:  " + str(len(all_peaks)))',
    'print("Map shape:    " + str(all_peaks.shape))',
    '',
    '# -- Polarity-invariant k-means ----------------------------',
    'print()',
    'print("=" * 55)',
    'print("2. Fitting k-means  k=" + str(N_MICROSTATES)',
    '      + "  restarts=" + str(N_RESTARTS) + "  band=" + BAND)',
    'print("=" * 55)',
    '',
    'rng = np.random.default_rng(42)',
    'best_gev       = -1.0',
    'best_templates = None',
    '',
    'for restart in range(N_RESTARTS):',
    '    idx  = rng.choice(len(all_peaks), N_MICROSTATES, replace=False)',
    '    maps = all_peaks[idx].copy()',
    '',
    '    prev_labels = None',
    '    for iteration in range(MAX_ITER):',
    '        corrs  = np.abs(all_peaks @ maps.T)',
    '        labels = corrs.argmax(axis=1)',
    '',
    '        new_maps = np.zeros_like(maps)',
    '        for k in range(N_MICROSTATES):',
    '            members = all_peaks[labels == k].copy()',
    '            if len(members) == 0:',
    '                new_maps[k] = all_peaks[rng.integers(len(all_peaks))]',
    '                continue',
    '            ref = members[0]',
    '            for i in range(len(members)):',
    '                if np.dot(members[i], ref) < 0:',
    '                    members[i] = -members[i]',
    '            new_maps[k] = normalize_map(members.mean(axis=0))',
    '',
    '        if prev_labels is not None:',
    '            if np.all(labels == prev_labels):',
    '                break',
    '        prev_labels = labels.copy()',
    '        maps = new_maps',
    '',
    '    gfp_sq = (all_peaks ** 2).sum(axis=1)',
    '    gev    = 0.0',
    '    for k in range(N_MICROSTATES):',
    '        corr_k = np.abs(all_peaks[labels == k] @ maps[k])',
    '        if len(corr_k) > 0:',
    '            gev += float((corr_k ** 2',
    '                          * gfp_sq[labels == k]).sum()',
    '                         / gfp_sq.sum())',
    '',
    '    print("  restart " + str(restart + 1).zfill(2)',
    '          + "  iter=" + str(iteration)',
    '          + "  GEV=" + "{:.4f}".format(gev))',
    '',
    '    if gev > best_gev:',
    '        best_gev       = gev',
    '        best_templates = maps.copy()',
    '',
    '# -- Save templates + channel list -------------------------',
    'print()',
    'print("=" * 55)',
    'print("3. Saving templates  (" + TAG + ")")',
    'print("=" * 55)',
    '',
    'ms_dir = OUT_DIR / "microstates"',
    'ms_dir.mkdir(parents=True, exist_ok=True)',
    '',
    'out_templates = ms_dir / ("templates_" + TAG + ".npy")',
    'out_gev       = ms_dir / ("gev_"       + TAG + ".npy")',
    'out_channels  = ms_dir / ("channels_"  + TAG + ".json")',
    '',
    'np.save(str(out_templates), best_templates)',
    'np.save(str(out_gev),       np.array([best_gev]))',
    'with open(str(out_channels), "w") as f:',
    '    json.dump(ref_ch_names, f, indent=2)',
    '',
    'print("Templates: " + str(best_templates.shape))',
    'print("Best GEV:  " + "{:.4f}".format(best_gev))',
    'print("Channels:  " + str(len(ref_ch_names)) + "  " + str(ref_ch_names))',
    'print("Saved:     " + str(out_templates))',
    '',
    '# -- QC per-map stats ----------------------------------------',
    'print()',
    'print("=" * 55)',
    'print("4. Per-map stats")',
    'print("=" * 55)',
    '',
    'corrs  = np.abs(all_peaks @ best_templates.T)',
    'labels = corrs.argmax(axis=1)',
    'gfp_sq = (all_peaks ** 2).sum(axis=1)',
    '',
    'for k in range(N_MICROSTATES):',
    '    cov    = 100.0 * (labels == k).sum() / len(labels)',
    '    corr_k = np.abs(all_peaks[labels == k] @ best_templates[k])',
    '    gev_k  = float((corr_k ** 2',
    '                    * gfp_sq[labels == k]).sum()',
    '                   / gfp_sq.sum())',
    '    print("  MS" + chr(65 + k)',
    '          + "  coverage=" + "{:.1f}".format(cov) + "%"',
    '          + "  GEV="      + "{:.4f}".format(gev_k))',
    '',
    '# -- Classic A/B/C/D canonical matching -----------------------',
    'print()',
    'print("=" * 55)',
    'print("5. Classic microstate A/B/C/D matching (Koenig 2002 / Michel & Koenig 2018)")',
    'print("=" * 55)',
    '',
    'ch_names = ref_ch_names',
    'print("  Channels: " + str(len(ch_names)))',
    'print("  " + str(ch_names))',
    'print()',
    '',
    '# Canonical A/B/C/D signature vectors -- built from the QUALITATIVE',
    '# topographic descriptions in Michel & Koenig (2018) / Koenig et al.',
    '# (2002); neither publishes exact channel weight lists, so this is our',
    '# own approximation on the 29-channel rtBPD montage (a real',
    '# methodological choice, flagged here rather than silently assumed):',
    '#   A: right-frontal -> left-posterior/occipital diagonal',
    '#   B: left-frontal  -> right-posterior/occipital diagonal (mirror of A)',
    '#   C: anterior-posterior, midline Fz/FC-to-Pz/POz/Oz axis',
    '#   D: fronto-central positive maximum, posterior negative',
    '# (No "FCz" channel in this montage -- Fz/Cz/FC1/FC2 used as the',
    '# midline/fronto-central proxy instead.)',
    'canonical_signatures = {',
    '    "A": {"pos": ["F4","FC2","FC6","Fp2"],',
    '          "neg": ["P3","O1","CP5","P7"]},',
    '    "B": {"pos": ["F3","FC1","FC5","Fp1"],',
    '          "neg": ["P4","O2","CP6","P8"]},',
    '    "C": {"pos": ["Fz","FC1","FC2","Cz"],',
    '          "neg": ["Pz","POz","Oz","CP1","CP2"]},',
    '    "D": {"pos": ["Fz","Cz","FC1","FC2","FC5","FC6"],',
    '          "neg": ["P7","P8","O1","O2","POz"]},',
    '}',
    '',
    'def make_canonical_vec(sig, ch_names):',
    '    v = np.zeros(len(ch_names))',
    '    for ch in sig["pos"]:',
    '        if ch in ch_names:',
    '            v[ch_names.index(ch)] += 1.0',
    '    for ch in sig["neg"]:',
    '        if ch in ch_names:',
    '            v[ch_names.index(ch)] -= 1.0',
    '    n = np.linalg.norm(v)',
    '    return v / n if n > 1e-10 else v',
    '',
    'canonical_vecs = {name: make_canonical_vec(sig, ch_names)',
    '                  for name, sig in canonical_signatures.items()}',
    'canon_names    = list(canonical_vecs.keys())',
    '',
    'corr_matrix = np.zeros((N_MICROSTATES, len(canon_names)))',
    'for k in range(N_MICROSTATES):',
    '    t = best_templates[k]',
    '    for j, cname in enumerate(canon_names):',
    '        cvec = canonical_vecs[cname]',
    '        corr_matrix[k, j] = abs(np.corrcoef(t, cvec)[0, 1])',
    '',
    'print("  Correlation matrix (rows=our maps, cols=canonical A/B/C/D):")',
    'header = "       " + "".join(["{:>7}".format(n) for n in canon_names])',
    'print(header)',
    'for k in range(N_MICROSTATES):',
    '    row = "  MS" + chr(65+k) + "  "',
    '    row += "".join(["{:>7.3f}".format(corr_matrix[k,j])',
    '                     for j in range(len(canon_names))])',
    '    print(row)',
    'print()',
    '',
    'assigned_maps   = set()',
    'assigned_canons = set()',
    'assignment_list = []',
    'flat_idx = np.argsort(corr_matrix.ravel())[::-1]',
    'for idx in flat_idx:',
    '    k = idx // len(canon_names)',
    '    j = idx  % len(canon_names)',
    '    if k not in assigned_maps and j not in assigned_canons:',
    '        assigned_maps.add(k)',
    '        assigned_canons.add(j)',
    '        assignment_list.append((k, canon_names[j],',
    '                                float(corr_matrix[k, j])))',
    'assignment_list.sort(key=lambda x: x[0])',
    '',
    'print("  Final assignments:")',
    'label_map = {}',
    'for k, canon, corr in assignment_list:',
    '    label_map[k] = canon',
    '    print("  MS" + chr(65+k)',
    '          + " -> " + canon',
    '          + "  r=" + "{:.3f}".format(corr))',
    '',
    'assign_path = ms_dir / ("assignments_" + TAG + ".json")',
    'with open(str(assign_path), "w") as f:',
    '    json.dump({str(k): v for k, v in label_map.items()}, f, indent=2)',
    'print()',
    'print("  Saved: " + str(assign_path))',
    '',
    'print()',
    'print("DONE  " + TAG)',
]

# ── 2. Save cluster script locally ─────────────────────────
script_name = "01_fit_microstates_band_k4_" + TAG + "_cluster.py"
script_path = SCRIPTS_DIR / script_name
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

# ── 4. Deploy ──────────────────────────────────────────────
print("\nDeploying...")
make_cluster_dirs()
remote_script = CLUSTER_BASE + "/scripts/" + script_name
scp_to(script_path, remote_script, verbose=False)
print("Deployed: " + script_name)

# ── 5. Submit SLURM job ────────────────────────────────────
job_name = "rtbpd_fit_band_k4_" + BAND
sbatch_lines = [
    "#!/bin/bash",
    "#SBATCH --job-name=" + job_name,
    "#SBATCH --output=" + CLUSTER_BASE + "/logs/" + job_name + "_%j.out",
    "#SBATCH --error="  + CLUSTER_BASE + "/logs/" + job_name + "_%j.err",
    "#SBATCH --partition=short",
    "#SBATCH --time=" + TIME_BUDGET,
    "#SBATCH --cpus-per-task=4",
    "#SBATCH --mem=32G",
    "#SBATCH --account=" + SLURM_ACCOUNT,
    "",
    PYTHON + " " + CLUSTER_BASE + "/scripts/" + script_name,
]

sbatch_name = "01_fit_microstates_band_k4_" + TAG + ".sh"
sbatch_path = SCRIPTS_DIR / sbatch_name
with open(sbatch_path, "w") as f:
    f.write("\n".join(sbatch_lines))

remote_sbatch = CLUSTER_BASE + "/scripts/" + sbatch_name
scp_to(sbatch_path, remote_sbatch, verbose=False)

print("\nSubmitting SLURM job (" + TAG + ", time=" + TIME_BUDGET + ")...")
result = run_ssh("sbatch " + remote_sbatch)
job_id = ""
for line in result.stdout.strip().split("\n"):
    if "Submitted" in line:
        job_id = line.strip().split()[-1]
        print("Job ID: " + job_id)

# ── 6. Monitor ─────────────────────────────────────────────
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
                    "tail -40 " + CLUSTER_BASE
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
