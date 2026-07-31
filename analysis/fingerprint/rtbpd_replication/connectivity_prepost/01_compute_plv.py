# 01_compute_plv.py
# Run locally: python 01_compute_plv.py
# Deploys cluster script, submits SLURM job, monitors.
#
# READ-ONLY reuse of the Phase 1 preprocessed rest FIFs (no new cluster
# preprocessing, no touching the microstate/band-power pipelines). For
# each of the 15 subjects x 4 rest runs: load the FIF, drop TP9/TP10/ECG
# (same as the other pipelines), bandpass into each of the 5 standard
# bands, Hilbert transform for instantaneous phase, compute whole-run PLV
# (|mean(exp(i*(phase_i - phase_j)))| over the FULL run duration -- no
# TR-blocking, no HRF convolution, both irrelevant for a pure rest EEG-EEG
# measure) for each pair in the curated DMN-relevant pair set, then
# average PLV across those pairs into one scalar per band per run.
#
# Bandpass/Hilbert/PLV formula and get_dmn_relevant_pairs() are ported
# (whole-run instead of TR-blocked, HRF step dropped) from the existing,
# validated DMNELF feedback-task code at
# wavelet_coupling/scripts/connectivity_features.py's compute_plv_run()
# and get_dmn_relevant_pairs() -- never previously run on rtBPD or used
# for a plain EEG-EEG connectivity result.
#
# Submitted directly on --partition=sharing (learned from the band-power
# run: `short` can get congested by other users' array jobs; `sharing`
# has a hard MaxTime=01:00:00 cap, well above what this job needs).

import py_compile
import time
from pathlib import Path
from utils_cluster_conn import run_ssh, scp_to, make_cluster_dirs
from config_connectivity import (
    CLUSTER_BASE, SLURM_ACCOUNT, PYTHON,
    SUBJECTS, EEG_ROOT, SESSION, ALL_RUNS, PRE_RUNS, POST_RUNS,
    SFREQ, EEG_DESC, EXCLUDE_CHS, BANDS, SCRIPTS_DIR,
)

# ── 1. Build cluster-side script ───────────────────────────
lines = [
    '#!/usr/bin/env python3',
    '"""',
    '01_compute_plv_cluster.py',
    'Whole-run PLV (phase-locking value) between curated DMN-relevant',
    'channel pairs, averaged across pairs, for each of the 15 rtBPD nf1',
    'subjects x 4 rest runs x 5 bands.',
    '"""',
    'import sys',
    'sys.stdout.reconfigure(line_buffering=True)',
    'import numpy as np',
    'import csv',
    'from pathlib import Path',
    'from itertools import combinations',
    'import mne',
    'from scipy.signal import hilbert',
    'mne.set_log_level("ERROR")',
    '',
    '# -- Paths and constants ---------------------------------',
    'EEG_ROOT      = Path("' + EEG_ROOT + '")',
    'OUT_DIR       = Path("' + CLUSTER_BASE + '")',
    'RESULTS_DIR   = OUT_DIR / "results"',
    'RESULTS_DIR.mkdir(parents=True, exist_ok=True)',
    'SUBJECTS      = ' + repr(SUBJECTS),
    'SESSION       = "' + SESSION + '"',
    'ALL_RUNS      = ' + repr(ALL_RUNS),
    'PRE_RUNS      = ' + repr(PRE_RUNS),
    'POST_RUNS     = ' + repr(POST_RUNS),
    'SFREQ         = ' + repr(SFREQ),
    'EEG_DESC      = "' + EEG_DESC + '"',
    'EXCLUDE_CHS   = ' + repr(EXCLUDE_CHS),
    'BANDS         = ' + repr(BANDS),
    '',
    '# -- Helpers ----------------------------------------------',
    'def load_eeg_raw(fif_path):',
    '    """Same drop/resample logic as the other pipelines\' load_eeg()."""',
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
    '    return raw',
    '',
    'def get_dmn_relevant_pairs(ch_names):',
    '    """Select channel pairs relevant to DMN connectivity.',
    '    DMN involves frontal-posterior and midline connectivity.',
    '    Ported VERBATIM from wavelet_coupling/scripts/connectivity_features.py',
    '    (frontal/posterior/midline/temporal channel-name sets unchanged).',
    '    Note: TP9/TP10 are excluded from this montage (EXCLUDE_CHS), so',
    '    the "temporal" set below naturally reduces to just T7/T8 here --',
    '    the function\'s own `if t in ch_idx` guard already handles this,',
    '    no code change needed.',
    '',
    '    Frontal: Fp1, Fp2, F3, F4, Fz, FC1, FC2',
    '    Posterior: P3, P4, Pz, O1, O2, Oz, POz',
    '    Midline: Fz, Cz, Pz, Oz, POz',
    '    """',
    '    frontal = {"Fp1", "Fp2", "F3", "F4", "Fz", "FC1", "FC2", "F7", "F8"}',
    '    posterior = {"P3", "P4", "Pz", "O1", "O2", "Oz", "POz", "P7", "P8"}',
    '    midline = {"Fz", "Cz", "Pz", "Oz", "POz"}',
    '    temporal = {"T7", "T8", "TP9", "TP10"}',
    '',
    '    pairs = []',
    '    ch_idx = {ch: i for i, ch in enumerate(ch_names)}',
    '',
    '    # Frontal-posterior pairs (DMN long-range)',
    '    for f in frontal:',
    '        for p in posterior:',
    '            if f in ch_idx and p in ch_idx:',
    '                pairs.append((ch_idx[f], ch_idx[p]))',
    '',
    '    # Midline pairs (DMN midline axis)',
    '    midline_list = [ch for ch in midline if ch in ch_idx]',
    '    for a, b in combinations([ch_idx[ch] for ch in midline_list], 2):',
    '        pairs.append((a, b))',
    '',
    '    # Temporal-frontal pairs',
    '    for t in temporal:',
    '        for f in frontal:',
    '            if t in ch_idx and f in ch_idx:',
    '                pairs.append((ch_idx[t], ch_idx[f]))',
    '',
    '    return list(set(pairs))  # deduplicate',
    '',
    'def compute_whole_run_plv(raw, bands, pairs):',
    '    """Whole-run PLV per band, averaged across the curated pairs.',
    '    Adapted from connectivity_features.py\'s compute_plv_run(): same',
    '    bandpass -> Hilbert -> |mean(exp(i*phase_diff))| formula, but a',
    '    SINGLE PLV value over the full run instead of per-TR blocks, and',
    '    no HRF convolution (both TR-blocking and HRF are EEG-BOLD',
    '    alignment machinery, irrelevant for a pure rest EEG-EEG measure).',
    '    """',
    '    picks = mne.pick_types(raw.info, eeg=True, exclude=[])',
    '    data = raw.get_data(picks=picks)   # (n_ch, n_samples)',
    '    sfreq = raw.info["sfreq"]',
    '',
    '    band_plv = {}',
    '    for bname, (lo, hi) in bands.items():',
    '        filtered = mne.filter.filter_data(data, sfreq, lo, hi, verbose=False)',
    '        analytic = hilbert(filtered, axis=-1)',
    '        phase = np.angle(analytic)   # (n_ch, n_samples)',
    '',
    '        pair_plvs = []',
    '        for ch_a, ch_b in pairs:',
    '            phase_diff = phase[ch_a] - phase[ch_b]',
    '            plv = float(np.abs(np.mean(np.exp(1j * phase_diff))))',
    '            pair_plvs.append(plv)',
    '        band_plv[bname] = float(np.mean(pair_plvs))',
    '    return band_plv',
    '',
    '# -- Compute PLV for every run -------------------------------',
    'print("=" * 55)',
    'print("Whole-run PLV, " + str(len(SUBJECTS)) + " subjects x "',
    '      + str(len(ALL_RUNS)) + " runs x " + str(len(BANDS)) + " bands")',
    'print("Excluding: " + str(EXCLUDE_CHS)',
    '      + "  (+ ECG/EKG/EMG/EOG/STIM/STATUS by substring)")',
    'print("Bands: " + str(BANDS))',
    'print("=" * 55)',
    '',
    'rows = []',
    'n_ok = 0',
    'n_missing = 0',
    'n_pairs_reported = None',
    '',
    'for subject in SUBJECTS:',
    '    for run in ALL_RUNS:',
    '        condition = "pre" if run in PRE_RUNS else "post"',
    '        fname = (subject + "_" + SESSION + "_task-rest"',
    '                 + "_run-" + run',
    '                 + "_desc-" + EEG_DESC + "_eeg.fif")',
    '        fif = (EEG_ROOT / subject / SESSION / "eeg" / fname)',
    '        if not fif.exists():',
    '            print("  MISSING: " + fname)',
    '            n_missing += 1',
    '            continue',
    '',
    '        raw = load_eeg_raw(fif)',
    '        picks = mne.pick_types(raw.info, eeg=True, exclude=[])',
    '        ch_names = [raw.ch_names[i] for i in picks]',
    '        pairs = get_dmn_relevant_pairs(ch_names)',
    '        if n_pairs_reported is None:',
    '            n_pairs_reported = len(pairs)',
    '            print("  DMN-relevant pairs: " + str(len(pairs))',
    '                  + "  (from " + str(len(ch_names)) + " channels)")',
    '',
    '        band_plv = compute_whole_run_plv(raw, BANDS, pairs)',
    '        for band, plv in band_plv.items():',
    '            rows.append(dict(',
    '                subject=subject, run=run, condition=condition,',
    '                band=band, plv=round(plv, 5),',
    '            ))',
    '',
    '        n_ok += 1',
    '        print("  " + subject + "  run-" + run + "  (" + condition + ")"',
    '              + "  n_samples=" + str(raw.n_times)',
    '              + "  plv=" + str({k: round(v, 3) for k, v in band_plv.items()}))',
    '',
    'print()',
    'print("Runs processed: " + str(n_ok) + "  missing: " + str(n_missing))',
    '',
    '# -- Write CSV ------------------------------------------------',
    'out_csv = RESULTS_DIR / "plv_connectivity.csv"',
    'fields = ["subject", "run", "condition", "band", "plv"]',
    'with open(str(out_csv), "w", newline="") as f:',
    '    w = csv.DictWriter(f, fieldnames=fields)',
    '    w.writeheader()',
    '    w.writerows(rows)',
    'print("Saved: " + str(out_csv) + "  (" + str(len(rows)) + " rows)")',
    '',
    'print()',
    'print("DONE")',
]

# ── 2. Save cluster script locally ─────────────────────────
script_name = "01_compute_plv_cluster.py"
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
job_name = "rtbpd_plv"
sbatch_lines = [
    "#!/bin/bash",
    "#SBATCH --job-name=" + job_name,
    "#SBATCH --output=" + CLUSTER_BASE + "/logs/" + job_name + "_%j.out",
    "#SBATCH --error="  + CLUSTER_BASE + "/logs/" + job_name + "_%j.err",
    "#SBATCH --partition=sharing",
    "#SBATCH --time=00:55:00",
    "#SBATCH --cpus-per-task=2",
    "#SBATCH --mem=16G",
    "#SBATCH --account=" + SLURM_ACCOUNT,
    "",
    PYTHON + " " + CLUSTER_BASE + "/scripts/" + script_name,
]

sbatch_name = "01_compute_plv.sh"
sbatch_path = SCRIPTS_DIR / sbatch_name
with open(sbatch_path, "w") as f:
    f.write("\n".join(sbatch_lines))

remote_sbatch = CLUSTER_BASE + "/scripts/" + sbatch_name
scp_to(sbatch_path, remote_sbatch, verbose=False)

print("\nSubmitting SLURM job...")
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
