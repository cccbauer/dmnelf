# 01_compute_band_power.py
# Run locally: python 01_compute_band_power.py
# Deploys cluster script, submits SLURM job, monitors.
#
# READ-ONLY reuse of the Phase 1 preprocessed rest FIFs (no new cluster
# preprocessing, no touching the microstate pipelines). For each of the 15
# subjects x 4 runs: load the FIF, drop TP9/TP10/ECG (same as the
# microstate pipelines' EXCLUDE_CHS + substring logic), compute Welch PSD
# per channel (mne Raw.compute_psd(method="welch"), 4s windows/50%
# overlap), average power across the 29 channels within each of the 5
# standard bands (delta/theta/alpha/beta/gamma, identical edges to
# rtbpd_replication/config.yaml), then log-transform (10*log10) the
# channel-averaged linear power to get one dB scalar per (subject, run,
# band). This is a single modest job -- no k-means, so no SLURM array
# needed.

import py_compile
import time
from pathlib import Path
from utils_cluster_bp import run_ssh, scp_to, make_cluster_dirs
from config_band_power import (
    CLUSTER_BASE, SLURM_ACCOUNT, PYTHON,
    SUBJECTS, EEG_ROOT, SESSION, ALL_RUNS, PRE_RUNS, POST_RUNS,
    SFREQ, SFREQ_TAG, EEG_DESC, EXCLUDE_CHS,
    WELCH_WIN_SEC, WELCH_OVERLAP_PCT, BANDS, SCRIPTS_DIR,
)

# ── 1. Build cluster-side script ───────────────────────────
lines = [
    '#!/usr/bin/env python3',
    '"""',
    '01_compute_band_power_cluster.py',
    'Welch PSD -> 5 standard band powers (dB), averaged across 29 EEG',
    'channels, for each of the 15 rtBPD nf1 subjects x 4 rest runs.',
    '"""',
    'import sys',
    'sys.stdout.reconfigure(line_buffering=True)',
    'import numpy as np',
    'import csv',
    'from pathlib import Path',
    'import mne',
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
    'WELCH_WIN_SEC = ' + repr(WELCH_WIN_SEC),
    'WELCH_OVERLAP = ' + repr(WELCH_OVERLAP_PCT),
    'BANDS         = ' + repr(BANDS),
    '',
    '# -- Helpers ----------------------------------------------',
    'def load_eeg_raw(fif_path):',
    '    """Same drop/resample logic as the microstate pipelines\' load_eeg(),',
    '    but returns the MNE Raw object itself (compute_psd needs it),',
    '    rather than a bare numpy array."""',
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
    '# -- Compute band power for every run -------------------------',
    'print("=" * 55)',
    'print("Welch PSD -> band power (dB), " + str(len(SUBJECTS))',
    '      + " subjects x " + str(len(ALL_RUNS)) + " runs")',
    'print("Excluding: " + str(EXCLUDE_CHS)',
    '      + "  (+ ECG/EKG/EMG/EOG/STIM/STATUS by substring)")',
    'print("Welch window: " + str(WELCH_WIN_SEC) + "s  overlap="',
    '      + str(int(WELCH_OVERLAP * 100)) + "%")',
    'print("Bands: " + str(BANDS))',
    'print("=" * 55)',
    '',
    'n_per_seg = int(WELCH_WIN_SEC * SFREQ)',
    'n_overlap = int(WELCH_OVERLAP * n_per_seg)',
    '',
    'rows = []',
    'n_ok = 0',
    'n_missing = 0',
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
    '        n_ch = len(mne.pick_types(raw.info, eeg=True))',
    '',
    '        # Welch PSD per channel: (n_channels, n_freqs), linear power',
    '        # (V^2/Hz on the microvolt-scaled data MNE stores internally',
    '        # in volts -- absolute scale does not matter here since we',
    '        # only ever compare pre vs post on the same scale).',
    '        #',
    '        # reject_by_annotation=False: some runs carry a small trailing',
    '        # BAD_edge_end annotation from Phase 1 preprocessing (edge-ramp',
    '        # artifact, typically <0.1% of samples). MNE\'s default',
    '        # reject_by_annotation=True excises that span and Welch-PSDs',
    '        # the tiny leftover tail, which can be shorter than the 4s',
    '        # window and crash ("noverlap must be less than nperseg").',
    '        # The microstate pipeline never applies annotation-based',
    '        # rejection either -- it treats the full continuous recording',
    '        # as valid -- so we match that same approach here.',
    '        spectrum = raw.compute_psd(',
    '            method="welch", picks="eeg",',
    '            fmin=0.5, fmax=45.0,',
    '            n_fft=n_per_seg, n_per_seg=n_per_seg, n_overlap=n_overlap,',
    '            reject_by_annotation=False,',
    '            verbose=False,',
    '        )',
    '        psd = spectrum.get_data()   # (n_ch, n_freqs), linear power',
    '        freqs = spectrum.freqs',
    '',
    '        for band, (lo, hi) in BANDS.items():',
    '            fmask = (freqs >= lo) & (freqs < hi)',
    '            # 1) mean over frequency bins within the band, per channel',
    '            band_power_per_ch = psd[:, fmask].mean(axis=1)',
    '            # 2) mean (linear) across the 29 channels',
    '            band_power_avg = float(band_power_per_ch.mean())',
    '            # 3) log-transform the channel-averaged linear power',
    '            log_power_db = 10.0 * np.log10(band_power_avg + 1e-30)',
    '            rows.append(dict(',
    '                subject=subject, run=run, condition=condition,',
    '                band=band, log_power_db=round(log_power_db, 4),',
    '            ))',
    '',
    '        n_ok += 1',
    '        print("  " + subject + "  run-" + run + "  (" + condition + ")"',
    '              + "  ch=" + str(n_ch)',
    '              + "  freqs=" + str(len(freqs)))',
    '',
    'print()',
    'print("Runs processed: " + str(n_ok) + "  missing: " + str(n_missing))',
    '',
    '# -- Write CSV ------------------------------------------------',
    'out_csv = RESULTS_DIR / "band_power.csv"',
    'fields = ["subject", "run", "condition", "band", "log_power_db"]',
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
script_name = "01_compute_band_power_cluster.py"
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
job_name = "rtbpd_band_power"
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

sbatch_name = "01_compute_band_power.sh"
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
