# 02_temporal_params_rtbpd.py
# Run locally: python 02_temporal_params_rtbpd.py
# Deploys cluster script, submits SLURM job, monitors.
#
# What it does on the cluster:
#   - Loads the 7 templates fit by 01_fit_microstates_rtbpd.py (templates_
#     500Hz.npy) plus their canonical-network labels (assignments_500Hz.json)
#     and the retained channel-name list (channels_500Hz.json).
#   - For each subject x each of the 4 rest runs: loads the FULL continuous
#     EEG (not just GFP peaks), back-fits every timepoint onto the 7
#     templates via polarity-invariant correlation (each timepoint's
#     topography L2-normalized the same way GFP-peak maps were normalized
#     in step 01, then argmax |corr| over the 7 templates -> raw label).
#   - Applies standard microstate temporal smoothing (see smooth_labels()
#     docstring below for the exact rule + citation).
#   - Computes per-run, per-microstate: mean duration (ms), occurrence rate
#     (segments/s), coverage (%), GEV (%).
#   - Writes one row per (subject, run, microstate) to
#     {CLUSTER_BASE}/results/temporal_params.csv

import py_compile
import time
from pathlib import Path
from utils_cluster import run_ssh, scp_to, make_cluster_dirs
from config_rtbpd_ms import (
    CLUSTER_BASE, SLURM_ACCOUNT, PYTHON,
    SUBJECTS, EEG_ROOT, SESSION, ALL_RUNS, PRE_RUNS, POST_RUNS,
    SFREQ, SFREQ_TAG, EEG_DESC,
    N_MICROSTATES, EXCLUDE_CHS,
    SCRIPTS_DIR,
)

# ── 1. Build cluster-side script ───────────────────────────
lines = [
    '#!/usr/bin/env python3',
    '"""',
    '02_temporal_params_rtbpd_cluster.py',
    'Back-fit continuous rest EEG onto the 7 templates fit by step 01 and',
    'compute classic microstate temporal parameters (duration, occurrence,',
    'coverage, GEV) per subject x run x microstate.',
    '"""',
    'import sys',
    'sys.stdout.reconfigure(line_buffering=True)',
    'import numpy as np',
    'import json',
    'import csv',
    'from pathlib import Path',
    'import mne',
    '',
    '# -- Paths and constants ---------------------------------',
    'EEG_ROOT      = Path("' + EEG_ROOT + '")',
    'OUT_DIR       = Path("' + CLUSTER_BASE + '")',
    'MS_DIR        = OUT_DIR / "microstates"',
    'RESULTS_DIR   = OUT_DIR / "results"',
    'RESULTS_DIR.mkdir(parents=True, exist_ok=True)',
    'SUBJECTS      = ' + repr(SUBJECTS),
    'SESSION       = "' + SESSION + '"',
    'ALL_RUNS      = ' + repr(ALL_RUNS),
    'PRE_RUNS      = ' + repr(PRE_RUNS),
    'POST_RUNS     = ' + repr(POST_RUNS),
    'SFREQ         = ' + repr(SFREQ),
    'SFREQ_TAG     = "' + SFREQ_TAG + '"',
    'EEG_DESC      = "' + EEG_DESC + '"',
    'N_MICROSTATES = ' + repr(N_MICROSTATES),
    'EXCLUDE_CHS   = ' + repr(EXCLUDE_CHS),
    'MIN_SEG_SAMPLES = 3   # Pascual-Marqui (1995) minimum-duration criterion',
    '',
    '# -- Load templates + labels + reference channel order ----',
    'templates = np.load(str(MS_DIR / ("templates_" + SFREQ_TAG + ".npy")))',
    'assign_path = MS_DIR / ("assignments_" + SFREQ_TAG + ".json")',
    'if assign_path.exists():',
    '    with open(str(assign_path)) as f:',
    '        _asgn = json.load(f)',
    '    MS_LABELS = [_asgn.get(str(i), "MS" + chr(65 + i))',
    '                 for i in range(N_MICROSTATES)]',
    'else:',
    '    MS_LABELS = ["MS" + chr(65 + i) for i in range(N_MICROSTATES)]',
    'print("Microstate labels:", MS_LABELS)',
    '',
    'chan_path = MS_DIR / ("channels_" + SFREQ_TAG + ".json")',
    'with open(str(chan_path)) as f:',
    '    REF_CHANNELS = json.load(f)',
    'print("Reference channels (" + str(len(REF_CHANNELS)) + "): "',
    '      + str(REF_CHANNELS))',
    '',
    '# -- Helpers ------------------------------------------------',
    'def load_eeg(fif_path):',
    '    """Identical drop/resample/centering logic to step 01\'s',
    '    load_eeg(), then reordered to match REF_CHANNELS so every run is',
    '    guaranteed aligned to the template channel order regardless of',
    '    the FIF\'s native channel order."""',
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
    '    missing = [ch for ch in REF_CHANNELS if ch not in raw.ch_names]',
    '    if missing:',
    '        raise ValueError("Run missing template channels: " + str(missing))',
    '    raw.reorder_channels(REF_CHANNELS)',
    '    data = (raw.get_data() * 1e6).astype(np.float32)',
    '    data -= data.mean(axis=1, keepdims=True)',
    '    return data',
    '',
    'def rle(labels):',
    '    """Run-length encode an integer label sequence.',
    '    Returns (labels_per_segment, start_indices, segment_lengths)."""',
    '    n = len(labels)',
    '    if n == 0:',
    '        return (np.array([], dtype=labels.dtype),',
    '                np.array([], dtype=int), np.array([], dtype=int))',
    '    change = np.where(np.diff(labels) != 0)[0] + 1',
    '    starts = np.concatenate(([0], change))',
    '    ends   = np.concatenate((change, [n]))',
    '    return labels[starts], starts, ends - starts',
    '',
    'def smooth_labels(labels, min_len=MIN_SEG_SAMPLES, max_passes=50):',
    '    """Standard microstate temporal-smoothing post-processing',
    '    (Pascual-Marqui, Michel & Lehmann, IEEE Trans Biomed Eng 1995 --',
    '    segments shorter than a minimum-duration criterion are not',
    '    physiologically plausible microstates and are reassigned).',
    '',
    '    We implement the SIMPLER "majority-merge" variant used by several',
    '    downstream toolboxes (e.g. the segment-rejection step in',
    '    Poulsen et al. 2018 / pycrostates), rather than Pascual-Marqui\'s',
    '    original iterative relabeling-by-correlation scheme: any segment',
    '    shorter than min_len samples is reassigned wholesale to whichever',
    '    of its two temporal neighbors is the LONGER segment (an edge',
    '    segment merges into its only neighbor). Because a merge can turn',
    '    a neighboring segment into a new short segment, this is iterated',
    '    to convergence (segmentation re-evaluated from scratch each pass)',
    '    or until max_passes is reached.',
    '    """',
    '    labels = labels.copy()',
    '    for _ in range(max_passes):',
    '        labs, starts, lens = rle(labels)',
    '        if len(labs) <= 1:',
    '            break',
    '        short = np.where(lens < min_len)[0]',
    '        if len(short) == 0:',
    '            break',
    '        changed = False',
    '        for si in short:',
    '            prev_len = lens[si - 1] if si > 0 else -1',
    '            next_len = lens[si + 1] if si < len(labs) - 1 else -1',
    '            if prev_len < 0 and next_len < 0:',
    '                continue',
    '            new_lab = labs[si - 1] if prev_len >= next_len else labs[si + 1]',
    '            labels[starts[si]:starts[si] + lens[si]] = new_lab',
    '            changed = True',
    '        if not changed:',
    '            break',
    '    return labels',
    '',
    '# -- Back-fit every run onto the templates -------------------',
    'print()',
    'print("=" * 55)',
    'print("Back-fitting continuous EEG onto " + str(N_MICROSTATES)',
    '      + " templates  (" + SFREQ_TAG + ")")',
    'print("=" * 55)',
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
    '        eeg = load_eeg(fif)',
    '        n_samples = eeg.shape[1]',
    '',
    '        # Real (non-normalized) GFP^2 per sample -- used as the GEV',
    '        # weighting term. NOTE: this deliberately differs from step',
    '        # 01\'s literal GEV code, which computed gfp_sq from the',
    '        # ALREADY L2-normalized GFP-peak maps (making that weight',
    '        # trivially ~1 for every peak). Here we have the genuine',
    '        # continuous-amplitude signal available, so gfp_sq is the',
    '        # true GFP^2 -- the textbook-correct GEV weighting -- rather',
    '        # than mirroring step 01\'s degenerate simplification.',
    '        gfp    = eeg.std(axis=0).astype(np.float32)',
    '        gfp_sq = (gfp.astype(np.float64) ** 2)',
    '',
    '        # Per-timepoint polarity-invariant correlation against',
    '        # templates: L2-normalize each timepoint\'s topography (same',
    '        # normalization step 01 applies to GFP-peak maps before',
    '        # k-means), then abs(dot product) with each unit-norm',
    '        # template.',
    '        norms = np.linalg.norm(eeg, axis=0)',
    '        norms_safe = np.where(norms > 1e-10, norms, 1.0).astype(np.float32)',
    '        eeg_norm = eeg / norms_safe',
    '        corr_matrix = np.abs(templates @ eeg_norm)   # (N_MICROSTATES, n_samples)',
    '        labels_raw = corr_matrix.argmax(axis=0)',
    '',
    '        labels_smooth = smooth_labels(labels_raw)',
    '',
    '        labs_f, starts_f, lens_f = rle(labels_smooth)',
    '        duration_s = n_samples / SFREQ',
    '',
    '        for k in range(N_MICROSTATES):',
    '            seg_mask = labs_f == k',
    '            seg_lens = lens_f[seg_mask]',
    '            n_segs_k = len(seg_lens)',
    '            coverage_pct = 100.0 * float(seg_lens.sum()) / n_samples \\',
    '                           if n_samples > 0 else 0.0',
    '            if n_segs_k > 0:',
    '                duration_ms = float(seg_lens.mean()) / SFREQ * 1000.0',
    '                occurrence_hz = n_segs_k / duration_s',
    '            else:',
    '                duration_ms = None',
    '                occurrence_hz = 0.0',
    '            samp_mask = labels_smooth == k',
    '            if samp_mask.sum() > 0 and gfp_sq.sum() > 0:',
    '                gev_frac = float((corr_matrix[k, samp_mask].astype(np.float64) ** 2',
    '                                  * gfp_sq[samp_mask]).sum() / gfp_sq.sum())',
    '            else:',
    '                gev_frac = 0.0',
    '            rows.append(dict(',
    '                subject=subject, run=run, condition=condition,',
    '                microstate=MS_LABELS[k],',
    '                duration_ms=round(duration_ms, 3) if duration_ms is not None else "",',
    '                occurrence_hz=round(occurrence_hz, 4),',
    '                coverage_pct=round(coverage_pct, 3),',
    '                gev_pct=round(gev_frac * 100.0, 3),',
    '            ))',
    '',
    '        cov_sum = sum(r["coverage_pct"] for r in rows[-N_MICROSTATES:])',
    '        n_ok += 1',
    '        print("  " + subject + "  run-" + run + "  (" + condition + ")"',
    '              + "  samples=" + str(n_samples)',
    '              + "  segs=" + str(len(labs_f))',
    '              + "  coverage_sum=" + "{:.1f}".format(cov_sum) + "%")',
    '',
    'print()',
    'print("Runs processed: " + str(n_ok) + "  missing: " + str(n_missing))',
    '',
    '# -- Write CSV ------------------------------------------------',
    'out_csv = RESULTS_DIR / "temporal_params.csv"',
    'fields = ["subject", "run", "condition", "microstate",',
    '          "duration_ms", "occurrence_hz", "coverage_pct", "gev_pct"]',
    'with open(str(out_csv), "w", newline="") as f:',
    '    w = csv.DictWriter(f, fieldnames=fields)',
    '    w.writeheader()',
    '    w.writerows(rows)',
    'print("Saved: " + str(out_csv) + "  (" + str(len(rows)) + " rows)")',
    '',
    'print()',
    'print("DONE  " + SFREQ_TAG)',
]

# ── 2. Save cluster script locally ─────────────────────────
script_name = "02_temporal_params_rtbpd_" + SFREQ_TAG + "_cluster.py"
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
job_name = "rtbpd_ms_temporal_" + SFREQ_TAG
sbatch_lines = [
    "#!/bin/bash",
    "#SBATCH --job-name=" + job_name,
    "#SBATCH --output=" + CLUSTER_BASE + "/logs/" + job_name + "_%j.out",
    "#SBATCH --error="  + CLUSTER_BASE + "/logs/" + job_name + "_%j.err",
    "#SBATCH --partition=short",
    "#SBATCH --time=02:00:00",
    "#SBATCH --cpus-per-task=4",
    "#SBATCH --mem=32G",
    "#SBATCH --account=" + SLURM_ACCOUNT,
    "",
    PYTHON + " " + CLUSTER_BASE + "/scripts/" + script_name,
]

sbatch_name = "02_temporal_params_rtbpd_" + SFREQ_TAG + ".sh"
sbatch_path = SCRIPTS_DIR / sbatch_name
with open(sbatch_path, "w") as f:
    f.write("\n".join(sbatch_lines))

remote_sbatch = CLUSTER_BASE + "/scripts/" + sbatch_name
scp_to(sbatch_path, remote_sbatch, verbose=False)

print("\nSubmitting SLURM job (" + SFREQ_TAG + ")...")
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
                    "tail -60 " + CLUSTER_BASE
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
