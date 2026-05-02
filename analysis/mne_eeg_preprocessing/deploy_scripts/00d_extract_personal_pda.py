# 00d_extract_personal_pda.py
# Run locally: python 00d_extract_personal_pda.py
# Deploys cluster script, submits SLURM job.
#
# What it does on the cluster:
#   For each subject x task x run:
#   1. Load personalized DMN and CEN binary masks (from 00b)
#   2. Apply NiftiMasker to fMRIPrep BOLD — mean signal within each mask
#   3. Baseline z-score (first 25 vols, matching MURFI)
#   4. Compute PDA_direct = CEN_z - DMN_z
#   5. Save as (n_vols,) float32
#
# Output:
#   targets/{subject}_task-{task}_run-{run}_pda_direct.npy

import py_compile
import time
from pathlib import Path
from utils import run_ssh, scp_to
from config import (
    CLUSTER_BASE, SLURM_ACCOUNT, PYTHON,
    SUBJECTS_EEG_FMRI_ALL, LOCAL_BASE,
    MISSING_RUNS
)

FMRIPREP_ROOT = "/projects/swglab/data/DMNELF/derivatives/fmriprep_25.2.5_fmap"
MASKS_ROOT    = "/projects/swglab/data/DMNELF/derivatives/network_masks"

lines = [
    '#!/usr/bin/env python3',
    '"""',
    '00d_extract_personal_pda_cluster.py',
    'Extract PDA_direct from personalized DMN/CEN masks applied directly',
    'to fMRIPrep BOLD. Matches MURFI real-time computation logic.',
    '"""',
    'import sys',
    'sys.stdout.reconfigure(line_buffering=True)',
    'import numpy as np',
    'from pathlib import Path',
    'import warnings',
    'import csv',
    'import matplotlib',
    'matplotlib.use("Agg")',
    'import matplotlib.pyplot as plt',
    '',
    'FMRIPREP_ROOT = Path("' + FMRIPREP_ROOT + '")',
    'MASKS_ROOT    = Path("' + MASKS_ROOT + '")',
    'CLUSTER_BASE  = Path("' + CLUSTER_BASE + '")',
    'OUT_DIR        = CLUSTER_BASE / "targets"',
    'OUT_DIR.mkdir(parents=True, exist_ok=True)',
    'PDA_PLOTS_ROOT = Path("/projects/swglab/data/DMNELF/derivatives/PDA_plots")',
    'PDA_PLOTS_ROOT.mkdir(parents=True, exist_ok=True)',
    '',
    'SUBJECTS     = ' + str(SUBJECTS_EEG_FMRI_ALL),
    'MISSING_RUNS = ' + str(MISSING_RUNS),
    'BASELINE_VOLS = 25',
    'TR            = 1.2   # seconds per volume',
    '',
    'TASK_RUNS = {',
    '    "rest":      ["01", "02"],',
    '    "shortrest": ["01"],',
    '    "feedback":  ["01", "02", "03", "04"],',
    '}',
    '',
    'def baseline_zscore(x, n=25):',
    '    mu  = x[:n].mean()',
    '    sig = x[:n].std()',
    '    if sig < 1e-10:',
    '        sig = 1e-10',
    '    return (x - mu) / sig',
    '',
    'def detect_pda_epochs(pda, tr=1.2, min_vols=3):',
    '    """Detect consecutive positive/negative PDA epochs.',
    '    Short isolated sign flips (< min_vols) are merged into surrounding epoch.',
    '    Returns list of (onset_vol, offset_vol, sign) tuples.',
    '    """',
    '    sign = np.where(pda >= 0, 1, -1)',
    '    # Merge isolated flips shorter than min_vols',
    '    i = 0',
    '    while i < len(sign):',
    '        if i > 0 and i < len(sign) - 1 and sign[i] != sign[i - 1]:',
    '            j = i',
    '            while j < len(sign) and sign[j] != sign[i - 1]:',
    '                j += 1',
    '            if (j - i) < min_vols:',
    '                sign[i:j] = sign[i - 1]',
    '                i = j',
    '                continue',
    '        i += 1',
    '    # Run-length encode',
    '    epochs = []',
    '    s, c = 0, sign[0]',
    '    for i in range(1, len(sign)):',
    '        if sign[i] != c:',
    '            epochs.append((s, i - 1, int(c)))',
    '            s, c = i, sign[i]',
    '    epochs.append((s, len(sign) - 1, int(c)))',
    '    return epochs',
    '',
    'print("=" * 55)',
    'print("Extracting PDA_direct from personal masks")',
    'print("=" * 55)',
    '',
    'from nilearn.maskers import NiftiMasker',
    '',
    'n_done    = 0',
    'n_missing = 0',
    '',
    'for subject in SUBJECTS:',
    '    # Load personal masks',
    '    mask_dir = MASKS_ROOT / subject',
    '    dmn_mask = mask_dir / (subject',
    '               + "_space-MNI152NLin6Asym_res-2_dmn_mask.nii.gz")',
    '    cen_mask = mask_dir / (subject',
    '               + "_space-MNI152NLin6Asym_res-2_cen_mask.nii.gz")',
    '',
    '    if not dmn_mask.exists() or not cen_mask.exists():',
    '        print("  MISSING MASKS: " + subject)',
    '        n_missing += 1',
    '        continue',
    '',
    '    # Build maskers',
    '    with warnings.catch_warnings():',
    '        warnings.simplefilter("ignore")',
    '        dmn_masker = NiftiMasker(',
    '            mask_img=str(dmn_mask),',
    '            standardize=False,',
    '            verbose=0',
    '        )',
    '        cen_masker = NiftiMasker(',
    '            mask_img=str(cen_mask),',
    '            standardize=False,',
    '            verbose=0',
    '        )',
    '        dmn_masker.fit()',
    '        cen_masker.fit()',
    '',
    '    print()',
    '    print(subject)',
    '',
    '    for task, runs in TASK_RUNS.items():',
    '        for run in runs:',
    '',
    '            if subject in MISSING_RUNS:',
    '                if (task, run) in MISSING_RUNS[subject]:',
    '                    continue',
    '',
    '            bold_name = (subject + "_ses-dmnelf"',
    '                         + "_task-" + task',
    '                         + "_run-"  + run',
    '                         + "_space-MNI152NLin6Asym_res-2"',
    '                         + "_desc-preproc_bold.nii.gz")',
    '            bold_path = (FMRIPREP_ROOT / subject',
    '                         / "ses-dmnelf" / "func" / bold_name)',
    '',
    '            out_path = OUT_DIR / (subject + "_task-" + task',
    '                                  + "_run-" + run',
    '                                  + "_pda_direct.npy")',
    '',
    '            if not bold_path.exists():',
    '                print("  MISSING BOLD: " + bold_name)',
    '                n_missing += 1',
    '                continue',
    '',
    '            try:',
    '                with warnings.catch_warnings():',
    '                    warnings.simplefilter("ignore")',
    '                    dmn_ts = dmn_masker.transform(str(bold_path))',
    '                    cen_ts = cen_masker.transform(str(bold_path))',
    '            except Exception as e:',
    '                print("  ERROR: " + str(e))',
    '                n_missing += 1',
    '                continue',
    '',
    '            # Mean across voxels within each mask',
    '            dmn_mean = dmn_ts.mean(axis=1)',
    '            cen_mean = cen_ts.mean(axis=1)',
    '',
    '            # Baseline z-score (MURFI convention)',
    '            dmn_z = baseline_zscore(dmn_mean)',
    '            cen_z = baseline_zscore(cen_mean)',
    '',
    '            # PDA = CEN - DMN',
    '            pda = (cen_z - dmn_z).astype(np.float32)',
    '',
    '            np.save(str(out_path), pda)',
    '',
    '            # Save intermediate signals',
    '            _dmn_z_path = OUT_DIR / (subject + "_task-" + task',
    '                          + "_run-" + run + "_dmn_z.npy")',
    '            _cen_z_path = OUT_DIR / (subject + "_task-" + task',
    '                          + "_run-" + run + "_cen_z.npy")',
    '            np.save(str(_dmn_z_path), dmn_z.astype(np.float32))',
    '            np.save(str(_cen_z_path), cen_z.astype(np.float32))',
    '',
    '            # Per-run polarity plot',
    '            _plot_dir = PDA_PLOTS_ROOT / subject',
    '            _plot_dir.mkdir(parents=True, exist_ok=True)',
    '            _tvec    = np.arange(len(pda)) * TR',
    '            _pol_png = _plot_dir / (subject + "_task-" + task',
    '                        + "_run-" + run + "_pda_polarity.png")',
    '            _fig_p, _ax_p = plt.subplots(1, 1, figsize=(14, 3))',
    '            _ax_p.axvspan(0, BASELINE_VOLS * TR, color="lightgray",',
    '                          alpha=0.5, label="baseline")',
    '            _ax_p.fill_between(_tvec, pda, 0, where=(pda >= 0),',
    '                               color="#FF1A00", alpha=0.6, label="PDA+")',
    '            _ax_p.fill_between(_tvec, pda, 0, where=(pda < 0),',
    '                               color="#0077FF", alpha=0.4, label="PDA-")',
    '            _ax_p.plot(_tvec, pda, color="black", lw=0.8)',
    '            _ax_p.axhline(0, color="black", lw=0.5, ls="--")',
    '            _ax_p.set_xlabel("Time (s)")',
    '            _ax_p.set_ylabel("z-score")',
    '            _ax_p.set_title(subject + "  " + task + " run-" + run',
    '                           + "  |  PDA = CEN - DMN")',
    '            _ax_p.legend(loc="upper right", fontsize=8, ncol=3)',
    '            plt.tight_layout()',
    '            plt.savefig(str(_pol_png), dpi=150, bbox_inches="tight")',
    '            plt.close(_fig_p)',
    '',
    '            # PDA epoch TSV',
    '            _epochs   = detect_pda_epochs(pda, tr=TR)',
    '            _tsv_path = _plot_dir / (subject + "_task-" + task',
    '                         + "_run-" + run + "_pda_epochs.tsv")',
    '            with open(str(_tsv_path), "w", newline="") as _f:',
    '                _writer = csv.writer(_f, delimiter="\\t")',
    '                _writer.writerow(["onset_vol", "offset_vol",',
    '                                  "onset_sec", "offset_sec",',
    '                                  "duration_sec", "sign"])',
    '                for _ov, _ev, _sgn in _epochs:',
    '                    _writer.writerow([',
    '                        _ov, _ev,',
    '                        round(_ov * TR, 3),',
    '                        round(_ev * TR, 3),',
    '                        round((_ev - _ov + 1) * TR, 3),',
    '                        _sgn',
    '                    ])',
    '',
    '            print("  " + task + " run-" + run',
    '                  + "  n=" + str(len(pda))',
    '                  + "  pda_std=" + str(round(pda.std(), 3))',
    '                  + "  dmn_vox=" + str(dmn_ts.shape[1])',
    '                  + "  cen_vox=" + str(cen_ts.shape[1]))',
    '            n_done += 1',
    '',
    '',
    '# ── Per-subject timeseries plots ─────────────────────────────',
    'print()',
    'print("=" * 55)',
    'print("Generating per-subject timeseries plots")',
    'print("=" * 55)',
    '',
    'TASK_RUNS_ORDERED = [',
    '    ("rest",      "01"), ("rest",      "02"),',
    '    ("shortrest", "01"),',
    '    ("feedback",  "01"), ("feedback",  "02"),',
    '    ("feedback",  "03"), ("feedback",  "04"),',
    ']',
    '',
    'for subject in SUBJECTS:',
    '    _plot_dir = PDA_PLOTS_ROOT / subject',
    '    _out_ts   = _plot_dir / (subject + "_pda_timeseries.png")',
    '    if _out_ts.exists():',
    '        print("  EXISTS (skip): " + subject)',
    '        continue',
    '    _runs = []',
    '    for _task, _run in TASK_RUNS_ORDERED:',
    '        if subject in MISSING_RUNS:',
    '            if (_task, _run) in MISSING_RUNS[subject]:',
    '                continue',
    '        _pda_p = OUT_DIR / (subject + "_task-" + _task',
    '                            + "_run-" + _run + "_pda_direct.npy")',
    '        _dmn_p = OUT_DIR / (subject + "_task-" + _task',
    '                            + "_run-" + _run + "_dmn_z.npy")',
    '        _cen_p = OUT_DIR / (subject + "_task-" + _task',
    '                            + "_run-" + _run + "_cen_z.npy")',
    '        if _pda_p.exists() and _dmn_p.exists() and _cen_p.exists():',
    '            _runs.append((_task, _run, _pda_p, _dmn_p, _cen_p))',
    '    if not _runs:',
    '        print("  NO DATA: " + subject)',
    '        continue',
    '    _n = len(_runs)',
    '    _fig, _axes = plt.subplots(_n, 1, figsize=(14, 2.5 * _n), squeeze=False)',
    '    _fig.suptitle(subject + "  |  DMN (blue)  CEN (red)  PDA (black)",',
    '                  fontsize=11, fontweight="bold")',
    '    for _row, (_task, _run, _pp, _dp, _cp) in enumerate(_runs):',
    '        _ax  = _axes[_row, 0]',
    '        _pda = np.load(str(_pp))',
    '        _dmn = np.load(str(_dp))',
    '        _cen = np.load(str(_cp))',
    '        _t   = np.arange(len(_pda)) * TR',
    '        _ax.axvspan(0, BASELINE_VOLS * TR, color="lightgray",',
    '                    alpha=0.4, label="baseline")',
    '        _ax.axhline(0, color="black", lw=0.5, ls="--")',
    '        _ax.plot(_t, _dmn, color="#0077FF", lw=1.0, alpha=0.8, label="DMN")',
    '        _ax.plot(_t, _cen, color="#FF1A00", lw=1.0, alpha=0.8, label="CEN")',
    '        _ax.plot(_t, _pda, color="black",   lw=1.5, label="PDA")',
    '        _ax.set_ylabel("z-score", fontsize=8)',
    '        _ax.set_title(_task + " run-" + _run, fontsize=9, loc="left")',
    '        if _row == 0:',
    '            _ax.legend(loc="upper right", fontsize=7, ncol=4)',
    '        if _row == _n - 1:',
    '            _ax.set_xlabel("Time (s)")',
    '    plt.tight_layout()',
    '    _plot_dir.mkdir(parents=True, exist_ok=True)',
    '    plt.savefig(str(_out_ts), dpi=150, bbox_inches="tight")',
    '    plt.close(_fig)',
    '    print("  Saved: " + _out_ts.name)',
    '',
    '# ── Overview: all subjects ────────────────────────────────────',
    'print()',
    'print("=" * 55)',
    'print("Generating all-subjects overview PNG")',
    'print("=" * 55)',
    '',
    '_out_ov      = PDA_PLOTS_ROOT / "all_subjects_pda_overview.png"',
    '_ov_tasks    = ["rest", "shortrest", "feedback"]',
    '_ov_task_runs = {',
    '    "rest":      ["01", "02"],',
    '    "shortrest": ["01"],',
    '    "feedback":  ["01", "02", "03", "04"],',
    '}',
    '_fig_ov, _ax_ov = plt.subplots(',
    '    len(SUBJECTS), 3,',
    '    figsize=(16, 2.2 * len(SUBJECTS)),',
    '    squeeze=False',
    ')',
    '_fig_ov.suptitle("All subjects  |  PDA = CEN - DMN",',
    '                 fontsize=13, fontweight="bold")',
    'for _si, _sub in enumerate(SUBJECTS):',
    '    for _ti, _task in enumerate(_ov_tasks):',
    '        _ax = _ax_ov[_si, _ti]',
    '        _ax.axhline(0, color="black", lw=0.5, ls="--")',
    '        _traces = []',
    '        for _run in _ov_task_runs[_task]:',
    '            if _sub in MISSING_RUNS:',
    '                if (_task, _run) in MISSING_RUNS[_sub]:',
    '                    continue',
    '            _p = OUT_DIR / (_sub + "_task-" + _task',
    '                            + "_run-" + _run + "_pda_direct.npy")',
    '            if _p.exists():',
    '                _traces.append(np.load(str(_p)).astype(float))',
    '        if _traces:',
    '            _ax.axvspan(0, BASELINE_VOLS * TR, color="lightgray", alpha=0.4)',
    '            for _tr in _traces:',
    '                _ax.plot(np.arange(len(_tr)) * TR, _tr,',
    '                         color="black", lw=0.7, alpha=0.5)',
    '            if len(_traces) > 1:',
    '                _min_l = min(len(_tr) for _tr in _traces)',
    '                _mat   = np.array([_tr[:_min_l] for _tr in _traces])',
    '                _t2    = np.arange(_min_l) * TR',
    '                _mean  = _mat.mean(0)',
    '                _ax.fill_between(_t2, _mean, 0, where=(_mean >= 0),',
    '                                 color="#FF1A00", alpha=0.35)',
    '                _ax.fill_between(_t2, _mean, 0, where=(_mean < 0),',
    '                                 color="#0077FF", alpha=0.35)',
    '                _ax.plot(_t2, _mean, color="black", lw=1.5)',
    '            else:',
    '                _tr0 = _traces[0]',
    '                _t0  = np.arange(len(_tr0)) * TR',
    '                _ax.fill_between(_t0, _tr0, 0, where=(_tr0 >= 0),',
    '                                 color="#FF1A00", alpha=0.35)',
    '                _ax.fill_between(_t0, _tr0, 0, where=(_tr0 < 0),',
    '                                 color="#0077FF", alpha=0.35)',
    '        else:',
    '            _ax.text(0.5, 0.5, "missing", ha="center", va="center",',
    '                     transform=_ax.transAxes, color="gray", fontsize=8)',
    '        if _si == 0:',
    '            _ax.set_title(_task, fontsize=10, fontweight="bold")',
    '        if _ti == 0:',
    '            _ax.set_ylabel(_sub.replace("sub-dmnelf", "sub-"), fontsize=7)',
    '        _ax.tick_params(labelsize=6)',
    'plt.tight_layout()',
    'plt.savefig(str(_out_ov), dpi=150, bbox_inches="tight")',
    'plt.close(_fig_ov)',
    'print("  Saved: " + str(_out_ov))',
    '',
    'print()',
    'print("=" * 55)',
    'print("DONE")',
    'print("  Computed: " + str(n_done))',
    'print("  Missing:  " + str(n_missing))',
    'print("=" * 55)',
]

# ── Save cluster script ─────────────────────────────────────
script_name = "00d_extract_personal_pda_cluster.py"
script_path = LOCAL_BASE / "scripts" / script_name
script_path.parent.mkdir(parents=True, exist_ok=True)

with open(script_path, "w") as f:
    f.write("\n".join(lines))

# ── Syntax check ───────────────────────────────────────────
print("Checking syntax...")
try:
    py_compile.compile(str(script_path), doraise=True)
    print("Syntax OK: " + script_name)
except py_compile.PyCompileError as e:
    print("SYNTAX ERROR: " + str(e))
    raise

# ── SLURM script ───────────────────────────────────────────
job_name = "extract_pda_direct"
sbatch_lines = [
    "#!/bin/bash",
    "#SBATCH --job-name=" + job_name,
    "#SBATCH --output=" + CLUSTER_BASE + "/logs/" + job_name + "_%j.out",
    "#SBATCH --error="  + CLUSTER_BASE + "/logs/" + job_name + "_%j.err",
    "#SBATCH --partition=short",
    "#SBATCH --time=06:00:00",
    "#SBATCH --cpus-per-task=4",
    "#SBATCH --mem=64G",
    "#SBATCH --account=" + SLURM_ACCOUNT,
    "",
    PYTHON + " " + CLUSTER_BASE + "/scripts/" + script_name,
]

sbatch_name = "00d_extract_personal_pda.sh"
sbatch_path = LOCAL_BASE / "scripts" / sbatch_name
with open(sbatch_path, "w") as f:
    f.write("\n".join(sbatch_lines))

# ── Deploy ─────────────────────────────────────────────────
print("\nDeploying...")
scp_to(script_path,
       CLUSTER_BASE + "/scripts/" + script_name,
       verbose=False)
scp_to(sbatch_path,
       CLUSTER_BASE + "/scripts/" + sbatch_name,
       verbose=False)
print("Deployed: " + script_name)

# ── Submit ─────────────────────────────────────────────────
print("\nSubmitting SLURM job...")
result = run_ssh("sbatch " + CLUSTER_BASE + "/scripts/" + sbatch_name)
job_id = ""
for line in result.stdout.strip().split("\n"):
    if "Submitted" in line:
        job_id = line.strip().split()[-1]
        print("Job ID: " + job_id)

# ── Monitor ────────────────────────────────────────────────
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
                    "tail -30 " + CLUSTER_BASE
                    + "/logs/" + job_name + "_" + job_id
                    + ".out 2>/dev/null",
                    verbose=False
                )
                print(log.stdout)
                break
            time.sleep(20)
    except KeyboardInterrupt:
        print("\nStopped watching.")
        print("  tail -f " + CLUSTER_BASE
              + "/logs/" + job_name + "_" + job_id + ".out")