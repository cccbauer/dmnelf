# 08b_decoder_timeseries.py
# Run locally: python deploy_scripts/08b_decoder_timeseries.py
# Deploys cluster script, submits SLURM job.
#
# What it does on the cluster:
#   Re-fits the ridge decoder per subject (rest+shortrest → pda_direct)
#   and generates timeseries figures showing predicted vs actual PDA
#   for all feedback runs.
#
# Outputs (CLUSTER_BASE/results/personalized_decoder/timeseries/):
#   {subject}_decoder_timeseries.png   per-subject: all 4 feedback runs
#   group_decoder_overview.png         all subjects × 4 runs grid

import sys
import py_compile
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils import run_ssh, scp_to
from config import (
    CLUSTER_BASE, SLURM_ACCOUNT, PYTHON,
    SUBJECTS_EEG_FMRI_ALL, LOCAL_BASE,
    MISSING_RUNS,
)

lines = [
    '#!/usr/bin/env python3',
    '"""08b_decoder_timeseries_cluster.py',
    'Generates per-subject and group timeseries plots for the',
    'rest-trained Ridge decoder (T-hat nohrf → pda_direct).',
    '"""',
    'import sys',
    'sys.stdout.reconfigure(line_buffering=True)',
    'import numpy as np',
    'from pathlib import Path',
    'from scipy.stats import pearsonr',
    'from sklearn.linear_model import Ridge',
    'from sklearn.preprocessing import StandardScaler',
    'import matplotlib',
    'matplotlib.use("Agg")',
    'import matplotlib.pyplot as plt',
    '',
    'CLUSTER_BASE = Path("' + CLUSTER_BASE + '")',
    'FEATURES_DIR = CLUSTER_BASE / "features"',
    'TARGETS_DIR  = CLUSTER_BASE / "targets"',
    'OUT_DIR      = CLUSTER_BASE / "results" / "personalized_decoder" / "timeseries"',
    'OUT_DIR.mkdir(parents=True, exist_ok=True)',
    '',
    'SUBJECTS     = ' + str(SUBJECTS_EEG_FMRI_ALL),
    'MISSING_RUNS = ' + str(MISSING_RUNS),
    'SFREQ_TAG    = "250Hz"',
    'TR           = 1.2',
    '',
    'TASK_RUNS = {',
    '    "rest":      ["01", "02"],',
    '    "shortrest": ["01"],',
    '    "feedback":  ["01", "02", "03", "04"],',
    '}',
    '',
    'def load_run(subject, task, run):',
    '    feat_path = FEATURES_DIR / (subject + "_task-" + task',
    '                + "_run-" + run + "_" + SFREQ_TAG + "_nohrf_features.npy")',
    '    pda_path  = TARGETS_DIR  / (subject + "_task-" + task',
    '                + "_run-" + run + "_pda_direct.npy")',
    '    if not feat_path.exists() or not pda_path.exists():',
    '        return None, None',
    '    X = np.load(str(feat_path))[:, :7].astype(np.float64)',
    '    y = np.load(str(pda_path)).astype(np.float64)',
    '    n = min(len(X), len(y))',
    '    return X[:n], y[:n]',
    '',
    'def safe_r(a, b):',
    '    if len(a) < 3 or np.std(a) < 1e-10 or np.std(b) < 1e-10:',
    '        return float("nan")',
    '    return float(pearsonr(a, b)[0])',
    '',
    'print("=" * 60)',
    'print("Generating decoder timeseries figures")',
    'print("=" * 60)',
    '',
    '# Store per-subject results for group overview',
    'all_subjects_data = {}   # subject → list of (run, t, pda, pred)',
    '',
    'for subject in SUBJECTS:',
    '    print()',
    '    print("--- " + subject + " ---")',
    '',
    '    # Re-fit on rest + shortrest',
    '    X_parts, y_parts = [], []',
    '    for task in ("rest", "shortrest"):',
    '        for run in TASK_RUNS[task]:',
    '            if subject in MISSING_RUNS:',
    '                if (task, run) in MISSING_RUNS[subject]:',
    '                    continue',
    '            X, y = load_run(subject, task, run)',
    '            if X is not None:',
    '                X_parts.append(X)',
    '                y_parts.append(y)',
    '',
    '    if not X_parts:',
    '        print("  SKIP: no training data")',
    '        continue',
    '',
    '    X_train = np.vstack(X_parts)',
    '    y_train = np.concatenate(y_parts)',
    '    scaler  = StandardScaler()',
    '    X_tr_sc = scaler.fit_transform(X_train)',
    '    model   = Ridge(alpha=1.0)',
    '    model.fit(X_tr_sc, y_train)',
    '',
    '    # Collect feedback runs',
    '    fb_data = []',
    '    for run in TASK_RUNS["feedback"]:',
    '        if subject in MISSING_RUNS:',
    '            if ("feedback", run) in MISSING_RUNS[subject]:',
    '                continue',
    '        X_te, y_te = load_run(subject, "feedback", run)',
    '        if X_te is None:',
    '            continue',
    '        y_pred = model.predict(scaler.transform(X_te))',
    '        r_val  = safe_r(y_pred, y_te)',
    '        t_vec  = np.arange(len(y_te)) * TR',
    '        fb_data.append((run, t_vec, y_te, y_pred, r_val))',
    '        print("  fb run-" + run + "  r=" + str(round(r_val, 3)))',
    '',
    '    if not fb_data:',
    '        continue',
    '    all_subjects_data[subject] = fb_data',
    '',
    '    # ── Per-subject figure ────────────────────────────',
    '    n_runs = len(fb_data)',
    '    fig, axes = plt.subplots(n_runs, 1,',
    '                             figsize=(14, 2.8 * n_runs), squeeze=False)',
    '    fig.suptitle(subject + "  |  Ridge decoder: rest+shortrest → feedback\\n"',
    '                 "black = pda_direct,  orange = predicted,  shading = sign",',
    '                 fontsize=10, fontweight="bold")',
    '',
    '    for row_i, (run, t_vec, pda, pred, r_val) in enumerate(fb_data):',
    '        ax = axes[row_i, 0]',
    '        pda_z  = (pda  - pda.mean())  / max(pda.std(),  1e-10)',
    '        pred_z = (pred - pred.mean()) / max(pred.std(), 1e-10)',
    '',
    '        # Shade true PDA polarity (z-scored signal)',
    '        ax.fill_between(t_vec, pda_z, 0, where=(pda_z >= 0),',
    '                         color="#FF4444", alpha=0.18)',
    '        ax.fill_between(t_vec, pda_z, 0, where=(pda_z <  0),',
    '                         color="#4488FF", alpha=0.18)',
    '',
    '        ax.axhline(0, color="black", lw=0.6, ls="--")',
    '        ax.plot(t_vec, pda_z,  color="black",   lw=1.2, label="pda_direct (z)")',
    '        ax.plot(t_vec, pred_z, color="#E07020",  lw=1.2,',
    '                alpha=0.85, label="predicted (z,  r=" + str(round(r_val, 3)) + ")")',
    '',
    '        ax.set_ylabel("z-scored", fontsize=8)',
    '        ax.tick_params(labelsize=7)',
    '        ax.set_title("feedback run-" + run, fontsize=9, loc="left")',
    '        if row_i == 0:',
    '            ax.legend(loc="upper right", fontsize=8, ncol=2)',
    '        if row_i == n_runs - 1:',
    '            ax.set_xlabel("Time (s)", fontsize=9)',
    '',
    '    plt.tight_layout()',
    '    fig_path = OUT_DIR / (subject + "_decoder_timeseries.png")',
    '    plt.savefig(str(fig_path), dpi=130, bbox_inches="tight")',
    '    plt.close(fig)',
    '    print("  Saved: " + fig_path.name)',
    '',
    '# ── Group overview ────────────────────────────────────',
    'print()',
    'print("Generating group overview...")',
    '',
    'subjs_with_data = [s for s in SUBJECTS if s in all_subjects_data]',
    'n_subj = len(subjs_with_data)',
    'if n_subj == 0:',
    '    print("No data for group overview.")',
    'else:',
    '    MAX_RUNS = 4',
    '    fig2, axes2 = plt.subplots(',
    '        n_subj, MAX_RUNS,',
    '        figsize=(MAX_RUNS * 4.5, n_subj * 2.2),',
    '        squeeze=False',
    '    )',
    '    fig2.suptitle("Group overview: Ridge decoder  (rest+shortrest → feedback)\\n"',
    '                  "black = pda_direct,  orange = predicted",',
    '                  fontsize=11, fontweight="bold")',
    '',
    '    for si, subject in enumerate(subjs_with_data):',
    '        fb_data = all_subjects_data[subject]',
    '        run_dict = {run: (t, pda, pred, r) for run, t, pda, pred, r in fb_data}',
    '',
    '        for ri, run in enumerate(["01", "02", "03", "04"]):',
    '            ax = axes2[si, ri]',
    '            if run not in run_dict:',
    '                ax.text(0.5, 0.5, "missing", ha="center", va="center",',
    '                        transform=ax.transAxes, color="gray", fontsize=8)',
    '                ax.set_xticks([]); ax.set_yticks([])',
    '                continue',
    '            t_vec, pda, pred, r_val = run_dict[run]',
    '            pda_z  = (pda  - pda.mean())  / max(pda.std(),  1e-10)',
    '            pred_z = (pred - pred.mean()) / max(pred.std(), 1e-10)',
    '            ax.fill_between(t_vec, pda_z, 0, where=(pda_z >= 0),',
    '                             color="#FF4444", alpha=0.15)',
    '            ax.fill_between(t_vec, pda_z, 0, where=(pda_z <  0),',
    '                             color="#4488FF", alpha=0.15)',
    '            ax.axhline(0, color="black", lw=0.5, ls="--")',
    '            ax.plot(t_vec, pda_z,  color="black",  lw=0.9)',
    '            ax.plot(t_vec, pred_z, color="#E07020", lw=0.9, alpha=0.85)',
    '            ax.set_title("r=" + str(round(r_val, 3)), fontsize=8)',
    '            ax.tick_params(labelsize=6)',
    '            if ri == 0:',
    '                ax.set_ylabel(subject.replace("sub-dmnelf","s"), fontsize=7)',
    '            if si == 0:',
    '                ax.set_title("run-" + run + "  (r=" + str(round(r_val,3)) + ")",',
    '                             fontsize=8)',
    '            elif ri == 0:',
    '                ax.set_title("r=" + str(round(r_val, 3)), fontsize=8)',
    '            else:',
    '                ax.set_title("r=" + str(round(r_val, 3)), fontsize=8)',
    '',
    '    plt.tight_layout()',
    '    fig2_path = OUT_DIR / "group_decoder_overview.png"',
    '    plt.savefig(str(fig2_path), dpi=120, bbox_inches="tight")',
    '    plt.close(fig2)',
    '    print("Saved: " + fig2_path.name)',
    '',
    'print()',
    'print("=" * 60)',
    'print("DONE")',
    'print("=" * 60)',
]

# ── Save cluster script ─────────────────────────────────────────
script_name = "08b_decoder_timeseries_cluster.py"
script_path = LOCAL_BASE / "scripts" / script_name
script_path.parent.mkdir(parents=True, exist_ok=True)

with open(script_path, "w") as f:
    f.write("\n".join(lines))

# ── Syntax check ───────────────────────────────────────────────
print("Checking syntax...")
try:
    py_compile.compile(str(script_path), doraise=True)
    print("Syntax OK: " + script_name)
except py_compile.PyCompileError as e:
    print("SYNTAX ERROR: " + str(e))
    raise

# ── SLURM script ───────────────────────────────────────────────
job_name    = "ms_decoder_ts"
sbatch_name = "08b_decoder_timeseries.sh"
sbatch_path = LOCAL_BASE / "scripts" / sbatch_name

sbatch_lines = [
    "#!/bin/bash",
    "#SBATCH --job-name=" + job_name,
    "#SBATCH --output=" + CLUSTER_BASE + "/logs/" + job_name + "_%j.out",
    "#SBATCH --error="  + CLUSTER_BASE + "/logs/" + job_name + "_%j.err",
    "#SBATCH --partition=short",
    "#SBATCH --time=00:20:00",
    "#SBATCH --cpus-per-task=2",
    "#SBATCH --mem=8G",
    "#SBATCH --account=" + SLURM_ACCOUNT,
    "",
    PYTHON + " " + CLUSTER_BASE + "/scripts/" + script_name,
]

with open(sbatch_path, "w") as f:
    f.write("\n".join(sbatch_lines))

# ── Deploy ─────────────────────────────────────────────────────
print("\nDeploying...")
scp_to(script_path,
       CLUSTER_BASE + "/scripts/" + script_name,
       verbose=False)
scp_to(sbatch_path,
       CLUSTER_BASE + "/scripts/" + sbatch_name,
       verbose=False)
print("Deployed: " + script_name)

# ── Submit ─────────────────────────────────────────────────────
print("\nSubmitting SLURM job...")
result = run_ssh("sbatch " + CLUSTER_BASE + "/scripts/" + sbatch_name)
job_id = ""
for line in result.stdout.strip().split("\n"):
    if "Submitted" in line:
        job_id = line.strip().split()[-1]
        print("Job ID: " + job_id)

# ── Monitor ────────────────────────────────────────────────────
if job_id:
    print("\nMonitoring job " + job_id + "  (Ctrl+C to stop)")
    print("-" * 60)
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
            time.sleep(20)
    except KeyboardInterrupt:
        print("\nStopped watching.")
        print("  tail -f " + CLUSTER_BASE
              + "/logs/" + job_name + "_" + job_id + ".out")
