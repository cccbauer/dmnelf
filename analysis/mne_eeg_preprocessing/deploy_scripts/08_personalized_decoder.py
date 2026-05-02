# 08_personalized_decoder.py
# Run locally: python deploy_scripts/08_personalized_decoder.py
# Deploys cluster script, submits SLURM job.
#
# What it does on the cluster:
#   Per-subject personalized decoder:
#     Training: rest + shortrest  (T-hat → pda_direct)
#     Test:     each feedback run (held-out, never seen during training)
#
#   Models:
#     Ridge   (alpha=1.0, fast baseline)
#     ElasticNet (alpha=0.1, l1_ratio=0.5, sparse/interpretable)
#
#   Metrics per feedback run:
#     r         — Pearson correlation (magnitude)
#     sign_acc  — fraction of TRs where predicted sign matches pda_direct sign
#     auc       — ROC AUC (pda_direct>0 as positive class, predicted value as score)
#
#   Comparison baseline: step 04 LORO r from *_cv_personal.json
#
# Outputs (CLUSTER_BASE/results/personalized_decoder/):
#   personalized_decoder_250Hz.csv       per-subject × run × model
#   personalized_decoder_coefs_250Hz.csv per-subject × model coefficients
#   personalized_decoder_250Hz.png       summary figure

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
    '"""08_personalized_decoder_cluster.py',
    'Per-subject decoder trained on rest+shortrest, tested on feedback.',
    'Target: pda_direct (personal masks). Features: T-hat (nohrf, 7 cols).',
    '"""',
    'import sys',
    'sys.stdout.reconfigure(line_buffering=True)',
    'import numpy as np',
    'import json, csv',
    'from pathlib import Path',
    'from scipy.stats import pearsonr',
    'from sklearn.linear_model import Ridge, ElasticNet',
    'from sklearn.preprocessing import StandardScaler',
    'from sklearn.metrics import roc_auc_score',
    'import matplotlib',
    'matplotlib.use("Agg")',
    'import matplotlib.pyplot as plt',
    '',
    '# ── Paths ────────────────────────────────────────────',
    'CLUSTER_BASE  = Path("' + CLUSTER_BASE + '")',
    'FEATURES_DIR  = CLUSTER_BASE / "features"',
    'TARGETS_DIR   = CLUSTER_BASE / "targets"',
    'MODELS_DIR    = CLUSTER_BASE / "models"',
    'OUT_DIR       = CLUSTER_BASE / "results" / "personalized_decoder"',
    'OUT_DIR.mkdir(parents=True, exist_ok=True)',
    '',
    'SUBJECTS     = ' + str(SUBJECTS_EEG_FMRI_ALL),
    'MISSING_RUNS = ' + str(MISSING_RUNS),
    'SFREQ_TAG    = "250Hz"',
    '',
    'TASK_RUNS = {',
    '    "rest":      ["01", "02"],',
    '    "shortrest": ["01"],',
    '    "feedback":  ["01", "02", "03", "04"],',
    '}',
    '',
    'MS_LABELS = ["DMN","AUD","VIS","CEN","SAL","DANT","SOM"]',
    '',
    '# ── Helper ───────────────────────────────────────────',
    'def load_run(subject, task, run):',
    '    feat_path = FEATURES_DIR / (subject + "_task-" + task',
    '                + "_run-" + run + "_" + SFREQ_TAG + "_nohrf_features.npy")',
    '    pda_path  = TARGETS_DIR  / (subject + "_task-" + task',
    '                + "_run-" + run + "_pda_direct.npy")',
    '    if not feat_path.exists() or not pda_path.exists():',
    '        return None, None',
    '    X = np.load(str(feat_path))[:, :7].astype(np.float64)  # T-hat only',
    '    y = np.load(str(pda_path)).astype(np.float64)',
    '    n = min(len(X), len(y))',
    '    return X[:n], y[:n]',
    '',
    'def safe_r(a, b):',
    '    if len(a) < 3 or np.std(a) < 1e-10 or np.std(b) < 1e-10:',
    '        return float("nan")',
    '    return float(pearsonr(a, b)[0])',
    '',
    'def sign_accuracy(y_pred, y_true):',
    '    return float(np.mean(np.sign(y_pred) == np.sign(y_true)))',
    '',
    'def safe_auc(y_pred, y_true):',
    '    labels = (y_true > 0).astype(int)',
    '    if labels.sum() == 0 or labels.sum() == len(labels):',
    '        return float("nan")',
    '    return float(roc_auc_score(labels, y_pred))',
    '',
    '# ── Main loop ─────────────────────────────────────────',
    'print()',
    'print("=" * 60)',
    'print("Personalized decoder: rest+shortrest → feedback")',
    'print("=" * 60)',
    '',
    'result_rows = []',
    'coef_rows   = []',
    '',
    'models_cfg = [',
    '    ("ridge",      Ridge(alpha=1.0)),',
    '    ("elasticnet", ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=5000)),',
    ']',
    '',
    'for subject in SUBJECTS:',
    '    print()',
    '    print("--- " + subject + " ---")',
    '',
    '    # ── Load training data (rest + shortrest) ────────',
    '    X_parts, y_parts = [], []',
    '    for task in ("rest", "shortrest"):',
    '        for run in TASK_RUNS[task]:',
    '            if subject in MISSING_RUNS:',
    '                if (task, run) in MISSING_RUNS[subject]:',
    '                    continue',
    '            X, y = load_run(subject, task, run)',
    '            if X is None:',
    '                continue',
    '            X_parts.append(X)',
    '            y_parts.append(y)',
    '',
    '    if not X_parts:',
    '        print("  SKIP: no rest/shortrest data")',
    '        continue',
    '',
    '    X_train = np.vstack(X_parts)',
    '    y_train = np.concatenate(y_parts)',
    '    print("  Train: " + str(X_train.shape[0]) + " TRs from "',
    '          + str(len(X_parts)) + " runs")',
    '',
    '    # ── Fit models ───────────────────────────────────',
    '    scaler = StandardScaler()',
    '    X_tr_sc = scaler.fit_transform(X_train)',
    '',
    '    fitted = {}',
    '    for name, model in models_cfg:',
    '        model.fit(X_tr_sc, y_train)',
    '        fitted[name] = model',
    '        coef_rows.append({',
    '            "subject": subject, "model": name,',
    '            **{"w_" + lbl: round(float(c), 6)',
    '               for lbl, c in zip(MS_LABELS, model.coef_)}',
    '        })',
    '',
    '    # ── Load step 04 LORO r (baseline comparison) ────',
    '    cv_path = MODELS_DIR / (subject + "_250Hz_cv_personal.json")',
    '    step04_r = float("nan")',
    '    if cv_path.exists():',
    '        with open(str(cv_path)) as _f:',
    '            _cv = json.load(_f)',
    '        step04_r = float(_cv.get("loro_r", float("nan")))',
    '',
    '    # ── Test on each feedback run ─────────────────────',
    '    for run in TASK_RUNS["feedback"]:',
    '        if subject in MISSING_RUNS:',
    '            if ("feedback", run) in MISSING_RUNS[subject]:',
    '                continue',
    '        X_te, y_te = load_run(subject, "feedback", run)',
    '        if X_te is None:',
    '            continue',
    '        X_te_sc = scaler.transform(X_te)',
    '',
    '        row = {"subject": subject, "run": run,',
    '               "step04_r": round(step04_r, 4)}',
    '',
    '        for name, model in fitted.items():',
    '            y_pred = model.predict(X_te_sc)',
    '            row["r_"        + name] = round(safe_r(y_pred, y_te), 4)',
    '            row["sign_acc_" + name] = round(sign_accuracy(y_pred, y_te), 4)',
    '            row["auc_"      + name] = round(safe_auc(y_pred, y_te), 4)',
    '',
    '        result_rows.append(row)',
    '        print("  fb run-" + run',
    '              + "  r_ridge="  + str(row.get("r_ridge", "?"))',
    '              + "  r_enet="   + str(row.get("r_elasticnet", "?"))',
    '              + "  sign_acc=" + str(row.get("sign_acc_ridge", "?")))',
    '',
    '# ── Save CSVs ─────────────────────────────────────────',
    'def write_csv(path, rows):',
    '    if not rows: return',
    '    seen = {}',
    '    for r in rows:',
    '        for k in r: seen[k] = None',
    '    fields = list(seen.keys())',
    '    with open(str(path), "w", newline="") as f:',
    '        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")',
    '        w.writeheader(); w.writerows(rows)',
    '    print("  " + path.name)',
    '',
    'write_csv(OUT_DIR / "personalized_decoder_250Hz.csv",       result_rows)',
    'write_csv(OUT_DIR / "personalized_decoder_coefs_250Hz.csv", coef_rows)',
    '',
    '# ── Figure ────────────────────────────────────────────',
    'print()',
    'print("=" * 60)',
    'print("Generating figure")',
    'print("=" * 60)',
    '',
    'from collections import defaultdict',
    'C_RIDGE = "#5080D0"',
    'C_ENET  = "#E07020"',
    'C_S04   = "#444444"',
    '',
    'def subj_means(col):',
    '    sm = defaultdict(list)',
    '    for r in result_rows:',
    '        v = r.get(col)',
    '        if v is not None and str(v) != "nan":',
    '            sm[r["subject"]].append(float(v))',
    '    return [float(np.mean(vs)) for vs in sm.values() if vs]',
    '',
    'fig, axes = plt.subplots(1, 3, figsize=(18, 5))',
    'fig.suptitle("Personalized decoder: train rest+shortrest  →  test feedback\\n"',
    '             "(T-hat nohrf → pda_direct, per subject mean ± SE)",',
    '             fontsize=11, fontweight="bold")',
    '',
    '# ── Panel A: Pearson r ────────────────────────────────',
    'ax = axes[0]',
    'groups = [',
    '    ("Ridge",      "r_ridge",       C_RIDGE),',
    '    ("ElasticNet", "r_elasticnet",  C_ENET),',
    '    ("Step04\\nLORO", "step04_r",   C_S04),',
    ']',
    'jrng = np.random.default_rng(0)',
    'for ci, (lbl, col, col_c) in enumerate(groups):',
    '    vals = subj_means(col)',
    '    if not vals: continue',
    '    jit = jrng.uniform(-0.08, 0.08, len(vals))',
    '    ax.scatter(np.full(len(vals), ci) + jit, vals,',
    '               color=col_c, alpha=0.75, s=60, zorder=3)',
    '    m  = float(np.mean(vals))',
    '    se = float(np.std(vals, ddof=1) / np.sqrt(len(vals))) if len(vals) > 1 else 0',
    '    ax.errorbar(ci, m, yerr=se, fmt="D", color=col_c,',
    '                markersize=11, capsize=6, lw=2.4, zorder=5)',
    '    ax.text(ci, m + se + 0.01, f"{m:+.3f}", ha="center",',
    '            fontsize=9, color=col_c, fontweight="bold")',
    'ax.axhline(0, color="black", lw=0.8, ls="--")',
    'ax.set_xticks(range(len(groups)))',
    'ax.set_xticklabels([g[0] for g in groups], fontsize=11)',
    'ax.set_ylabel("Pearson r  (predicted vs pda_direct)", fontsize=10)',
    'ax.set_title("A   Correlation", fontsize=11, fontweight="bold")',
    '',
    '# ── Panel B: Sign accuracy ────────────────────────────',
    'ax = axes[1]',
    'groups_sa = [',
    '    ("Ridge",      "sign_acc_ridge",       C_RIDGE),',
    '    ("ElasticNet", "sign_acc_elasticnet",   C_ENET),',
    ']',
    'for ci, (lbl, col, col_c) in enumerate(groups_sa):',
    '    vals = subj_means(col)',
    '    if not vals: continue',
    '    jit = jrng.uniform(-0.08, 0.08, len(vals))',
    '    ax.scatter(np.full(len(vals), ci) + jit, vals,',
    '               color=col_c, alpha=0.75, s=60, zorder=3)',
    '    m  = float(np.mean(vals))',
    '    se = float(np.std(vals, ddof=1) / np.sqrt(len(vals))) if len(vals) > 1 else 0',
    '    ax.errorbar(ci, m, yerr=se, fmt="D", color=col_c,',
    '                markersize=11, capsize=6, lw=2.4, zorder=5)',
    '    ax.text(ci, m + se + 0.005, f"{m:.3f}", ha="center",',
    '            fontsize=9, color=col_c, fontweight="bold")',
    'ax.axhline(0.5, color="black", lw=0.8, ls="--", label="chance (0.5)")',
    'ax.set_ylim(0.35, 0.75)',
    'ax.set_xticks(range(len(groups_sa)))',
    'ax.set_xticklabels([g[0] for g in groups_sa], fontsize=11)',
    'ax.set_ylabel("Sign accuracy (fraction correct)", fontsize=10)',
    'ax.set_title("B   Sign accuracy (BCI metric)", fontsize=11, fontweight="bold")',
    'ax.legend(fontsize=9)',
    '',
    '# ── Panel C: AUC ─────────────────────────────────────',
    'ax = axes[2]',
    'groups_auc = [',
    '    ("Ridge",      "auc_ridge",       C_RIDGE),',
    '    ("ElasticNet", "auc_elasticnet",   C_ENET),',
    ']',
    'for ci, (lbl, col, col_c) in enumerate(groups_auc):',
    '    vals = subj_means(col)',
    '    if not vals: continue',
    '    jit = jrng.uniform(-0.08, 0.08, len(vals))',
    '    ax.scatter(np.full(len(vals), ci) + jit, vals,',
    '               color=col_c, alpha=0.75, s=60, zorder=3)',
    '    m  = float(np.mean(vals))',
    '    se = float(np.std(vals, ddof=1) / np.sqrt(len(vals))) if len(vals) > 1 else 0',
    '    ax.errorbar(ci, m, yerr=se, fmt="D", color=col_c,',
    '                markersize=11, capsize=6, lw=2.4, zorder=5)',
    '    ax.text(ci, m + se + 0.005, f"{m:.3f}", ha="center",',
    '            fontsize=9, color=col_c, fontweight="bold")',
    'ax.axhline(0.5, color="black", lw=0.8, ls="--", label="chance (0.5)")',
    'ax.set_ylim(0.35, 0.75)',
    'ax.set_xticks(range(len(groups_auc)))',
    'ax.set_xticklabels([g[0] for g in groups_auc], fontsize=11)',
    'ax.set_ylabel("ROC AUC  (pda_direct > 0 as positive)", fontsize=10)',
    'ax.set_title("C   AUC", fontsize=11, fontweight="bold")',
    'ax.legend(fontsize=9)',
    '',
    'plt.tight_layout()',
    'fig_path = OUT_DIR / "personalized_decoder_250Hz.png"',
    'plt.savefig(str(fig_path), dpi=150, bbox_inches="tight")',
    'plt.close(fig)',
    'print("  " + fig_path.name)',
    '',
    'print()',
    'print("=" * 60)',
    'print("DONE")',
    'print("=" * 60)',
]

# ── Save cluster script ─────────────────────────────────────────
script_name = "08_personalized_decoder_cluster.py"
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
job_name    = "ms_pers_decoder"
sbatch_name = "08_personalized_decoder.sh"
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
