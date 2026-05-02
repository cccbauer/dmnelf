# 04_train_decoder.py
# Run locally: python 04_train_decoder.py [--sfreq 250|500] [--overwrite]
# Deploys cluster script, submits SLURM job.
#
# What it does on the cluster:
#   For each subject:
#   1. Load TESS features (n_vols, 9) for feedback runs
#   2. Load PDA targets (group and personal) for feedback runs
#   3. Train ElasticNet decoder using Leave-One-Run-Out CV
#   4. Save decoder weights, CV scores, and predictions
#
# Two decoders per subject:
#   - decoder_group:    trained to predict PDA_group
#   - decoder_personal: trained to predict PDA_personal
#
# Output:
#   models/{subject}_{sfreq}Hz_decoder_group.pkl
#   models/{subject}_{sfreq}Hz_decoder_personal.pkl
#   models/{subject}_{sfreq}Hz_cv_results.json

import argparse
import py_compile
import time
from pathlib import Path
from utils import run_ssh, scp_to
from config import (
    CLUSTER_BASE, SLURM_ACCOUNT, PYTHON,
    SUBJECTS_EEG_FMRI_ALL, LOCAL_BASE,
    MISSING_RUNS, N_MICROSTATES
)

parser = argparse.ArgumentParser()
parser.add_argument("--sfreq", type=int, default=250,
                    choices=[250, 500])
parser.add_argument("--overwrite", action="store_true")
args      = parser.parse_args()
SFREQ_TAG = str(args.sfreq) + "Hz"

# ── 1. Build cluster-side script ───────────────────────────
lines = [
    '#!/usr/bin/env python3',
    '"""',
    '04_train_decoder_cluster.py',
    'Train ElasticNet EEG->PDA decoders using Leave-One-Run-Out CV.',
    'Two decoders per subject: PDA_group and PDA_personal.',
    '"""',
    'import sys',
    'sys.stdout.reconfigure(line_buffering=True)',
    'import numpy as np',
    'import json',
    'import pickle',
    'from pathlib import Path',
    'from sklearn.linear_model import ElasticNetCV',
    'from sklearn.preprocessing import StandardScaler',
    'from scipy.stats import pearsonr',
    '',
    '# ── Paths ─────────────────────────────────────────────',
    'CLUSTER_BASE  = Path("' + CLUSTER_BASE + '")',
    'FEATURES_DIR  = CLUSTER_BASE / "features"',
    'TARGETS_DIR   = CLUSTER_BASE / "targets"',
    'MODELS_DIR    = CLUSTER_BASE / "models"',
    'MODELS_DIR.mkdir(parents=True, exist_ok=True)',
    '',
    'SUBJECTS     = ' + str(SUBJECTS_EEG_FMRI_ALL),
    'MISSING_RUNS = ' + str(MISSING_RUNS),
    'OVERWRITE    = ' + str(args.overwrite),
    'SFREQ_TAG    = "' + SFREQ_TAG + '"',
    'N_MICROSTATES = ' + str(N_MICROSTATES),
    '',
    'FEEDBACK_RUNS = ["01", "02", "03", "04"]',
    '',
    '# Feature column names for reporting',
    'FEAT_NAMES = (["T-hat_" + chr(65+i) for i in range(N_MICROSTATES)]',
    '              + ["GFP", "GMD"])',
    '',
    '# ── Helper: load one run ──────────────────────────────',
    'def load_run(subject, task, run, pda_variant):',
    '    feat_name = (subject + "_task-" + task',
    '                 + "_run-" + run',
    '                 + "_" + SFREQ_TAG + "_nohrf_features.npy")',
    '    pda_name  = (subject + "_task-" + task',
    '                 + "_run-" + run',
    '                 + "_pda_" + pda_variant + ".npy")',
    '    feat_path = FEATURES_DIR / feat_name',
    '    pda_path  = TARGETS_DIR  / pda_name',
    '    if not feat_path.exists() or not pda_path.exists():',
    '        return None, None',
    '    X = np.load(str(feat_path)).astype(np.float64)',
    '    y = np.load(str(pda_path)).astype(np.float64)',
    '    # Trim to same length',
    '    n = min(len(X), len(y))',
    '    return X[:n], y[:n]',
    '',
    '# ── Helper: LORO CV ───────────────────────────────────',
    'def loro_cv(runs_X, runs_y):',
    '    """',
    '    Leave-One-Run-Out cross-validation.',
    '    Returns: predictions (concatenated), true values, per-run r.',
    '    """',
    '    all_pred = []',
    '    all_true = []',
    '    run_rs   = []',
    '',
    '    for left_out in range(len(runs_X)):',
    '        # Train on all other runs',
    '        X_train = np.concatenate([runs_X[i] for i in range(len(runs_X))',
    '                                   if i != left_out])',
    '        y_train = np.concatenate([runs_y[i] for i in range(len(runs_y))',
    '                                   if i != left_out])',
    '        X_test  = runs_X[left_out]',
    '        y_test  = runs_y[left_out]',
    '',
    '        # Scale',
    '        scaler_X = StandardScaler()',
    '        scaler_y = StandardScaler()',
    '        X_train_s = scaler_X.fit_transform(X_train)',
    '        y_train_s = scaler_y.fit_transform(y_train.reshape(-1, 1)).ravel()',
    '        X_test_s  = scaler_X.transform(X_test)',
    '',
    '        # Fit ElasticNet',
    '        from sklearn.linear_model import RidgeCV',
    '        model = RidgeCV(',
    '            alphas=[0.01, 0.1, 1.0, 10.0, 100.0, 1000.0],',
    '            cv=3',
    '        )',
    '        model.fit(X_train_s, y_train_s)',
    '',
    '        # Predict (unscale y)',
    '        y_pred_s = model.predict(X_test_s)',
    '        y_pred   = scaler_y.inverse_transform(',
    '            y_pred_s.reshape(-1, 1)).ravel()',
    '',
    '        r, _ = pearsonr(y_test, y_pred)',
    '        run_rs.append(float(r))',
    '        all_pred.append(y_pred)',
    '        all_true.append(y_test)',
    '',
    '    all_pred = np.concatenate(all_pred)',
    '    all_true = np.concatenate(all_true)',
    '    r_total, _ = pearsonr(all_true, all_pred)',
    '    return all_pred, all_true, run_rs, float(r_total)',
    '',
    '# ── Helper: train final model on all runs ─────────────',
    'def train_final(runs_X, runs_y):',
    '    X_all = np.concatenate(runs_X)',
    '    y_all = np.concatenate(runs_y)',
    '    scaler_X = StandardScaler()',
    '    scaler_y = StandardScaler()',
    '    X_s = scaler_X.fit_transform(X_all)',
    '    y_s = scaler_y.fit_transform(y_all.reshape(-1, 1)).ravel()',
    '    from sklearn.linear_model import RidgeCV',
    '    model = RidgeCV(',
    '        alphas=[0.01, 0.1, 1.0, 10.0, 100.0, 1000.0],',
    '        cv=3',
    '    )',
    '    model.fit(X_s, y_s)',
    '    return model, scaler_X, scaler_y',
    '',
    '# ── Main loop ─────────────────────────────────────────',
    'print("=" * 55)',
    'print("Training decoders  " + SFREQ_TAG)',
    'print("=" * 55)',
    '',
    'n_done    = 0',
    'n_skipped = 0',
    'n_failed  = 0',
    '',
    'for subject in SUBJECTS:',
    '    print()',
    '    print("--- " + subject + " ---")',
    '',
    '    for pda_variant in ["group", "personal"]:',
    '',
    '        out_model = MODELS_DIR / (subject + "_" + SFREQ_TAG',
    '                                  + "_decoder_" + pda_variant + ".pkl")',
    '        out_cv    = MODELS_DIR / (subject + "_" + SFREQ_TAG',
    '                                  + "_cv_" + pda_variant + ".json")',
    '',
    '        if out_model.exists() and out_cv.exists() and not OVERWRITE:',
    '            print("  EXISTS (skip): " + pda_variant)',
    '            n_skipped += 1',
    '            continue',
    '',
    '        # Load feedback runs',
    '        runs_X = []',
    '        runs_y = []',
    '        for run in FEEDBACK_RUNS:',
    '            if subject in MISSING_RUNS:',
    '                if ("feedback", run) in MISSING_RUNS[subject]:',
    '                    continue',
    '            X, y = load_run(subject, "feedback", run, pda_variant)',
    '            if X is not None:',
    '                runs_X.append(X)',
    '                runs_y.append(y)',
    '',
    '        if len(runs_X) < 2:',
    '            print("  SKIP " + pda_variant',
    '                  + ": need >= 2 runs, got " + str(len(runs_X)))',
    '            n_failed += 1',
    '            continue',
    '',
    '        print("  Training " + pda_variant',
    '              + "  runs=" + str(len(runs_X))',
    '              + "  total_vols=" + str(sum(len(x) for x in runs_X)))',
    '',
    '        # LORO CV',
    '        preds, trues, run_rs, r_total = loro_cv(runs_X, runs_y)',
    '',
    '        print("  LORO r=" + "{:.3f}".format(r_total)',
    '              + "  per-run: " + str([round(r, 3) for r in run_rs]))',
    '',
    '        # Train final model on all runs',
    '        model, scaler_X, scaler_y = train_final(runs_X, runs_y)',
    '',
    '        # Feature weights',
    '        weights = model.coef_',
    '        print("  Weights: " + " ".join([',
    '            n + "=" + "{:.3f}".format(w)',
    '            for n, w in zip(FEAT_NAMES, weights)',
    '        ]))',
    '',
    '        # Save model + scalers',
    '        with open(str(out_model), "wb") as f:',
    '            pickle.dump({',
    '                "model":    model,',
    '                "scaler_X": scaler_X,',
    '                "scaler_y": scaler_y,',
    '                "feat_names": FEAT_NAMES,',
    '            }, f)',
    '',
    '        # Save CV results',
    '        cv_results = {',
    '            "subject":      subject,',
    '            "pda_variant":  pda_variant,',
    '            "sfreq_tag":    SFREQ_TAG,',
    '            "n_runs":       len(runs_X),',
    '            "loro_r":       r_total,',
    '            "loro_r_sq":    float(r_total ** 2),',
    '            "run_rs":       run_rs,',
    '            "alpha":        float(model.alpha_),',
    '            "l1_ratio":     float(model.l1_ratio_),',
    '            "weights":      weights.tolist(),',
    '            "feat_names":   FEAT_NAMES,',
    '        }',
    '        with open(str(out_cv), "w") as f:',
    '            json.dump(cv_results, f, indent=2)',
    '',
    '        print("  Saved: " + out_model.name)',
    '        n_done += 1',
    '',
    'print()',
    'print("=" * 55)',
    'print("DONE")',
    'print("  Trained:  " + str(n_done))',
    'print("  Skipped:  " + str(n_skipped))',
    'print("  Failed:   " + str(n_failed))',
    'print("=" * 55)',
]

# ── 2. Save cluster script locally ─────────────────────────
script_name = "04_train_decoder_" + SFREQ_TAG + "_cluster.py"
script_path = LOCAL_BASE / "scripts" / script_name
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

# ── 4. Build SLURM script ──────────────────────────────────
job_name  = "train_dec_" + SFREQ_TAG
run_cmd   = (PYTHON + " " + CLUSTER_BASE + "/scripts/" + script_name
             + (" --overwrite" if args.overwrite else ""))

sbatch_lines = [
    "#!/bin/bash",
    "#SBATCH --job-name=" + job_name,
    "#SBATCH --output=" + CLUSTER_BASE + "/logs/" + job_name + "_%j.out",
    "#SBATCH --error="  + CLUSTER_BASE + "/logs/" + job_name + "_%j.err",
    "#SBATCH --partition=short",
    "#SBATCH --time=04:00:00",
    "#SBATCH --cpus-per-task=8",
    "#SBATCH --mem=32G",
    "#SBATCH --account=" + SLURM_ACCOUNT,
    "",
    run_cmd,
]

sbatch_name = "04_train_decoder_" + SFREQ_TAG + ".sh"
sbatch_path = LOCAL_BASE / "scripts" / sbatch_name
with open(sbatch_path, "w") as f:
    f.write("\n".join(sbatch_lines))

# ── 5. Deploy ──────────────────────────────────────────────
print("\nDeploying...")
scp_to(script_path,
       CLUSTER_BASE + "/scripts/" + script_name,
       verbose=False)
scp_to(sbatch_path,
       CLUSTER_BASE + "/scripts/" + sbatch_name,
       verbose=False)
print("Deployed: " + script_name)
print("Command:  " + run_cmd)

# ── 6. Submit ──────────────────────────────────────────────
print("\nSubmitting SLURM job...")
result = run_ssh("sbatch " + CLUSTER_BASE + "/scripts/" + sbatch_name)
job_id = ""
for line in result.stdout.strip().split("\n"):
    if "Submitted" in line:
        job_id = line.strip().split()[-1]
        print("Job ID: " + job_id)

# ── 7. Monitor ─────────────────────────────────────────────
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