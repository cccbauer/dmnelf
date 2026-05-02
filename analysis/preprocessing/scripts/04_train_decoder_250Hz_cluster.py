#!/usr/bin/env python3
"""
04_train_decoder_cluster.py
Train ElasticNet EEG->PDA decoders using Leave-One-Run-Out CV.
Two decoders per subject: PDA_group and PDA_personal.
"""
import sys
sys.stdout.reconfigure(line_buffering=True)
import numpy as np
import json
import pickle
from pathlib import Path
from sklearn.linear_model import ElasticNetCV
from sklearn.preprocessing import StandardScaler
from scipy.stats import pearsonr

# ── Paths ─────────────────────────────────────────────
CLUSTER_BASE  = Path("/projects/swglab/data/DMNELF/analysis/MNE/jupyter/microstate_pda_v3")
FEATURES_DIR  = CLUSTER_BASE / "features"
TARGETS_DIR   = CLUSTER_BASE / "targets"
MODELS_DIR    = CLUSTER_BASE / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

SUBJECTS     = ['sub-dmnelf001', 'sub-dmnelf004', 'sub-dmnelf005', 'sub-dmnelf006', 'sub-dmnelf007', 'sub-dmnelf008', 'sub-dmnelf009', 'sub-dmnelf010', 'sub-dmnelf011', 'sub-dmnelf1001', 'sub-dmnelf1002', 'sub-dmnelf1003']
MISSING_RUNS = {'sub-dmnelf1002': [('rest', '02')], 'sub-dmnelf1003': [('feedback', '04')]}
OVERWRITE    = True
SFREQ_TAG    = "250Hz"
N_MICROSTATES = 7

FEEDBACK_RUNS = ["01", "02", "03", "04"]

# Feature column names for reporting
FEAT_NAMES = (["T-hat_" + chr(65+i) for i in range(N_MICROSTATES)]
              + ["GFP", "GMD"])

# ── Helper: load one run ──────────────────────────────
def load_run(subject, task, run, pda_variant):
    feat_name = (subject + "_task-" + task
                 + "_run-" + run
                 + "_" + SFREQ_TAG + "_nohrf_features.npy")
    pda_name  = (subject + "_task-" + task
                 + "_run-" + run
                 + "_pda_" + pda_variant + ".npy")
    feat_path = FEATURES_DIR / feat_name
    pda_path  = TARGETS_DIR  / pda_name
    if not feat_path.exists() or not pda_path.exists():
        return None, None
    X = np.load(str(feat_path)).astype(np.float64)
    y = np.load(str(pda_path)).astype(np.float64)
    # Trim to same length
    n = min(len(X), len(y))
    return X[:n], y[:n]

# ── Helper: LORO CV ───────────────────────────────────
def loro_cv(runs_X, runs_y):
    """
    Leave-One-Run-Out cross-validation.
    Returns: predictions (concatenated), true values, per-run r.
    """
    all_pred = []
    all_true = []
    run_rs   = []

    for left_out in range(len(runs_X)):
        # Train on all other runs
        X_train = np.concatenate([runs_X[i] for i in range(len(runs_X))
                                   if i != left_out])
        y_train = np.concatenate([runs_y[i] for i in range(len(runs_y))
                                   if i != left_out])
        X_test  = runs_X[left_out]
        y_test  = runs_y[left_out]

        # Scale
        scaler_X = StandardScaler()
        scaler_y = StandardScaler()
        X_train_s = scaler_X.fit_transform(X_train)
        y_train_s = scaler_y.fit_transform(y_train.reshape(-1, 1)).ravel()
        X_test_s  = scaler_X.transform(X_test)

        # Fit ElasticNet
        from sklearn.linear_model import RidgeCV
        model = RidgeCV(
            alphas=[0.01, 0.1, 1.0, 10.0, 100.0, 1000.0],
            cv=3
        )
        model.fit(X_train_s, y_train_s)

        # Predict (unscale y)
        y_pred_s = model.predict(X_test_s)
        y_pred   = scaler_y.inverse_transform(
            y_pred_s.reshape(-1, 1)).ravel()

        r, _ = pearsonr(y_test, y_pred)
        run_rs.append(float(r))
        all_pred.append(y_pred)
        all_true.append(y_test)

    all_pred = np.concatenate(all_pred)
    all_true = np.concatenate(all_true)
    r_total, _ = pearsonr(all_true, all_pred)
    return all_pred, all_true, run_rs, float(r_total)

# ── Helper: train final model on all runs ─────────────
def train_final(runs_X, runs_y):
    X_all = np.concatenate(runs_X)
    y_all = np.concatenate(runs_y)
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    X_s = scaler_X.fit_transform(X_all)
    y_s = scaler_y.fit_transform(y_all.reshape(-1, 1)).ravel()
    from sklearn.linear_model import RidgeCV
    model = RidgeCV(
        alphas=[0.01, 0.1, 1.0, 10.0, 100.0, 1000.0],
        cv=3
    )
    model.fit(X_s, y_s)
    return model, scaler_X, scaler_y

# ── Main loop ─────────────────────────────────────────
print("=" * 55)
print("Training decoders  " + SFREQ_TAG)
print("=" * 55)

n_done    = 0
n_skipped = 0
n_failed  = 0

for subject in SUBJECTS:
    print()
    print("--- " + subject + " ---")

    for pda_variant in ["group", "personal"]:

        out_model = MODELS_DIR / (subject + "_" + SFREQ_TAG
                                  + "_decoder_" + pda_variant + ".pkl")
        out_cv    = MODELS_DIR / (subject + "_" + SFREQ_TAG
                                  + "_cv_" + pda_variant + ".json")

        if out_model.exists() and out_cv.exists() and not OVERWRITE:
            print("  EXISTS (skip): " + pda_variant)
            n_skipped += 1
            continue

        # Load feedback runs
        runs_X = []
        runs_y = []
        for run in FEEDBACK_RUNS:
            if subject in MISSING_RUNS:
                if ("feedback", run) in MISSING_RUNS[subject]:
                    continue
            X, y = load_run(subject, "feedback", run, pda_variant)
            if X is not None:
                runs_X.append(X)
                runs_y.append(y)

        if len(runs_X) < 2:
            print("  SKIP " + pda_variant
                  + ": need >= 2 runs, got " + str(len(runs_X)))
            n_failed += 1
            continue

        print("  Training " + pda_variant
              + "  runs=" + str(len(runs_X))
              + "  total_vols=" + str(sum(len(x) for x in runs_X)))

        # LORO CV
        preds, trues, run_rs, r_total = loro_cv(runs_X, runs_y)

        print("  LORO r=" + "{:.3f}".format(r_total)
              + "  per-run: " + str([round(r, 3) for r in run_rs]))

        # Train final model on all runs
        model, scaler_X, scaler_y = train_final(runs_X, runs_y)

        # Feature weights
        weights = model.coef_
        print("  Weights: " + " ".join([
            n + "=" + "{:.3f}".format(w)
            for n, w in zip(FEAT_NAMES, weights)
        ]))

        # Save model + scalers
        with open(str(out_model), "wb") as f:
            pickle.dump({
                "model":    model,
                "scaler_X": scaler_X,
                "scaler_y": scaler_y,
                "feat_names": FEAT_NAMES,
            }, f)

        # Save CV results
        cv_results = {
            "subject":      subject,
            "pda_variant":  pda_variant,
            "sfreq_tag":    SFREQ_TAG,
            "n_runs":       len(runs_X),
            "loro_r":       r_total,
            "loro_r_sq":    float(r_total ** 2),
            "run_rs":       run_rs,
            "alpha":        float(model.alpha_),
            "l1_ratio":     float(model.l1_ratio_),
            "weights":      weights.tolist(),
            "feat_names":   FEAT_NAMES,
        }
        with open(str(out_cv), "w") as f:
            json.dump(cv_results, f, indent=2)

        print("  Saved: " + out_model.name)
        n_done += 1

print()
print("=" * 55)
print("DONE")
print("  Trained:  " + str(n_done))
print("  Skipped:  " + str(n_skipped))
print("  Failed:   " + str(n_failed))
print("=" * 55)