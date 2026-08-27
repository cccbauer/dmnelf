#!/usr/bin/env python3
"""
cross_cohort_predict.py
-----------------------
Train DWT+stats Ridge model on DMNELF cohort (n=17),
predict on rtBPD cohort (n=3 pilot) — true cross-cohort validation.

Usage:
  python cross_cohort_predict.py
"""
import sys, warnings, time
import argparse
from pathlib import Path
import numpy as np, pandas as pd, yaml
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from scipy.stats import pearsonr

SCRIPT_DIR = Path(__file__).resolve().parent
PROJ_DIR = SCRIPT_DIR.parent

# Import from existing projects (local repo layout and cluster DMNELF layout)
IMPORT_ROOTS = [
    PROJ_DIR.parent,
    Path("/projects/swglab/data/DMNELF/analysis/fingerprint"),
]
for root in IMPORT_ROOTS:
    wavelet = root / "wavelet_coupling" / "scripts"
    ebc = root / "eeg_bold_coupling" / "scripts"
    if wavelet.exists():
        sys.path.insert(0, str(wavelet))
    if ebc.exists():
        sys.path.insert(0, str(ebc))

from bandpower_wavelet import gather_subject_wavelet, canonical_hrf, zscore, hrf_convolve
from multivariate_decode_pda import (
    load_confounds_run, prepare_targets, car_and_flatten, residualize
)

warnings.filterwarnings("ignore")
import mne; mne.set_log_level("ERROR")

BAND_NAMES = ["delta", "theta", "alpha", "beta", "gamma"]
TARGETS = ["GSR_CEN", "PDA", "RAW_DMN", "GSR_DMN"]


def parse_args():
    ap = argparse.ArgumentParser(description="Cross-cohort DMNELF to rtBPD prediction")
    ap.add_argument(
        "--subjects-set",
        choices=["pilot", "all"],
        default="pilot",
        help="Which rtBPD subject set from config.yaml to evaluate (default: pilot)",
    )
    return ap.parse_args()


def load_config(p):
    with open(p) as f:
        return yaml.safe_load(f)


def make_dmnelf_config(cfg):
    """Create a config dict that points to DMNELF data."""
    dmnelf = cfg["dmnelf"]
    return {
        "data": {
            "features_dir": dmnelf["features_dir"],
            "eeg_preproc_dir": dmnelf["eeg_preproc_dir"],
            "confounds_dir": dmnelf["confounds_dir"],
            "session": dmnelf["session"],
            "task": "feedback",
            "eeg": cfg["data"]["eeg"],
            "fmri": cfg["data"]["fmri"],
        },
        "bands": cfg["bands"],
        "hrf": cfg["hrf"],
    }


def make_rtbpd_config(cfg):
    """Create a config dict that points to rtBPD data."""
    d = cfg["data"]
    suffix = "_cluster" if Path("/projects/swglab").exists() else "_local"
    return {
        "data": {
            "features_dir": d.get("features_dir" + suffix, d.get("features_dir_cluster")),
            "eeg_preproc_dir": d.get("eeg_preproc_dir" + suffix, d.get("eeg_preproc_dir_cluster")),
            "confounds_dir": d.get("confounds_dir" + suffix, d.get("confounds_dir_cluster")),
            "session": d["session_eeg"],  # ses-nf for EEG files
            "session_fmri": d["session_fmri"],  # ses-nf1 for fMRI files
            "task": d["task"],
            "eeg": d["eeg"],
            "fmri": d["fmri"],
        },
        "bands": cfg["bands"],
        "hrf": cfg["hrf"],
    }


def load_confounds_rtbpd(cfg_rtbpd, sub, run_idx):
    """Load global_signal from fMRIPrep confounds for rtBPD (handles session mapping)."""
    cdir = Path(cfg_rtbpd["data"]["confounds_dir"])
    ses = cfg_rtbpd["data"]["session_fmri"]  # ses-nf1
    task = cfg_rtbpd["data"]["task"]
    tsv = (cdir / f"sub-{sub}" / ses / "func" /
           f"sub-{sub}_{ses}_task-{task}_run-{int(run_idx):02d}_desc-confounds_timeseries.tsv")
    df = pd.read_csv(tsv, sep="\t")
    gs = df["global_signal"].values.astype(float)
    gs[0] = gs[1]
    return gs


def resolve_rtbpd_eeg_session(cfg_rtbpd, sub):
    """Resolve EEG session label for a subject without mutating raw data labels."""
    d = cfg_rtbpd["data"]
    eroot = Path(d["eeg_preproc_dir"])
    preferred = d["session"]
    candidates = [preferred]
    if preferred == "ses-nf":
        candidates.append("ses-nf1")

    for ses in candidates:
        eeg_dir = eroot / f"sub-{sub}" / ses / "eeg"
        if eeg_dir.exists() and any(eeg_dir.glob("*_eeg.fif")):
            return ses
    return preferred


def gather_rtbpd_subject(cfg_rtbpd, sub, hrf):
    """Load rtBPD subject's feedback data with DWT+stats features.
    Handles session mapping: EEG in ses-nf, fMRI features in ses-nf1."""
    d = cfg_rtbpd["data"]
    ses_eeg = resolve_rtbpd_eeg_session(cfg_rtbpd, sub)
    task = d["task"]
    spt = d["eeg"]["samples_per_tr"]
    desc = d["eeg"]["desc"]
    dmn_i = d["fmri"]["dmn_idx"]
    cen_i = d["fmri"]["cen_idx"]
    ndi = d["fmri"]["n_difumo"]

    fdir = Path(d["features_dir"]) / f"sub-{sub}"
    eroot = Path(d["eeg_preproc_dir"])
    bands = cfg_rtbpd["bands"]

    runs = []
    for npz in sorted(fdir.glob(f"sub-{sub}_task-{task}_run-*_features.npz")):
        z = np.load(npz, allow_pickle=True)
        fm = np.asarray(z["fmri_features"], float)
        n_tr = fm.shape[0]
        run = npz.stem.split("run-")[1].split("_")[0]

        fif = (eroot / f"sub-{sub}" / ses_eeg / "eeg" /
               f"sub-{sub}_{ses_eeg}_task-{task}_run-{int(run):02d}_desc-{desc}_eeg.fif")
        if not fif.exists():
            print(f"    EEG not found: {fif.name}")
            continue

        raw = mne.io.read_raw_fif(str(fif), preload=True, verbose=False)

        from bandpower_wavelet import wavelet_power_run
        bp, chs = wavelet_power_run(raw, bands, spt, n_tr, hrf, method="dwt_stats")

        targets = dict(DMN=fm[:, dmn_i], CEN=fm[:, cen_i],
                       PDA=fm[:, cen_i] - fm[:, dmn_i])
        runs.append(dict(run=run, n_tr=n_tr, targets=targets,
                         bp=bp, parcels=fm[:, :ndi], chs=chs))
    return runs, ses_eeg


def main():
    args = parse_args()
    cfg = load_config(str(PROJ_DIR / "config.yaml"))
    cfg_dmnelf = make_dmnelf_config(cfg)
    cfg_rtbpd = make_rtbpd_config(cfg)

    dmnelf_subjects = cfg["dmnelf"]["subjects"]
    rtbpd_subjects = cfg["data"]["subjects"][args.subjects_set]
    exclude = set(cfg["data"]["subjects"].get("exclude", []))
    rtbpd_subjects = [s for s in rtbpd_subjects if s not in exclude]
    hrf = canonical_hrf(tr=cfg["data"]["fmri"]["tr"], length_s=cfg["hrf"]["length_s"])

    out_dir = PROJ_DIR / "results"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Cross-Cohort Prediction: DMNELF → rtBPD")
    print(f"  Training: {len(dmnelf_subjects)} DMNELF subjects")
    print(f"  Testing ({args.subjects_set}): {len(rtbpd_subjects)} rtBPD subjects")
    print()

    # ── Load DMNELF training data (DWT+stats) ──
    print("Loading DMNELF training data...")
    t0 = time.time()
    train_X_parts, train_targets = [], {t: [] for t in TARGETS}

    for sub in dmnelf_subjects:
        runs = gather_subject_wavelet(cfg_dmnelf, sub, hrf, method="dwt_stats")
        if not runs:
            print(f"  {sub}: SKIPPED")
            continue
        confounds = [load_confounds_run(cfg_dmnelf, sub, rd["run"])[:rd["n_tr"]] for rd in runs]
        X, bounds = car_and_flatten(runs, BAND_NAMES)
        train_X_parts.append(X)
        targets = prepare_targets(runs, confounds, TARGETS)
        for t in TARGETS:
            train_targets[t].append(targets[t])
        print(f"  {sub}: {X.shape[0]} TRs, {X.shape[1]} features")

    X_train = np.vstack(train_X_parts)
    y_train = {t: np.concatenate(train_targets[t]) for t in TARGETS}
    print(f"  Total training: {X_train.shape[0]} TRs, {X_train.shape[1]} features")

    # ── Fit group model ──
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)

    models = {}
    for t in TARGETS:
        models[t] = Ridge(alpha=1000)
        models[t].fit(X_train_sc, y_train[t])
    print(f"  Models fitted in {time.time()-t0:.0f}s\n")

    # ── Predict on rtBPD subjects ──
    print("Predicting on rtBPD subjects...")
    results = []

    for sub in rtbpd_subjects:
        print(f"\n  {sub}:")
        runs, ses_eeg = gather_rtbpd_subject(cfg_rtbpd, sub, hrf)
        print(f"    Session map: EEG={ses_eeg}, fMRI={cfg_rtbpd['data']['session_fmri']}")
        if not runs:
            print(f"    No data, skipping")
            continue

        confounds = [load_confounds_rtbpd(cfg_rtbpd, sub, rd["run"])[:rd["n_tr"]] for rd in runs]
        X_test, test_bounds = car_and_flatten(runs, BAND_NAMES)
        test_targets = prepare_targets(runs, confounds, TARGETS)
        X_test_sc = scaler.transform(X_test)

        print(f"    {X_test.shape[0]} TRs, {len(test_bounds)} runs, {X_test.shape[1]} features")

        for t in TARGETS:
            pred = models[t].predict(X_test_sc)
            y_test = test_targets[t]
            r_all, _ = pearsonr(y_test, pred)

            run_rs = []
            for ri, (start, end) in enumerate(test_bounds):
                r_run, _ = pearsonr(y_test[start:end], pred[start:end])
                run_rs.append(r_run)

            row = {"cohort": "rtBPD", "subject": sub, "target": t,
                   "overall_r": r_all, "n_runs": len(test_bounds),
                   "n_trs": X_test.shape[0], "n_features": X_test.shape[1]}
            for ri, rr in enumerate(run_rs):
                row[f"run{ri+1}_r"] = rr
            results.append(row)

            run_str = "  ".join(f"r{ri+1}={rr:+.2f}" for ri, rr in enumerate(run_rs))
            print(f"    {t:8s}  overall r={r_all:+.4f}  {run_str}")

    # Save
    df = pd.DataFrame(results)
    df.to_csv(out_dir / "cross_cohort_results.csv", index=False)

    # Summary
    print(f"\n{'='*70}")
    print(f"  CROSS-COHORT SUMMARY (DMNELF → rtBPD)")
    print(f"{'='*70}")
    for t in TARGETS:
        sub_df = df[df["target"] == t]
        if len(sub_df) == 0:
            continue
        mean_r = sub_df["overall_r"].mean()
        print(f"  {t:8s}  mean r={mean_r:+.4f}  (n={len(sub_df)})")
        for _, row in sub_df.iterrows():
            print(f"    {row['subject']}: r={row['overall_r']:+.4f}")

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed/60:.1f} min")
    print(f"Results: {out_dir / 'cross_cohort_results.csv'}")


if __name__ == "__main__":
    main()
