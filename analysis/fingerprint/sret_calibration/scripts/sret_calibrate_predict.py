#!/usr/bin/env python3
"""
sret_calibrate_predict.py
-------------------------
Use SRET (self-referential encoding task) as calibration for EEG→fMRI decoding.

Strategy:
  1. Train group model on DMNELF (DWT+stats, Ridge 1000)
  2. Build SRET pseudo-target from events TSV (self=1, other=0.3, rest=0, HRF convolved)
  3. Extract DWT+stats from SRET EEG
  4. Calibrate: fit Ridge on SRET EEG features → SRET pseudo-target
  5. Predict feedback runs using calibrated model
  6. Compare: no-cal vs feedback-cal vs SRET-cal

Usage:
  python sret_calibrate_predict.py
"""
import sys, warnings, time, glob
from pathlib import Path
import numpy as np, pandas as pd, yaml
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from scipy.stats import pearsonr, gamma as gamma_dist

SCRIPT_DIR = Path(__file__).resolve().parent
PROJ_DIR = SCRIPT_DIR.parent
WAVELET_SCRIPTS = PROJ_DIR.parent / "wavelet_coupling" / "scripts"
EBC_SCRIPTS = PROJ_DIR.parent / "eeg_bold_coupling" / "scripts"
sys.path.insert(0, str(WAVELET_SCRIPTS))
sys.path.insert(0, str(EBC_SCRIPTS))

from bandpower_wavelet import gather_subject_wavelet, canonical_hrf, zscore, wavelet_power_run, hrf_convolve
from multivariate_decode_pda import (
    load_confounds_run, prepare_targets, car_and_flatten
)

warnings.filterwarnings("ignore")
import mne; mne.set_log_level("ERROR")

BAND_NAMES = ["delta", "theta", "alpha", "beta", "gamma"]
TARGETS = ["GSR_CEN", "PDA", "RAW_DMN"]
BASELINE_TRS = 25


def load_config(p):
    with open(p) as f:
        return yaml.safe_load(f)


def make_feedback_pseudo_target(n_tr, baseline_trs=25, tr=1.2):
    """Simple baseline→feedback step function, HRF convolved."""
    block = np.zeros(n_tr)
    block[baseline_trs:] = 1.0
    t_hrf = np.arange(0, 32, tr)
    h = gamma_dist.pdf(t_hrf, 6) - 0.35 * gamma_dist.pdf(t_hrf, 16)
    h = h / h.sum()
    return zscore(np.convolve(block, h, mode="full")[:n_tr])


def make_sret_pseudo_target(events_path, n_tr, tr=1.2, condition_weights=None):
    """Build SRET pseudo-target from events TSV.
    self=1.0, other=0.3, semantic=0, rest=0, convolved with HRF."""
    if condition_weights is None:
        condition_weights = {"self": 1.0, "other": 0.3, "semantic": 0.0}

    # Read events
    sep = "\t" if events_path.suffix == ".tsv" else ","
    df = pd.read_csv(events_path, sep=sep)

    # Build block regressor
    total_time = n_tr * tr
    regressor = np.zeros(n_tr)

    # Find block-level rows (self, other, semantic with duration > 10)
    for _, row in df.iterrows():
        tt = str(row.get("trial_type", ""))
        if tt in condition_weights and not pd.isna(row.get("duration")):
            dur = float(row["duration"])
            if dur < 10:
                continue
            onset_tr = int(float(row["onset"]) / tr)
            dur_trs = int(dur / tr)
            end_tr = min(onset_tr + dur_trs, n_tr)
            if onset_tr < n_tr:
                regressor[onset_tr:end_tr] = condition_weights[tt]

    # HRF convolve
    t_hrf = np.arange(0, 32, tr)
    h = gamma_dist.pdf(t_hrf, 6) - 0.35 * gamma_dist.pdf(t_hrf, 16)
    h = h / h.sum()
    convolved = np.convolve(regressor, h, mode="full")[:n_tr]
    return zscore(convolved)


def load_sret_eeg(cfg, sub, hrf):
    """Load SRET EEG and extract DWT+stats features."""
    d = cfg["data"]
    ses_eeg = d["session_eeg"]
    task = d["task_sret"]
    spt = d["eeg"]["samples_per_tr"]
    desc = d["eeg"]["desc"]
    eroot = Path(d["eeg_preproc_dir_cluster"])
    bands = cfg["bands"]

    runs = []
    for run in range(1, 5):
        fif = (eroot / f"sub-{sub}" / ses_eeg / "eeg" /
               f"sub-{sub}_{ses_eeg}_task-{task}_run-{run:02d}_desc-{desc}_eeg.fif")
        if not fif.exists():
            continue
        raw = mne.io.read_raw_fif(str(fif), preload=True, verbose=False)

        # Get n_tr from BOLD
        fmri_dir = Path(d["fmriprep_dir_cluster"]) / f"sub-{sub}" / d["session_fmri"] / "func"
        bold_files = sorted(fmri_dir.glob(f"sub-{sub}_{d['session_fmri']}_task-{task}_run-{run:02d}_*preproc_bold.nii.gz"))
        if not bold_files:
            continue
        import nibabel as nib
        n_tr = nib.load(str(bold_files[0])).shape[3]

        bp, chs = wavelet_power_run(raw, bands, spt, n_tr, hrf, method="dwt_stats")
        runs.append(dict(run=run, n_tr=n_tr, bp=bp, chs=chs))

    return runs


def load_feedback_data(cfg, sub, hrf):
    """Load feedback EEG with DWT+stats and fMRI targets."""
    d = cfg["data"]
    ses_eeg = d["session_eeg"]
    ses_fmri = d["session_fmri"]
    task = d["task_feedback"]
    spt = d["eeg"]["samples_per_tr"]
    desc = d["eeg"]["desc"]
    dmn_i = d["fmri"]["dmn_idx"]
    cen_i = d["fmri"]["cen_idx"]
    ndi = d["fmri"]["n_difumo"]
    fdir = Path(d["features_dir_cluster"]) / f"sub-{sub}"
    eroot = Path(d["eeg_preproc_dir_cluster"])
    bands = cfg["bands"]

    runs = []
    for npz in sorted(fdir.glob(f"sub-{sub}_task-{task}_run-*_features.npz")):
        z = np.load(npz, allow_pickle=True)
        fm = np.asarray(z["fmri_features"], float)
        n_tr = fm.shape[0]; run = npz.name.split("run-")[1][0]
        fif = (eroot / f"sub-{sub}" / ses_eeg / "eeg" /
               f"sub-{sub}_{ses_eeg}_task-{task}_run-{int(run):02d}_desc-{desc}_eeg.fif")
        if not fif.exists():
            continue
        raw = mne.io.read_raw_fif(str(fif), preload=True, verbose=False)
        bp, chs = wavelet_power_run(raw, bands, spt, n_tr, hrf, method="dwt_stats")
        targets = dict(DMN=fm[:, dmn_i], CEN=fm[:, cen_i], PDA=fm[:, cen_i] - fm[:, dmn_i])
        runs.append(dict(run=run, n_tr=n_tr, targets=targets, bp=bp, parcels=fm[:, :ndi], chs=chs))
    return runs


def load_confounds_rtbpd(cfg, sub, run_idx, task="feedback"):
    cdir = Path(cfg["data"]["confounds_dir_cluster"])
    ses = cfg["data"]["session_fmri"]
    tsv = (cdir / f"sub-{sub}" / ses / "func" /
           f"sub-{sub}_{ses}_task-{task}_run-{int(run_idx):02d}_desc-confounds_timeseries.tsv")
    df = pd.read_csv(tsv, sep="\t")
    gs = df["global_signal"].values.astype(float)
    gs[0] = gs[1]
    return gs


def find_events_file(cfg, sub, run):
    """Find SRET events file (TSV or CSV)."""
    events_dir = Path(cfg["data"]["events_dir_cluster"])
    # Try different naming conventions
    patterns = [
        events_dir / f"rtBPD{sub[-3:]}" / f"sub-{sub}_{cfg['data']['session_eeg']}_task-selfref_run-{run}_events.tsv",
        events_dir / f"rtBPD{sub[-3:]}" / f"sub-{sub}_{cfg['data']['session_eeg']}_task-selfref_run-{run}_events.csv",
        events_dir / f"{sub}" / f"sub-{sub}_{cfg['data']['session_eeg']}_task-selfref_run-{run}_events.tsv",
    ]
    for p in patterns:
        if p.exists():
            return p
    return None


def main():
    cfg = load_config(str(PROJ_DIR / "config.yaml"))
    cfg_dmnelf = {
        "data": {
            "features_dir": cfg["dmnelf"]["features_dir"],
            "eeg_preproc_dir": cfg["dmnelf"]["eeg_preproc_dir"],
            "confounds_dir": cfg["dmnelf"]["confounds_dir"],
            "session": cfg["dmnelf"]["session"],
            "task": "feedback",
            "eeg": cfg["data"]["eeg"],
            "fmri": cfg["data"]["fmri"],
        },
        "bands": cfg["bands"],
        "hrf": cfg["hrf"],
    }
    subjects = cfg["data"]["subjects"]["pilot"]
    hrf = canonical_hrf(tr=cfg["data"]["fmri"]["tr"], length_s=cfg["hrf"]["length_s"])
    tr = cfg["data"]["fmri"]["tr"]
    sret_contrasts = cfg["data"]["sret"]["contrasts"]
    out_dir = PROJ_DIR / "results"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("SRET Calibration: DMNELF group model + SRET calibration → feedback prediction")
    print(f"  Subjects: {subjects}")
    print(f"  SRET contrasts: {list(sret_contrasts.keys())}")
    print()

    # ── Load DMNELF group model ──
    print("Loading DMNELF training data...")
    t0 = time.time()
    train_X, train_targets = [], {t: [] for t in TARGETS}
    for sub in cfg["dmnelf"]["subjects"]:
        runs = gather_subject_wavelet(cfg_dmnelf, sub, hrf, method="dwt_stats")
        if not runs:
            continue
        confounds = [load_confounds_run(cfg_dmnelf, sub, rd["run"])[:rd["n_tr"]] for rd in runs]
        X, bounds = car_and_flatten(runs, BAND_NAMES)
        train_X.append(X)
        targets = prepare_targets(runs, confounds, TARGETS)
        for t in TARGETS:
            train_targets[t].append(targets[t])
    X_train = np.vstack(train_X)
    y_train = {t: np.concatenate(train_targets[t]) for t in TARGETS}
    scaler_grp = StandardScaler()
    X_train_sc = scaler_grp.fit_transform(X_train)
    group_models = {}
    for t in TARGETS:
        group_models[t] = Ridge(alpha=1000)
        group_models[t].fit(X_train_sc, y_train[t])
    print(f"  Group model: {X_train.shape[0]} TRs, {X_train.shape[1]} features\n")

    # ── Test each rtBPD subject ──
    results = []

    for sub in subjects:
        print(f"\n{'='*60}")
        print(f"  {sub}")
        print(f"{'='*60}")

        # Load feedback data
        fb_runs = load_feedback_data(cfg, sub, hrf)
        if not fb_runs:
            print("  No feedback data")
            continue
        fb_confounds = [load_confounds_rtbpd(cfg, sub, rd["run"], "feedback")[:rd["n_tr"]] for rd in fb_runs]
        X_fb, fb_bounds = car_and_flatten(fb_runs, BAND_NAMES)
        fb_targets = prepare_targets(fb_runs, fb_confounds, TARGETS)
        print(f"  Feedback: {X_fb.shape[0]} TRs, {len(fb_bounds)} runs")

        # Load SRET data
        sret_runs = load_sret_eeg(cfg, sub, hrf)
        if not sret_runs:
            print("  No SRET EEG data")
            continue
        X_sret, sret_bounds = car_and_flatten(sret_runs, BAND_NAMES)
        print(f"  SRET: {X_sret.shape[0]} TRs, {len(sret_bounds)} runs")

        # Build SRET pseudo-targets for each contrast
        sret_pseudos = {}  # contrast_name -> (full_pseudo, per_run_pieces)
        for contrast_name, cond_weights in sret_contrasts.items():
            pieces = []
            for ri, rd in enumerate(sret_runs):
                events_file = find_events_file(cfg, sub, rd["run"])
                if events_file is None:
                    pieces.append(np.zeros(rd["n_tr"]))
                else:
                    pseudo = make_sret_pseudo_target(events_file, rd["n_tr"], tr, cond_weights)
                    pieces.append(pseudo)
                    if contrast_name == list(sret_contrasts.keys())[0]:
                        print(f"    SRET run {rd['run']}: {rd['n_tr']} TRs, {events_file.name}")
            sret_pseudos[contrast_name] = (np.concatenate(pieces), pieces)

        # ── Strategy 1: No calibration (group only) ──
        X_fb_sc = scaler_grp.transform(X_fb)
        for t in TARGETS:
            pred = group_models[t].predict(X_fb_sc)
            r, _ = pearsonr(fb_targets[t], pred)
            results.append({"subject": sub, "target": t, "strategy": "no_cal", "r": r})

        # ── Strategy 2: Feedback pseudo-cal (1 run) ──
        cal_s, cal_e = fb_bounds[0]
        fb_pseudo = make_feedback_pseudo_target(cal_e - cal_s, BASELINE_TRS, tr)
        scaler_fb_cal = StandardScaler()
        model_fb_cal = Ridge(alpha=100)
        model_fb_cal.fit(scaler_fb_cal.fit_transform(X_fb[cal_s:cal_e]), fb_pseudo)
        test_idx = np.concatenate([np.arange(s, e) for s, e in fb_bounds[1:]])
        for t in TARGETS:
            pred = model_fb_cal.predict(scaler_fb_cal.transform(X_fb[test_idx]))
            r, _ = pearsonr(fb_targets[t][test_idx], pred)
            results.append({"subject": sub, "target": t, "strategy": "feedback_1run_cal", "r": r,
                            "n_test_runs": len(fb_bounds) - 1})

        # ── Strategy 3-5: SRET calibration (each contrast, all runs) ──
        for contrast_name in sret_contrasts:
            sret_pseudo, sret_pieces = sret_pseudos[contrast_name]
            scaler_sret = StandardScaler()
            model_sret = Ridge(alpha=100)
            model_sret.fit(scaler_sret.fit_transform(X_sret), sret_pseudo)
            for t in TARGETS:
                pred = model_sret.predict(scaler_sret.transform(X_fb))
                r, _ = pearsonr(fb_targets[t], pred)
                results.append({"subject": sub, "target": t,
                                "strategy": f"sret_{contrast_name}_all", "r": r})

            # Also test with 1 SRET run only
            sret_s, sret_e = sret_bounds[0]
            scaler_sret1 = StandardScaler()
            model_sret1 = Ridge(alpha=100)
            model_sret1.fit(scaler_sret1.fit_transform(X_sret[sret_s:sret_e]), sret_pieces[0])
            for t in TARGETS:
                pred = model_sret1.predict(scaler_sret1.transform(X_fb))
                r, _ = pearsonr(fb_targets[t], pred)
                results.append({"subject": sub, "target": t,
                                "strategy": f"sret_{contrast_name}_1run", "r": r})

        # Print summary for this subject
        print(f"\n  Results for {sub}:")
        for t in TARGETS:
            sub_df = [r for r in results if r["subject"] == sub and r["target"] == t]
            for r in sub_df:
                print(f"    {t:8s} {r['strategy']:20s} r={r['r']:+.4f}")

    # Save
    df = pd.DataFrame(results)
    df.to_csv(out_dir / "sret_calibration_results.csv", index=False)

    # Summary
    print(f"\n{'='*70}")
    print(f"  SUMMARY")
    print(f"{'='*70}")
    for t in TARGETS:
        print(f"\n  {t}:")
        for strat in sorted(df["strategy"].unique()):
            sub_df = df[(df["target"] == t) & (df["strategy"] == strat)]
            if len(sub_df) == 0:
                continue
            mean_r = sub_df["r"].mean()
            per_sub = "  ".join(f"{r['subject'][-3:]}={r['r']:+.2f}" for _, r in sub_df.iterrows())
            print(f"    {strat:35s}  mean r={mean_r:+.4f}  ({per_sub})")

    print(f"\nDone in {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
