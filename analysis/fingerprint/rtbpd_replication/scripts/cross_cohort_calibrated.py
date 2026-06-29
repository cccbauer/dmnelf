#!/usr/bin/env python3
"""
cross_cohort_calibrated.py
--------------------------
Train DWT+stats Ridge on DMNELF, apply to rtBPD with pseudo-target calibration.
Tests minimum calibration: 1, 2, 3 runs as calibration, predict remaining.

Usage:
  python cross_cohort_calibrated.py
"""
import sys, warnings, time
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

from bandpower_wavelet import gather_subject_wavelet, canonical_hrf, zscore, hrf_convolve
from multivariate_decode_pda import (
    load_confounds_run, prepare_targets, car_and_flatten, residualize
)

warnings.filterwarnings("ignore")
import mne; mne.set_log_level("ERROR")

BAND_NAMES = ["delta", "theta", "alpha", "beta", "gamma"]
TARGETS = ["GSR_CEN", "PDA", "RAW_DMN"]
BASELINE_TRS = 25  # ~24s baseline before feedback onset


def load_config(p):
    with open(p) as f:
        return yaml.safe_load(f)


def make_pseudo_target(n_tr, baseline_trs=20, tr=1.2):
    block = np.zeros(n_tr)
    block[baseline_trs:] = 1.0
    t_hrf = np.arange(0, 32, tr)
    h = gamma_dist.pdf(t_hrf, 6) - 0.35 * gamma_dist.pdf(t_hrf, 16)
    h = h / h.sum()
    return zscore(np.convolve(block, h, mode="full")[:n_tr])


def make_dmnelf_config(cfg):
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


def load_confounds_rtbpd(cfg, sub, run_idx):
    cdir = Path(cfg["data"]["confounds_dir_cluster"])
    ses = cfg["data"]["session_fmri"]
    task = cfg["data"]["task"]
    tsv = (cdir / f"sub-{sub}" / ses / "func" /
           f"sub-{sub}_{ses}_task-{task}_run-{int(run_idx):02d}_desc-confounds_timeseries.tsv")
    df = pd.read_csv(tsv, sep="\t")
    gs = df["global_signal"].values.astype(float)
    gs[0] = gs[1]
    return gs


def gather_rtbpd_subject(cfg, sub, hrf):
    d = cfg["data"]
    ses_eeg = d["session_eeg"]
    ses_fmri = d["session_fmri"]
    task = d["task"]
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
        from bandpower_wavelet import wavelet_power_run
        bp, chs = wavelet_power_run(raw, bands, spt, n_tr, hrf, method="dwt_stats")
        targets = dict(DMN=fm[:, dmn_i], CEN=fm[:, cen_i], PDA=fm[:, cen_i] - fm[:, dmn_i])
        runs.append(dict(run=run, n_tr=n_tr, targets=targets, bp=bp, parcels=fm[:, :ndi], chs=chs))
    return runs


def main():
    cfg = load_config(str(PROJ_DIR / "config.yaml"))
    cfg_dmnelf = make_dmnelf_config(cfg)
    dmnelf_subjects = cfg["dmnelf"]["subjects"]
    rtbpd_subjects = cfg["data"]["subjects"]["pilot"]
    hrf = canonical_hrf(tr=cfg["data"]["fmri"]["tr"], length_s=cfg["hrf"]["length_s"])
    out_dir = PROJ_DIR / "results"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Cross-Cohort Calibrated Prediction: DMNELF → rtBPD")
    print(f"  Training: {len(dmnelf_subjects)} DMNELF subjects")
    print(f"  Testing: {len(rtbpd_subjects)} rtBPD subjects")
    print()

    # ── Load DMNELF training data ──
    print("Loading DMNELF training data (DWT+stats)...")
    t0 = time.time()
    train_X_parts, train_targets = [], {t: [] for t in TARGETS}
    for sub in dmnelf_subjects:
        runs = gather_subject_wavelet(cfg_dmnelf, sub, hrf, method="dwt_stats")
        if not runs:
            continue
        confounds = [load_confounds_run(cfg_dmnelf, sub, rd["run"])[:rd["n_tr"]] for rd in runs]
        X, bounds = car_and_flatten(runs, BAND_NAMES)
        train_X_parts.append(X)
        targets = prepare_targets(runs, confounds, TARGETS)
        for t in TARGETS:
            train_targets[t].append(targets[t])
        print(f"  {sub}: {X.shape[0]} TRs")

    X_train = np.vstack(train_X_parts)
    y_train = {t: np.concatenate(train_targets[t]) for t in TARGETS}
    scaler_grp = StandardScaler()
    X_train_sc = scaler_grp.fit_transform(X_train)
    group_models = {}
    for t in TARGETS:
        group_models[t] = Ridge(alpha=1000)
        group_models[t].fit(X_train_sc, y_train[t])
    print(f"  Group models fitted ({X_train.shape[0]} TRs, {X_train.shape[1]} features)\n")

    # ── Test on rtBPD with varying calibration ──
    results = []
    cal_sizes = [0, 1, 2, 3]  # 0 = no calibration

    for sub in rtbpd_subjects:
        print(f"\n  {sub}:")
        runs = gather_rtbpd_subject(cfg, sub, hrf)
        if not runs:
            print("    No data")
            continue
        confounds = [load_confounds_rtbpd(cfg, sub, rd["run"])[:rd["n_tr"]] for rd in runs]
        X_all, run_bounds = car_and_flatten(runs, BAND_NAMES)
        all_targets = prepare_targets(runs, confounds, TARGETS)
        n_runs = len(run_bounds)
        print(f"    {X_all.shape[0]} TRs, {n_runs} runs")

        for n_cal in cal_sizes:
            if n_cal >= n_runs:
                continue

            if n_cal == 0:
                # No calibration — group model only
                X_test_sc = scaler_grp.transform(X_all)
                for t in TARGETS:
                    pred = group_models[t].predict(X_test_sc)
                    r, _ = pearsonr(all_targets[t], pred)
                    results.append({"subject": sub, "target": t, "n_cal_runs": 0,
                                    "r": r, "n_test_runs": n_runs, "method": "group_only"})
            else:
                # Calibrate on first n_cal runs, predict rest
                cal_idx = np.concatenate([np.arange(s, e) for s, e in run_bounds[:n_cal]])
                test_idx = np.concatenate([np.arange(s, e) for s, e in run_bounds[n_cal:]])

                X_cal = X_all[cal_idx]
                X_test = X_all[test_idx]

                # Build pseudo-target for calibration runs
                pseudo_pieces = []
                for s, e in run_bounds[:n_cal]:
                    pseudo_pieces.append(make_pseudo_target(e - s, BASELINE_TRS, cfg["data"]["fmri"]["tr"]))
                pseudo_y = np.concatenate(pseudo_pieces)

                # Calibrate
                scaler_cal = StandardScaler()
                cal_model = Ridge(alpha=100)
                cal_model.fit(scaler_cal.fit_transform(X_cal), pseudo_y)

                for t in TARGETS:
                    # Group prediction on test
                    pred_grp = group_models[t].predict(scaler_grp.transform(X_test))
                    r_grp, _ = pearsonr(all_targets[t][test_idx], pred_grp)

                    # Calibrated prediction on test
                    pred_cal = cal_model.predict(scaler_cal.transform(X_test))
                    r_cal, _ = pearsonr(all_targets[t][test_idx], pred_cal)

                    results.append({"subject": sub, "target": t, "n_cal_runs": n_cal,
                                    "r": r_cal, "n_test_runs": n_runs - n_cal,
                                    "r_group": r_grp, "method": "pseudo_cal"})

                    print(f"    {t:8s} cal={n_cal}: group r={r_grp:+.3f}  cal r={r_cal:+.3f}  "
                          f"(test {n_runs-n_cal} runs)")

    df = pd.DataFrame(results)
    df.to_csv(out_dir / "cross_cohort_calibrated.csv", index=False)

    # Summary
    print(f"\n{'='*70}")
    print(f"  SUMMARY: Effect of Calibration Size")
    print(f"{'='*70}")
    for t in TARGETS:
        print(f"\n  {t}:")
        for n_cal in cal_sizes:
            sub_df = df[(df["target"] == t) & (df["n_cal_runs"] == n_cal)]
            if len(sub_df) == 0:
                continue
            mean_r = sub_df["r"].mean()
            label = "no cal" if n_cal == 0 else f"{n_cal} run cal"
            per_sub = "  ".join(f"{r['subject'][-3:]}={r['r']:+.2f}" for _, r in sub_df.iterrows())
            print(f"    {label:12s}  mean r={mean_r:+.4f}  ({per_sub})")

    print(f"\nDone in {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
