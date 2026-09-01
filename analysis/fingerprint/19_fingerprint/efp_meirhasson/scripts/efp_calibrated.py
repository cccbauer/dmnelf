#!/usr/bin/env python3
"""
efp_calibrated.py
-----------------
Per-individual PSEUDO-TARGET calibration of the general EFP fingerprint.

Every feedback run has a known design: ~30 s rest, then the noting-practice task
(CEN up, DMN down). So the FIRST feedback run(s) can calibrate the general EFP to the
individual using only that known design — a task-design "pseudo-target" (boxcar,
HRF-convolved) — with NO held-out fMRI. We then predict the individual's REMAINING runs
and score against their real fMRI. This is the deployable EEG-only neurofeedback scenario.

Methods compared per (subject, target), over calibration sizes n_cal in {0,1,2,3}:
  group_only  : the general EFP model (train on other subjects), no calibration [baseline]
  pseudo_cal  : a fresh per-subject RidgeCV trained on the cal run(s) pseudo-target
  blend       : mean of z-scored group_only and pseudo_cal predictions (general + individual)
(affine rescaling is omitted — it does not change Pearson r.)

Settings:
  --mode loso   : leave-one-subject-out within DMNELF (real fMRI on all runs -> honest gain).
                  Electrode = per-fold group-peak over TRAINING subjects only (leak-free).
  --mode cross  : train general model on ALL DMNELF, calibrate+predict each rtBPD subject
                  (--rtbpd-config for nf1/nf2). Electrode = all-DMNELF peak (loso common_ch).

Pseudo-target sign is target-specific: +boxcar for CEN/PDA/GSR_CEN/GSR_PDA (up during task),
-boxcar for DMN/GSR_DMN (down during task).

Usage (cluster):
  python efp_calibrated.py --mode loso
  python efp_calibrated.py --mode cross --rtbpd-config ../config_rtbpd.yaml --tag ""
  python efp_calibrated.py --mode cross --rtbpd-config ../config_rtbpd_nf2.yaml --tag _nf2
"""
import argparse, warnings
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import pearsonr
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeCV

from efp_features import load_config, load_subject_features, make_delay_design
from efp_decode import canonical_hrf, zscore, hrf_convolve
from cross_cohort_efp import n_delays_for, alphas_from, electrode_map, stack_design, sign_flip_p

warnings.filterwarnings("ignore")
PROJ = Path(__file__).resolve().parent.parent
RES = PROJ / "results" / "full"
OUT = PROJ / "results"
TARGETS = ["PDA", "GSR_CEN", "CEN", "DMN", "GSR_PDA", "GSR_DMN"]
PSEUDO_SIGN = {"CEN": +1, "PDA": +1, "GSR_CEN": +1, "GSR_PDA": +1, "DMN": -1, "GSR_DMN": -1}
CAL_SIZES = [0, 1, 2, 3]
FRONTAL_ELECTRODES = {"Fp1", "Fp2", "F3", "F4", "F7", "F8", "Fz", "FC1", "FC2", "FC5", "FC6"}
# the ACTUAL mindwear-deployed portable montage (export_model.py/train_pooled.py EPOC12) --
# distinct from FRONTAL_ELECTRODES above, which is a generic "frontal headset" proxy, not this
# device's real channel list (posterior/temporal T7/P7/O1/O2/P8/T8 included, Fp1/Fp2/Fz/FC1/FC2 not)
EPOC12_ELECTRODES = ["F7", "F3", "FC5", "T7", "P7", "O1", "O2", "P8", "T8", "FC6", "F4", "F8"]


def group_peak(er, target, res, exclude=None, restrict=None):
    """Group-mean-r peak electrode for a target from electrode_r_all.csv, optionally
    excluding one subject (leak-free per-fold selection for LOSO).
    restrict: optional set of electrode names to limit candidate pool."""
    d = er[(er.target == target) & (er.resolution == res)]
    if exclude is not None:
        d = d[d.subject != exclude]
    if restrict is not None:
        d = d[d.electrode.isin(restrict)]
    gm = d.groupby("electrode")["r"].agg(["mean", "count"])
    gm = gm[gm["count"] >= 5]
    return gm["mean"].idxmax() if len(gm) else None


def run_designs(runs, ci, target, res, n_delays, baseline_trs, hrf, sign):
    """Per-run (X, y_real, pseudo) at channel ci. X per-run standardized (matches assemble)."""
    bp_key = "bp_tr" if res == "tr" else "bp_hz4"
    tg_key = "tgt_tr" if res == "tr" else "tgt_hz4"
    out = []
    for rd in runs:
        if target not in rd[tg_key]:
            continue
        X, off = make_delay_design(rd[bp_key][ci], n_delays)
        if X.shape[0] == 0:
            continue
        X = (X - X.mean(0)) / (X.std(0) + 1e-12)
        nv = X.shape[0]
        y = zscore(rd[tg_key][target][off:off + nv])
        block = np.zeros(rd[bp_key][ci].shape[1]); block[baseline_trs:] = 1.0
        pseudo = sign * zscore(hrf_convolve(block, hrf)[off:off + nv])
        out.append((X, y, pseudo))
    return out


def run_designs_multi(runs, cis, target, res, n_delays, baseline_trs, hrf, sign):
    """Per-run (X, y, pseudo) with delay features from multiple electrodes concatenated.
    cis: ordered list of channel indices — must be consistent between train and test."""
    bp_key = "bp_tr" if res == "tr" else "bp_hz4"
    tg_key = "tgt_tr" if res == "tr" else "tgt_hz4"
    out = []
    for rd in runs:
        if target not in rd[tg_key]:
            continue
        Xparts, off, ok = [], None, True
        for ci in cis:
            Xi, o = make_delay_design(rd[bp_key][ci], n_delays)
            if Xi.shape[0] == 0:
                ok = False; break
            Xparts.append((Xi - Xi.mean(0)) / (Xi.std(0) + 1e-12)); off = o
        if not ok or not Xparts:
            continue
        X = np.hstack(Xparts)
        nv = X.shape[0]
        y = zscore(rd[tg_key][target][off:off + nv])
        block = np.zeros(rd[bp_key][cis[0]].shape[1]); block[baseline_trs:] = 1.0
        pseudo = sign * zscore(hrf_convolve(block, hrf)[off:off + nv])
        out.append((X, y, pseudo))
    return out


def stack_design_multi(cache, subs, electrodes, target, res, n_delays):
    """Multi-electrode delay features stacked across training subjects.
    electrodes: sorted list defining consistent column order across all subjects."""
    bp_key = "bp_tr" if res == "tr" else "bp_hz4"
    tg_key = "tgt_tr" if res == "tr" else "tgt_hz4"
    Xs, ys = [], []
    for sub in subs:
        try:
            runs, chs = load_subject_features(Path(cache), sub)
        except Exception:
            continue
        cis = [chs.index(el) for el in electrodes if el in chs]
        if len(cis) != len(electrodes):
            continue  # skip subjects with incomplete frontal montage
        for rd in runs:
            if target not in rd[tg_key]:
                continue
            Xparts, off, ok = [], None, True
            for ci in cis:
                Xi, o = make_delay_design(rd[bp_key][ci], n_delays)
                if Xi.shape[0] == 0:
                    ok = False; break
                Xparts.append((Xi - Xi.mean(0)) / (Xi.std(0) + 1e-12)); off = o
            if not ok or not Xparts:
                continue
            X = np.hstack(Xparts)
            nv = X.shape[0]
            y = zscore(rd[tg_key][target][off:off + nv])
            if len(y) >= 10:
                Xs.append(X); ys.append(y)
    return (np.vstack(Xs), np.concatenate(ys)) if Xs else (None, None)


def proxy_pda_run_designs(runs, ci_cen, ci_dmn, res, n_delays):
    """Per-run (X_cen, X_dmn, y_pda) for proxy_PDA = zscore(pred_CEN) - zscore(pred_DMN)."""
    bp_key = "bp_tr" if res == "tr" else "bp_hz4"
    tg_key = "tgt_tr" if res == "tr" else "tgt_hz4"
    out = []
    for rd in runs:
        if "PDA" not in rd[tg_key]:
            continue
        X_cen, off = make_delay_design(rd[bp_key][ci_cen], n_delays)
        X_dmn, _   = make_delay_design(rd[bp_key][ci_dmn], n_delays)
        if X_cen.shape[0] == 0:
            continue
        X_cen = (X_cen - X_cen.mean(0)) / (X_cen.std(0) + 1e-12)
        X_dmn = (X_dmn - X_dmn.mean(0)) / (X_dmn.std(0) + 1e-12)
        nv = X_cen.shape[0]
        out.append((X_cen, X_dmn, zscore(rd[tg_key]["PDA"][off:off + nv])))
    return out


def eval_subject(designs, group_scaler, group_model, alphas):
    """For one subject's per-run designs, return rows over CAL_SIZES x methods."""
    rows = []
    n_runs = len(designs)
    for n_cal in CAL_SIZES:
        if n_cal >= n_runs:
            continue
        test = designs[n_cal:]
        Xte = np.vstack([d[0] for d in test]); yte = np.concatenate([d[1] for d in test])
        gp = group_model.predict(group_scaler.transform(Xte))
        r_grp = pearsonr(yte, gp)[0] if np.std(gp) > 1e-9 else np.nan
        if n_cal == 0:
            rows.append(dict(n_cal_runs=0, method="group_only", r=r_grp, n_test_runs=n_runs))
            continue
        cal = designs[:n_cal]
        Xcal = np.vstack([d[0] for d in cal]); pcal = np.concatenate([d[2] for d in cal])
        rows.append(dict(n_cal_runs=n_cal, method="group_only", r=r_grp, n_test_runs=n_runs - n_cal))
        if np.std(pcal) < 1e-9:
            continue
        sc = StandardScaler().fit(Xcal)
        cal_model = RidgeCV(alphas=alphas).fit(sc.transform(Xcal), pcal)
        pc = cal_model.predict(sc.transform(Xte))
        r_ps = pearsonr(yte, pc)[0] if np.std(pc) > 1e-9 else np.nan
        rows.append(dict(n_cal_runs=n_cal, method="pseudo_cal", r=r_ps, n_test_runs=n_runs - n_cal))
        if np.isfinite(r_ps) and np.std(gp) > 1e-9 and np.std(pc) > 1e-9:
            bl = zscore(gp) + zscore(pc)
            r_bl = pearsonr(yte, bl)[0]
            rows.append(dict(n_cal_runs=n_cal, method="blend", r=r_bl, n_test_runs=n_runs - n_cal))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["loso", "cross"], required=True)
    ap.add_argument("--dmnelf-config", default=None)
    ap.add_argument("--rtbpd-config", default=None)
    ap.add_argument("--dmnelf-cache", default=str(PROJ / "results" / "features_cache"))
    ap.add_argument("--rtbpd-cache", default=None)
    ap.add_argument("--res", default="tr")
    ap.add_argument("--tag", default="")
    ap.add_argument("--frontal", action="store_true",
                    help="Restrict electrode pool to frontal channels (portable headset test)")
    ap.add_argument("--frontal-multi", action="store_true",
                    help="Use all frontal electrodes concatenated (multivariate frontal EFP)")
    ap.add_argument("--epoc12-multi", action="store_true",
                    help="Use the real mindwear-deployed EPOC12 montage, concatenated "
                         "(multivariate EFP matching export_model.py/train_pooled.py exactly)")
    args = ap.parse_args()
    elec_restrict = FRONTAL_ELECTRODES if args.frontal else None
    frontal_suffix = "_frontal" if args.frontal else ""
    frontal_multi = args.frontal_multi or args.epoc12_multi
    if frontal_multi:
        if args.epoc12_multi:
            frontal_suffix = "_epoc12_multi"
            frontal_elecs = EPOC12_ELECTRODES               # deployed montage order
        else:
            frontal_suffix = "_frontal_multi"
            frontal_elecs = sorted(FRONTAL_ELECTRODES)       # sorted for consistent column order

    dcfg = load_config(args.dmnelf_config) if args.dmnelf_config else load_config()
    res = args.res
    alphas = alphas_from(dcfg)
    n_delays = n_delays_for(dcfg, res)
    tr = dcfg["data"]["fmri"]["tr"]
    baseline_trs = int(round(30.0 / tr))                       # ~30 s rest
    h = dcfg["hrf"]
    hrf = canonical_hrf(tr, h["length_s"], h["delay"], h["undershoot"])
    dmnelf_subs = dcfg["data"]["subjects"]["all"]
    er = pd.read_csv(RES / "electrode_r_all.csv")

    rows = []
    if args.mode == "loso":
        for held in dmnelf_subs:
            train_subs = [s for s in dmnelf_subs if s != held]
            try:
                runs, chs = load_subject_features(Path(args.dmnelf_cache), held)
            except Exception:
                print(f"{held}: no features, skip"); continue
            target_models = {}  # cache for proxy PDA reuse
            for target in TARGETS:
                if frontal_multi:
                    cis = [chs.index(el) for el in frontal_elecs if el in chs]
                    if len(cis) != len(frontal_elecs):
                        continue
                    Xtr, ytr = stack_design_multi(args.dmnelf_cache, train_subs,
                                                  frontal_elecs, target, res, n_delays)
                    if Xtr is None:
                        continue
                    gs = StandardScaler().fit(Xtr)
                    gm = RidgeCV(alphas=alphas).fit(gs.transform(Xtr), ytr)
                    el_label = "+".join(frontal_elecs)
                    des = run_designs_multi(runs, cis, target, res, n_delays,
                                           baseline_trs, hrf, PSEUDO_SIGN[target])
                else:
                    el = group_peak(er, target, res, exclude=held, restrict=elec_restrict)
                    if el is None or el not in chs:
                        continue
                    Xtr, ytr, used = stack_design(args.dmnelf_cache, train_subs, el, target,
                                                  res, "EFP", n_delays, hrf)
                    if Xtr is None:
                        continue
                    gs = StandardScaler().fit(Xtr)
                    gm = RidgeCV(alphas=alphas).fit(gs.transform(Xtr), ytr)
                    target_models[target] = (el, gs, gm)
                    ci = chs.index(el)
                    el_label = el
                    des = run_designs(runs, ci, target, res, n_delays, baseline_trs,
                                      hrf, PSEUDO_SIGN[target])
                if len(des) < 2:
                    continue
                for row in eval_subject(des, gs, gm, alphas):
                    rows.append(dict(subject=held, target=target, electrode=el_label, **row))
            # proxy PDA: combine frontal CEN and DMN predictions (single-electrode frontal only)
            if elec_restrict is not None and not frontal_multi \
                    and "CEN" in target_models and "DMN" in target_models:
                el_c, gs_c, gm_c = target_models["CEN"]
                el_d, gs_d, gm_d = target_models["DMN"]
                ci_c, ci_d = chs.index(el_c), chs.index(el_d)
                pdes = proxy_pda_run_designs(runs, ci_c, ci_d, res, n_delays)
                if pdes:
                    Xte_c = np.vstack([d[0] for d in pdes])
                    Xte_d = np.vstack([d[1] for d in pdes])
                    yte_p = np.concatenate([d[2] for d in pdes])
                    p_c = gm_c.predict(gs_c.transform(Xte_c))
                    p_d = gm_d.predict(gs_d.transform(Xte_d))
                    proxy = zscore(p_c) - zscore(p_d)
                    r = pearsonr(yte_p, proxy)[0] if np.std(proxy) > 1e-9 else np.nan
                    rows.append(dict(subject=held, target="proxy_PDA",
                                     electrode=f"{el_c}-{el_d}", n_cal_runs=0,
                                     method="group_only", r=r, n_test_runs=len(pdes)))
            print(f"{held}: done")
        out = OUT / f"efp_calibrated_loso{frontal_suffix}.csv"
    else:
        rcfg = load_config(args.rtbpd_config)
        rt_subs = rcfg["data"]["subjects"]["all"]
        rt_cache = args.rtbpd_cache or str(PROJ / "results" / f"features_cache_rtbpd{args.tag}")
        # general models trained once per target on ALL DMNELF
        gmods = {}
        if frontal_multi:
            el_label = "+".join(frontal_elecs)
            for target in TARGETS:
                Xtr, ytr = stack_design_multi(args.dmnelf_cache, dmnelf_subs,
                                              frontal_elecs, target, res, n_delays)
                if Xtr is None:
                    continue
                gs = StandardScaler().fit(Xtr)
                gm = RidgeCV(alphas=alphas).fit(gs.transform(Xtr), ytr)
                gmods[target] = (el_label, gs, gm)
        else:
            elmap = electrode_map(str(RES / "efp_group_loso.csv"), res)   # all-DMNELF peak
            if elec_restrict is not None:
                elmap = {t: e for t, e in elmap.items() if e in elec_restrict}
            for target in TARGETS:
                el = elmap.get(target)
                if el is None:
                    continue
                Xtr, ytr, used = stack_design(args.dmnelf_cache, dmnelf_subs, el, target,
                                              res, "EFP", n_delays, hrf)
                if Xtr is None:
                    continue
                gs = StandardScaler().fit(Xtr)
                gm = RidgeCV(alphas=alphas).fit(gs.transform(Xtr), ytr)
                gmods[target] = (el, gs, gm)
        for sub in rt_subs:
            try:
                runs, chs = load_subject_features(Path(rt_cache), sub)
            except Exception:
                print(f"{sub}: no features, skip"); continue
            for target, (el, gs, gm) in gmods.items():
                if frontal_multi:
                    cis = [chs.index(e) for e in frontal_elecs if e in chs]
                    if len(cis) != len(frontal_elecs):
                        continue
                    des = run_designs_multi(runs, cis, target, res, n_delays,
                                           baseline_trs, hrf, PSEUDO_SIGN[target])
                else:
                    if el not in chs:
                        continue
                    ci = chs.index(el)
                    des = run_designs(runs, ci, target, res, n_delays, baseline_trs,
                                     hrf, PSEUDO_SIGN[target])
                if len(des) < 2:
                    continue
                for row in eval_subject(des, gs, gm, alphas):
                    rows.append(dict(subject=sub, target=target, electrode=el, **row))
            print(f"{sub}: done")
        out = OUT / f"efp_calibrated_crosscohort_{res}{frontal_suffix}{args.tag}.csv"

    df = pd.DataFrame(rows)
    df.to_csv(out, index=False)
    print(f"\nSaved {out}  ({len(df)} rows)")

    # summary: mean r per (target, n_cal, method) + gain vs group_only + sign-flip on gain
    summ = []
    for target in TARGETS:
        for n_cal in CAL_SIZES:
            sub_g = df[(df.target == target) & (df.n_cal_runs == n_cal) & (df.method == "group_only")]
            g_by_sub = sub_g.set_index("subject")["r"]
            for method in ["group_only", "pseudo_cal", "blend"]:
                d = df[(df.target == target) & (df.n_cal_runs == n_cal) & (df.method == method)]
                if not len(d):
                    continue
                mean_r = float(d.r.mean())
                gain_p = np.nan
                if method != "group_only" and len(g_by_sub):
                    gains = [d.set_index("subject")["r"].get(s, np.nan) - g_by_sub.get(s, np.nan)
                             for s in d.subject]
                    gains = [x for x in gains if np.isfinite(x)]
                    if len(gains) >= 3:
                        gain_p = sign_flip_p(gains)
                summ.append(dict(target=target, n_cal_runs=n_cal, method=method,
                                 n=len(d), mean_r=mean_r, gain_p=gain_p))
    sdf = pd.DataFrame(summ)
    sout = str(out).replace(".csv", "_summary.csv")
    sdf.to_csv(sout, index=False)
    print("Saved", sout)
    # quick view: n_cal=1
    print("\n=== n_cal=1 (mean r; gain_p = sign-flip that method beats group_only) ===")
    for target in TARGETS:
        d = sdf[(sdf.target == target) & (sdf.n_cal_runs == 1)]
        if not len(d):
            continue
        parts = []
        for _, r in d.iterrows():
            gp = "" if not np.isfinite(r.gain_p) else f" p={r.gain_p:.3f}"
            parts.append(f"{r.method}={r.mean_r:+.3f}{gp}")
        print(f"  {target:8s} " + "  ".join(parts))
    # proxy PDA summary (frontal mode only)
    proxy_rows = df[df.target == "proxy_PDA"]
    if len(proxy_rows):
        el_label = proxy_rows.electrode.iloc[0]
        mean_r = float(proxy_rows.r.mean())
        print(f"\n=== proxy_PDA ({el_label}): mean r = {mean_r:+.3f}  (n={len(proxy_rows)}) ===")


if __name__ == "__main__":
    main()
