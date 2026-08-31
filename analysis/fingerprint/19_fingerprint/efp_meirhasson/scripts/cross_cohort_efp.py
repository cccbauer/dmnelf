#!/usr/bin/env python3
"""
cross_cohort_efp.py
-------------------
External validation of the EFP fingerprint: train the DMNELF *general* fingerprint
on all DMNELF subjects (single transfer electrode per target, band-index aligned —
same construction as the within-DMNELF LOSO), then predict each rtBPD subject.

Transfer electrode per target = the DMNELF LOSO modal best electrode
(results/full/efp_group_loso.csv). Reuses efp_features/efp_decode.

Usage (on cluster):
  python cross_cohort_efp.py \
    --rtbpd-config ../config_rtbpd.yaml \
    --dmnelf-cache results/features_cache \
    --rtbpd-cache  results/features_cache_rtbpd \
    --res tr
"""
import argparse, warnings
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import pearsonr, zscore
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeCV

from efp_features import load_config, load_subject_features
from efp_decode import assemble, assemble_hrf, assemble_ta, canonical_hrf

warnings.filterwarnings("ignore")
PROJ = Path(__file__).resolve().parent.parent
RES = PROJ / "results" / "full"


def n_delays_for(cfg, res):
    e = cfg["efp"]; tr = cfg["data"]["fmri"]["tr"]
    return (int(round(e["delay_window_s"] / tr)) + 1 if res == "tr"
            else int(round(e["delay_window_s"] * e["hz4"])) + 1)


def alphas_from(cfg):
    e = cfg["efp"]
    return np.logspace(np.log10(e["alpha_grid_lo"]), np.log10(e["alpha_grid_hi"]), e["alpha_grid_n"])


def electrode_map(loso_csv, res):
    df = pd.read_csv(loso_csv)
    d = df[df.resolution == res]
    return {row["target"]: row["common_ch"] for _, row in d.iterrows()}


def build_design(runs, chs, el, target, res, method, n_delays, hrf):
    """Design for one subject at electrode `el` for the requested method.

    All designs are per-subject z-scored so features are scale-invariant across
    subjects/cohorts (HRF/TA are already per-run z-scored in their assemblers; EFP is
    z-scored here to match — otherwise cross-cohort scale mismatch penalizes EFP)."""
    if el not in chs:
        return None, None
    ci = chs.index(el)
    if method == "EFP":
        return assemble(runs, ci, target, res, n_delays)   # assemble now per-run z-scores
    if method == "HRF":
        return assemble_hrf(runs, ci, target, res, hrf)
    if method == "TA":
        return assemble_ta(runs, ci, target, res, hrf)
    return None, None


def stack_design(cache, subs, electrode, target, res, method, n_delays, hrf):
    """Concatenate one method's designs at a fixed electrode across subjects."""
    Xs, ys, used = [], [], []
    for sub in subs:
        try:
            runs, chs = load_subject_features(Path(cache), sub)
        except Exception:
            continue
        X, y = build_design(runs, chs, electrode, target, res, method, n_delays, hrf)
        if X is None or len(y) < 10:
            continue
        Xs.append(X); ys.append(y); used.append(sub)
    if not Xs:
        return None, None, used
    return np.vstack(Xs), np.concatenate(ys), used


def sign_flip_p(rs, n_flips=10000, seed=0):
    """One-sample sign-flip test that mean r > 0 across subjects."""
    rs = np.asarray([r for r in rs if np.isfinite(r)])
    if len(rs) < 2:
        return np.nan
    obs = rs.mean()
    rng = np.random.default_rng(seed)
    signs = rng.choice([-1, 1], size=(n_flips, len(rs)))
    null = (signs * np.abs(rs)).mean(axis=1)
    return float((np.sum(null >= obs) + 1) / (n_flips + 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dmnelf-config", default=None)
    ap.add_argument("--rtbpd-config", required=True)
    ap.add_argument("--dmnelf-cache", default=str(PROJ / "results" / "features_cache"))
    ap.add_argument("--rtbpd-cache", default=str(PROJ / "results" / "features_cache_rtbpd"))
    ap.add_argument("--loso", default=str(RES / "efp_group_loso.csv"))
    ap.add_argument("--res", default="tr")
    ap.add_argument("--tag", default="", help="suffix for output CSVs, e.g. _nf2")
    args = ap.parse_args()

    dcfg = load_config(args.dmnelf_config) if args.dmnelf_config else load_config()
    rcfg = load_config(args.rtbpd_config)
    res = args.res
    alphas = alphas_from(dcfg)
    n_delays = n_delays_for(dcfg, res)
    electrodes = electrode_map(args.loso, res)

    dmnelf_subs = dcfg["data"]["subjects"]["all"]
    rtbpd_subs = rcfg["data"]["subjects"]["all"]
    targets = dcfg["targets"]
    hrf = canonical_hrf(dcfg["data"]["fmri"]["tr"], dcfg["hrf"]["length_s"],
                        dcfg["hrf"]["delay"], dcfg["hrf"]["undershoot"])
    # cache rtBPD features once
    rt = {}
    for sub in rtbpd_subs:
        try:
            rt[sub] = load_subject_features(Path(args.rtbpd_cache), sub)
        except Exception:
            continue

    rows, summ = [], []
    for target in targets:
        el = electrodes.get(target)
        if el is None:
            print(f"{target}: no transfer electrode, skip"); continue
        for method in ("EFP", "HRF", "TA"):
            Xtr, ytr, tr_used = stack_design(args.dmnelf_cache, dmnelf_subs, el, target,
                                             res, method, n_delays, hrf)
            if Xtr is None:
                continue
            scaler = StandardScaler().fit(Xtr)
            model = RidgeCV(alphas=alphas).fit(scaler.transform(Xtr), ytr)
            subj_rs = []
            for sub, (runs, chs) in rt.items():
                X, y = build_design(runs, chs, el, target, res, method, n_delays, hrf)
                if X is None or len(y) < 10:
                    continue
                pred = model.predict(scaler.transform(X))
                r = pearsonr(y, pred)[0] if np.std(pred) > 1e-9 else np.nan
                subj_rs.append(r)
                rows.append(dict(cohort="rtBPD", subject=sub, target=target, method=method,
                                 electrode=el, n_trs=len(y), r=r))
            mean_r = float(np.nanmean(subj_rs)) if subj_rs else np.nan
            p = sign_flip_p(subj_rs)
            summ.append(dict(target=target, resolution=res, method=method, electrode=el,
                             n_train=len(tr_used), n_test=len(subj_rs),
                             mean_r=mean_r, sign_flip_p=p))
            print(f"{target:8s} {method:3s} el={el:4s} test={len(subj_rs):2d} "
                  f"mean_r={mean_r:+.3f} p={p:.3f}")

    out = PROJ / "results"
    pd.DataFrame(rows).to_csv(out / f"cross_cohort_efp_persubject_{res}{args.tag}.csv", index=False)
    pd.DataFrame(summ).to_csv(out / f"cross_cohort_efp_summary_{res}{args.tag}.csv", index=False)
    print(f"\nSaved {out}/cross_cohort_efp_summary_{res}{args.tag}.csv")


if __name__ == "__main__":
    main()
