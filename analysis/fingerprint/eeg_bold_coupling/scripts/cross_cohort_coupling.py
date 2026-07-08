#!/usr/bin/env python
"""
cross_cohort_coupling.py
------------------------
External validation (double replication) of the multivariate EEG band-power
fingerprint: train the DMNELF *general* group model on ALL DMNELF subjects
(CAR band-power [n_ch x n_bands] -> target), then predict each rtBPD subject.
Run once per rtBPD session (nf1, nf2) for a double replication — the band-power
analogue of efp_meirhasson/cross_cohort_efp.py.

Design decisions (matched to the within-DMNELF decoder and to the EFP cross-cohort):
- Features: per-channel HRF-convolved log band power, CAR within band, per-run
  z-scored (identical to car_and_flatten in multivariate_decode_pda.py).
- Channels are intersected to the COMMON set present in both cohorts and reordered
  to a single canonical order, so the group weight vector aligns across cohorts.
- Targets: PDA, RAW_DMN, RAW_CEN, GSR_DMN, GSR_CEN, GSR_PDA (GSR = residualize on
  fMRIPrep global_signal). z-scored per run.
- Models: ridge and elasticnet (alpha as in the within-subject decoder).
- Significance: one-sample sign-flip permutation that group-mean r > 0.

The DMNELF band-power cache (results/multivariate/cache/{sub}_bandpower.npz) is
reused; rtBPD features are extracted+cached on first run.

Usage (cluster):
  python cross_cohort_coupling.py --rtbpd-config ../config_rtbpd.yaml --tag ""      # nf1
  python cross_cohort_coupling.py --rtbpd-config ../config_rtbpd_nf2.yaml --tag _nf2
"""
import argparse, time, warnings
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import pearsonr
from sklearn.preprocessing import StandardScaler

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from bandpower import load_config, canonical_hrf, gather_subject, zscore
from multivariate_decode_pda import residualize, make_model

warnings.filterwarnings("ignore")
PROJ = Path(__file__).resolve().parent.parent
CACHE = PROJ / "results" / "multivariate" / "cache"          # DMNELF cache (reuse)
RES = PROJ / "results"
TARGETS = ["PDA", "RAW_DMN", "RAW_CEN", "GSR_DMN", "GSR_CEN", "GSR_PDA"]
MODELS = ["ridge", "elasticnet"]


def load_confounds(cfg, sub, run):
    """global_signal for one run, using session_fmri (rtBPD) or session (DMNELF)."""
    d = cfg["data"]; ses = d.get("session_fmri", d["session"])
    tsv = (Path(d["confounds_dir"]) / f"sub-{sub}" / ses / "func" /
           f"sub-{sub}_{ses}_task-{d['task']}_run-{int(run):02d}_desc-confounds_timeseries.tsv")
    df = pd.read_csv(tsv, sep="\t")
    gs = df["global_signal"].values.astype(float)
    gs[0] = gs[1]
    return gs


def get_runs(cfg, sub, hrf, cache_dir):
    """Load (or extract+cache) band-power runs for one subject."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    cf = cache_dir / f"{sub}_bandpower.npz"
    if cf.exists():
        z = np.load(cf, allow_pickle=True)
        return list(z["runs_data"]), list(z["ch_names"])
    runs = gather_subject(cfg, sub, hrf)
    if not runs:
        return None, None
    chs = runs[0]["chs"]
    np.savez_compressed(cf, runs_data=np.array(runs, dtype=object), ch_names=chs)
    return runs, chs


def target_vec(runs, confounds, tname):
    pieces = []
    for rd, gs in zip(runs, confounds):
        t = rd["targets"]
        if tname == "PDA":       y = t["PDA"].copy()
        elif tname == "RAW_DMN": y = t["DMN"].copy()
        elif tname == "RAW_CEN": y = t["CEN"].copy()
        elif tname == "GSR_DMN": y = residualize(t["DMN"].copy(), gs)
        elif tname == "GSR_CEN": y = residualize(t["CEN"].copy(), gs)
        elif tname == "GSR_PDA": y = residualize(t["CEN"].copy(), gs) - residualize(t["DMN"].copy(), gs)
        else: raise ValueError(tname)
        pieces.append(zscore(y))
    return np.concatenate(pieces)


def design(runs, chs, common, band_names):
    """CAR (over common chs) + per-run z-score, columns aligned to `common` order."""
    idx = [chs.index(c) for c in common]
    pieces = []
    for rd in runs:
        bl = []
        for b in band_names:
            bp = rd["bp"][b][:, idx].copy()          # reorder to common channels
            bp -= bp.mean(axis=1, keepdims=True)     # CAR over common set
            bl.append(bp)
        pieces.append(zscore(np.concatenate(bl, axis=1)))
    return np.vstack(pieces)


def sign_flip_p(rs, n=10000, seed=0):
    rs = np.asarray([r for r in rs if np.isfinite(r)])
    if len(rs) < 2:
        return np.nan
    obs = rs.mean(); rng = np.random.default_rng(seed)
    null = (rng.choice([-1, 1], size=(n, len(rs))) * np.abs(rs)).mean(1)
    return float((np.sum(null >= obs) + 1) / (n + 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dmnelf-config", default=str(PROJ / "config.yaml"))
    ap.add_argument("--rtbpd-config", required=True)
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--tag", default="", help="output suffix, e.g. _nf2")
    args = ap.parse_args()

    dcfg = load_config(args.dmnelf_config)
    rcfg = load_config(args.rtbpd_config)
    band_names = list(dcfg["bands"].keys())
    hrf = canonical_hrf(tr=dcfg["data"]["fmri"]["tr"], length_s=dcfg["hrf"]["length_s"])

    # ---- load both cohorts ----
    t0 = time.time()
    dmnelf = {}
    for sub in dcfg["data"]["subjects"]["all"]:
        runs, chs = get_runs(dcfg, sub, hrf, CACHE)
        if runs is None:
            print(f"  DMNELF {sub}: no runs, skip"); continue
        conf = [load_confounds(dcfg, sub, rd["run"])[:rd["n_tr"]] for rd in runs]
        dmnelf[sub] = (runs, chs, conf)
    rt_cache = RES / f"multivariate_cache_rtbpd{args.tag}"
    rtbpd = {}
    for sub in rcfg["data"]["subjects"]["all"]:
        try:
            runs, chs = get_runs(rcfg, sub, hrf, rt_cache)
        except Exception as e:
            print(f"  rtBPD {sub}: extract failed ({e}), skip"); continue
        if runs is None:
            print(f"  rtBPD {sub}: no runs, skip"); continue
        try:
            conf = [load_confounds(rcfg, sub, rd["run"])[:rd["n_tr"]] for rd in runs]
        except Exception as e:
            print(f"  rtBPD {sub}: no confounds ({e}), skip"); continue
        rtbpd[sub] = (runs, chs, conf)
    print(f"Loaded {len(dmnelf)} DMNELF + {len(rtbpd)} rtBPD subjects in {time.time()-t0:.0f}s")

    # ---- common channels across BOTH cohorts (canonical order = DMNELF sub-1 order) ----
    all_chs = [c for _, chs, _ in dmnelf.values() for c in [chs]]
    ref_order = list(dmnelf[next(iter(dmnelf))][1])
    common = set(ref_order)
    for _, chs, _ in list(dmnelf.values()) + list(rtbpd.values()):
        common &= set(chs)
    common = [c for c in ref_order if c in common]
    print(f"Common channels: {len(common)}  (features = {len(common)*len(band_names)})")

    rows, summ = [], []
    for tname in TARGETS:
        # training design across all DMNELF
        Xtr = np.vstack([design(runs, chs, common, band_names)
                         for runs, chs, _ in dmnelf.values()])
        ytr = np.concatenate([target_vec(runs, conf, tname)
                              for runs, _, conf in dmnelf.values()])
        for mname in MODELS:
            scaler = StandardScaler().fit(Xtr)
            model = make_model(mname, args.alpha).fit(scaler.transform(Xtr), ytr)
            subj_rs = []
            for sub, (runs, chs, conf) in rtbpd.items():
                X = design(runs, chs, common, band_names)
                y = target_vec(runs, conf, tname)
                if len(y) < 10:
                    continue
                pred = model.predict(scaler.transform(X))
                r = pearsonr(y, pred)[0] if np.std(pred) > 1e-9 else np.nan
                subj_rs.append(r)
                rows.append(dict(cohort="rtBPD", subject=sub, target=tname,
                                 method=mname, n_trs=len(y), r=r))
            mean_r = float(np.nanmean(subj_rs)) if subj_rs else np.nan
            p = sign_flip_p(subj_rs)
            summ.append(dict(target=tname, method=mname, n_train=len(dmnelf),
                             n_test=len(subj_rs), n_features=Xtr.shape[1],
                             mean_r=mean_r, sign_flip_p=p))
            print(f"{tname:8s} {mname:11s} test={len(subj_rs):2d} mean_r={mean_r:+.3f} p={p:.3f}")

    out = RES
    pd.DataFrame(rows).to_csv(out / f"cross_cohort_coupling_persubject{args.tag}.csv", index=False)
    pd.DataFrame(summ).to_csv(out / f"cross_cohort_coupling_summary{args.tag}.csv", index=False)
    print(f"\nSaved {out}/cross_cohort_coupling_summary{args.tag}.csv")


if __name__ == "__main__":
    main()
