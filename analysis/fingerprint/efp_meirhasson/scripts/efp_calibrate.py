#!/usr/bin/env python3
"""
efp_calibrate.py  (cluster)  —  few-shot calibration + joint CEN/DMN decoding
-----------------------------------------------------------------------------
Idea 1 (calibration): rescue the weak cross-cohort transfer by calibrating on ONE rtBPD run.
  schemes per rtBPD subject (single best electrode, DMNELF-selected; clean targets, feedback):
    transfer     : DMNELF-fit weights, no rtBPD data (0-shot)         -> all runs
    cal1         : fit on subject run 1 only                          -> test runs 2+
    dmnelf+cal1  : fit on DMNELF-concat + subject run 1               -> test runs 2+
    within_loro  : leave-one-run-out within subject (upper bound)
Idea 2 (joint): decode CEN & DMN together (multi-output ridge) vs alone; + CCA system r.
  evaluated within-subject LORO (where signal is robust), all electrodes.

Output: efp_calibrate.csv (calibration ladder) + efp_joint.csv (joint vs single).
"""
import argparse, glob
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import zscore, pearsonr
from sklearn.linear_model import RidgeCV, Ridge
from sklearn.cross_decomposition import CCA
from efp_features import load_config, load_subject_features, make_delay_design
from efp_transfer import subjects, clean_target, CLEAN_DIR, CLEAN_PREFIX, TRAIN_CACHE, TEST

BASELINE_TR, HRF_DROP = 25, 5
ALPHAS = np.logspace(-2, 5, 15)


def per_run_designs(cache, sub, target, n_delays):
    """Return list over runs of (per-channel X [n,110], y) — feedback block, clean target."""
    runs, ch = load_subject_features(Path(cache), sub); nch = len(ch)
    cmf = Path(CLEAN_DIR) / f"{CLEAN_PREFIX[cache]}{sub}.npz"
    if not cmf.exists():
        return None, nch
    cm = np.load(cmf, allow_pickle=True); out = []
    for rd in runs:
        yv = clean_target(cm, rd["run"], target)
        if yv is None:
            continue
        Xs, off = [], None
        for ci in range(nch):
            X, off = make_delay_design(rd["bp_tr"][ci], n_delays)
            Xs.append((X - X.mean(0)) / (X.std(0) + 1e-12))
        nv = Xs[0].shape[0]; t = off + np.arange(nv)
        y = yv[off:off + nv]; m = (t >= BASELINE_TR + HRF_DROP) & np.isfinite(y)
        if m.sum() < 20:
            continue
        out.append(([X[m] for X in Xs], zscore(y[m])))
    return out, nch


def r_(y, p):
    return pearsonr(y, p)[0] if np.std(p) > 1e-9 and np.std(y) > 1e-9 else np.nan


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True)
    a = ap.parse_args(); out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    cfg = load_config(); n_delays = int(round(cfg["efp"]["delay_window_s"] / cfg["data"]["fmri"]["tr"])) + 1
    tr_subs = subjects(TRAIN_CACHE)

    cal_rows, joint_rows = [], []
    for target in ["CEN", "DMN"]:
        # DMNELF prior: per-subject concatenated designs; select best electrode by LOSO; fit weights
        Dtr = {s: per_run_designs(TRAIN_CACHE, s, target, n_delays) for s in tr_subs}
        Dtr = {s: v[0] for s, v in Dtr.items() if v[0]}
        nch = load_subject_features(Path(TRAIN_CACHE), next(iter(Dtr)))[0][0]["bp_tr"].shape[0]
        Dcat = {s: ([np.vstack([r[0][ci] for r in runs]) for ci in range(nch)],
                    np.concatenate([r[1] for r in runs])) for s, runs in Dtr.items()}
        best_ci, best = None, -np.inf
        for ci in range(nch):
            rs = [r_(Dcat[h][1], RidgeCV(alphas=ALPHAS).fit(
                np.vstack([Dcat[s][0][ci] for s in Dcat if s != h]),
                np.concatenate([Dcat[s][1] for s in Dcat if s != h])).predict(Dcat[h][0][ci]))
                for h in Dcat]
            mr = np.nanmean(rs)
            if mr > best:
                best, best_ci = mr, ci
        Xtr_all = np.vstack([Dcat[s][0][best_ci] for s in Dcat]); ytr_all = np.concatenate([Dcat[s][1] for s in Dcat])
        m_dmnelf = RidgeCV(alphas=ALPHAS).fit(Xtr_all, ytr_all)

        # --- calibration ladder on rtBPD ---
        for sess, cache in TEST.items():
            for s in subjects(cache):
                runs, _ = per_run_designs(cache, s, target, n_delays)
                if not runs or len(runs) < 2:
                    continue
                Xr = [r[0][best_ci] for r in runs]; yr = [r[1] for r in runs]
                rest_X = np.vstack(Xr[1:]); rest_y = np.concatenate(yr[1:])
                # transfer (0-shot)
                p_t = m_dmnelf.predict(np.vstack(Xr)); y_all = np.concatenate(yr)
                # cal1
                m_c = RidgeCV(alphas=ALPHAS).fit(Xr[0], yr[0]); p_c = m_c.predict(rest_X)
                # dmnelf + cal1
                m_dc = RidgeCV(alphas=ALPHAS).fit(np.vstack([Xtr_all, Xr[0]]),
                                                  np.concatenate([ytr_all, yr[0]])); p_dc = m_dc.predict(rest_X)
                # within LORO
                oof = []
                for i in range(len(Xr)):
                    tr = [j for j in range(len(Xr)) if j != i]
                    mm = RidgeCV(alphas=ALPHAS).fit(np.vstack([Xr[j] for j in tr]), np.concatenate([yr[j] for j in tr]))
                    oof.append((yr[i], mm.predict(Xr[i])))
                yl = np.concatenate([o[0] for o in oof]); pl = np.concatenate([o[1] for o in oof])
                for scheme, y, p in [("transfer", y_all, p_t), ("cal1", rest_y, p_c),
                                     ("dmnelf+cal1", rest_y, p_dc), ("within_loro", yl, pl)]:
                    cal_rows.append(dict(session=sess, target=target, subject=s, scheme=scheme, r=r_(y, p)))
        print(f"{target}: best electrode idx={best_ci}", flush=True)
    pd.DataFrame(cal_rows).to_csv(out / "efp_calibrate.csv", index=False)

    # --- Idea 2: joint CEN+DMN (within-subject LORO, all electrodes) ---
    for sess, cache in [("dmnelf", TRAIN_CACHE)] + list(TEST.items()):
        for s in subjects(cache):
            rc, nch = per_run_designs(cache, s, "CEN", n_delays)
            rd_, _ = per_run_designs(cache, s, "DMN", n_delays)
            if not rc or not rd_ or len(rc) < 2:
                continue
            Xr = [np.column_stack(r[0]) for r in rc]           # all electrodes
            Yc = [r[1] for r in rc]; Yd = [r[1] for r in rd_]
            oc = oc_j = od = od_j = []; ccas = []
            preds = {k: ([], []) for k in ["cen_solo", "cen_joint", "dmn_solo", "dmn_joint"]}
            for i in range(len(Xr)):
                tr = [j for j in range(len(Xr)) if j != i]
                Xt = np.vstack([Xr[j] for j in tr]); Xe = Xr[i]
                yc = np.concatenate([Yc[j] for j in tr]); yd = np.concatenate([Yd[j] for j in tr])
                preds["cen_solo"][0].append(Yc[i]); preds["cen_solo"][1].append(Ridge(alpha=1e3).fit(Xt, yc).predict(Xe))
                preds["dmn_solo"][0].append(Yd[i]); preds["dmn_solo"][1].append(Ridge(alpha=1e3).fit(Xt, yd).predict(Xe))
                mj = Ridge(alpha=1e3).fit(Xt, np.column_stack([yc, yd])); pj = mj.predict(Xe)
                preds["cen_joint"][0].append(Yc[i]); preds["cen_joint"][1].append(pj[:, 0])
                preds["dmn_joint"][0].append(Yd[i]); preds["dmn_joint"][1].append(pj[:, 1])
                try:
                    cc = CCA(n_components=1).fit(Xt, np.column_stack([yc, yd]))
                    u, v = cc.transform(Xe, np.column_stack([Yc[i], Yd[i]])); ccas.append(r_(u[:, 0], v[:, 0]))
                except Exception:
                    pass
            row = dict(session=sess, subject=s,
                       cen_solo=r_(np.concatenate(preds["cen_solo"][0]), np.concatenate(preds["cen_solo"][1])),
                       cen_joint=r_(np.concatenate(preds["cen_joint"][0]), np.concatenate(preds["cen_joint"][1])),
                       dmn_solo=r_(np.concatenate(preds["dmn_solo"][0]), np.concatenate(preds["dmn_solo"][1])),
                       dmn_joint=r_(np.concatenate(preds["dmn_joint"][0]), np.concatenate(preds["dmn_joint"][1])),
                       cca_system=np.nanmean(ccas) if ccas else np.nan)
            joint_rows.append(row)
    pd.DataFrame(joint_rows).to_csv(out / "efp_joint.csv", index=False)
    print("saved efp_calibrate.csv + efp_joint.csv", flush=True)


if __name__ == "__main__":
    main()
