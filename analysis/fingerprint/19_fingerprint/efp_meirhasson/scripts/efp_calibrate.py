#!/usr/bin/env python3
"""
efp_calibrate.py  (cluster)  —  FAIR few-shot calibration + reduced-rank joint CEN/DMN
--------------------------------------------------------------------------------------
Idea 1 (calibration, MULTIVARIATE all-electrode so it can reach the ~0.11 ceiling):
  per rtBPD subject, clean target, feedback block, all 31 electrodes' EFP designs:
    transfer     : DMNELF-fit multivariate weights (0-shot)            -> all runs
    cal1         : fit on subject run 1 only                           -> test runs 2+
    dmnelf+cal1  : DMNELF-concat + subject run 1                       -> test runs 2+
    within_loro  : leave-one-run-out within subject (upper bound)
Idea 2 (joint, REDUCED-RANK regression): within-subject LORO, all electrodes; predict
  [CEN,DMN] with a rank-1 shared EEG->network map -> per-network r vs solo (rank-2/full).

Output: efp_calibrate.csv (multivariate ladder) + efp_joint.csv (solo vs rank-1 RRR).
"""
import argparse
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import zscore, pearsonr
from sklearn.linear_model import RidgeCV, Ridge
from efp_features import load_config, load_subject_features, make_delay_design
from efp_transfer import subjects, clean_target, CLEAN_DIR, CLEAN_PREFIX, TRAIN_CACHE, TEST

BASELINE_TR, HRF_DROP = 25, 5
ALPHAS = np.logspace(0, 6, 12)


def per_run_mv(cache, sub, target, n_delays):
    """List over runs of (X_allelec [n, 31*110], y) — feedback block, clean target."""
    runs, ch = load_subject_features(Path(cache), sub); nch = len(ch)
    cmf = Path(CLEAN_DIR) / f"{CLEAN_PREFIX[cache]}{sub}.npz"
    if not cmf.exists():
        return None
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
        out.append((np.column_stack([X[m] for X in Xs]), zscore(y[m])))
    return out if out else None


def r_(y, p):
    return pearsonr(y, p)[0] if np.std(p) > 1e-9 and np.std(y) > 1e-9 else np.nan


def rrr_predict(Xtr, Ytr, Xte, rank, alpha=1e3):
    """Reduced-rank ridge: project the fitted 2-output response onto its top-`rank` shared axis."""
    m = Ridge(alpha=alpha).fit(Xtr, Ytr)
    Ptr = m.predict(Xtr); pm = Ptr.mean(0)
    _, _, Vt = np.linalg.svd(Ptr - pm, full_matrices=False)
    Vr = Vt[:rank].T
    return (m.predict(Xte) - pm) @ Vr @ Vr.T + pm


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True)
    a = ap.parse_args(); out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    cfg = load_config(); n_delays = int(round(cfg["efp"]["delay_window_s"] / cfg["data"]["fmri"]["tr"])) + 1
    tr_subs = subjects(TRAIN_CACHE)

    # ---------- Idea 1: MULTIVARIATE calibration ladder ----------
    cal_rows = []
    for target in ["CEN", "DMN"]:
        D = {s: per_run_mv(TRAIN_CACHE, s, target, n_delays) for s in tr_subs}
        D = {s: v for s, v in D.items() if v}
        Xtr_all = np.vstack([np.vstack([r[0] for r in runs]) for runs in D.values()])
        ytr_all = np.concatenate([np.concatenate([r[1] for r in runs]) for runs in D.values()])
        m_dmnelf = RidgeCV(alphas=ALPHAS).fit(Xtr_all, ytr_all)
        for sess, cache in TEST.items():
            for s in subjects(cache):
                runs = per_run_mv(cache, s, target, n_delays)
                if not runs or len(runs) < 2:
                    continue
                Xr = [r[0] for r in runs]; yr = [r[1] for r in runs]
                restX = np.vstack(Xr[1:]); resty = np.concatenate(yr[1:])
                p_t = m_dmnelf.predict(np.vstack(Xr)); y_all = np.concatenate(yr)
                p_c = RidgeCV(alphas=ALPHAS).fit(Xr[0], yr[0]).predict(restX)
                p_dc = RidgeCV(alphas=ALPHAS).fit(np.vstack([Xtr_all, Xr[0]]),
                                                  np.concatenate([ytr_all, yr[0]])).predict(restX)
                oof = []
                for i in range(len(Xr)):
                    tr = [j for j in range(len(Xr)) if j != i]
                    mm = RidgeCV(alphas=ALPHAS).fit(np.vstack([Xr[j] for j in tr]),
                                                    np.concatenate([yr[j] for j in tr]))
                    oof.append((yr[i], mm.predict(Xr[i])))
                yl = np.concatenate([o[0] for o in oof]); pl = np.concatenate([o[1] for o in oof])
                for sch, y, p in [("transfer", y_all, p_t), ("cal1", resty, p_c),
                                  ("dmnelf+cal1", resty, p_dc), ("within_loro", yl, pl)]:
                    cal_rows.append(dict(session=sess, target=target, subject=s, scheme=sch, r=r_(y, p)))
        print(f"calibration {target} done", flush=True)
    pd.DataFrame(cal_rows).to_csv(out / "efp_calibrate.csv", index=False)

    # ---------- Idea 2: reduced-rank joint CEN+DMN ----------
    joint_rows = []
    for sess, cache in [("dmnelf", TRAIN_CACHE)] + list(TEST.items()):
        for s in subjects(cache):
            rc = per_run_mv(cache, s, "CEN", n_delays); rd_ = per_run_mv(cache, s, "DMN", n_delays)
            if not rc or not rd_ or len(rc) < 2:
                continue
            Xr = [r[0] for r in rc]; Yc = [r[1] for r in rc]; Yd = [r[1] for r in rd_]
            P = {k: ([], []) for k in ["cen_solo", "dmn_solo", "cen_rrr1", "dmn_rrr1"]}
            for i in range(len(Xr)):
                tr = [j for j in range(len(Xr)) if j != i]
                Xt = np.vstack([Xr[j] for j in tr]); Xe = Xr[i]
                yc = np.concatenate([Yc[j] for j in tr]); yd = np.concatenate([Yd[j] for j in tr])
                P["cen_solo"][0].append(Yc[i]); P["cen_solo"][1].append(Ridge(alpha=1e3).fit(Xt, yc).predict(Xe))
                P["dmn_solo"][0].append(Yd[i]); P["dmn_solo"][1].append(Ridge(alpha=1e3).fit(Xt, yd).predict(Xe))
                pr = rrr_predict(Xt, np.column_stack([yc, yd]), Xe, rank=1)
                P["cen_rrr1"][0].append(Yc[i]); P["cen_rrr1"][1].append(pr[:, 0])
                P["dmn_rrr1"][0].append(Yd[i]); P["dmn_rrr1"][1].append(pr[:, 1])
            joint_rows.append(dict(session=sess, subject=s,
                                   **{k: r_(np.concatenate(P[k][0]), np.concatenate(P[k][1])) for k in P}))
        print(f"joint {sess} done", flush=True)
    pd.DataFrame(joint_rows).to_csv(out / "efp_joint.csv", index=False)
    print("saved efp_calibrate.csv + efp_joint.csv", flush=True)


if __name__ == "__main__":
    main()
