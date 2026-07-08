#!/usr/bin/env python3
"""
efp_decode.py
-------------
Per-subject EEG Finger-Print (Meir-Hasson 2014) for DMN/CEN/PDA (+GSR).

For each target x resolution (TR, 4 Hz):
  - For each of 31 electrodes: sliding-delay ridge with DOUBLE cross-validation
    (outer m-k-fold contiguous block CV; inner RidgeCV for lambda). Metrics: NMSE + r.
  - Select best electrode = min mean CV NMSE; refit to get the [band x delay] EFP.
  - Baselines on the SAME folds:
      * T/A   : theta/alpha ratio (best occipital electrode), HRF-convolved, ridge(1 feat)
      * HRF   : all 10 bands HRF-convolved (fixed canonical delay) + ridge
    Expected ordering (paper): EFP >= HRF >= T/A.

Outputs (results/<outdir>/):
  efp_persubject.csv                 rows: subject,target,resolution,method,best_ch,mean_r,mean_nmse,...
  efp_<sub>_<target>_<res>.npz       best-electrode EFP matrix [n_bands x n_delays] + band_hz
"""
import argparse
from collections import Counter
from pathlib import Path
import numpy as np, pandas as pd, yaml
from scipy.stats import gamma, pearsonr, zscore
from sklearn.linear_model import Ridge, RidgeCV

from efp_features import (load_config, make_delay_design, load_subject_features,
                          build_subject_features)

SCRIPT_DIR = Path(__file__).resolve().parent
PROJ_DIR = SCRIPT_DIR.parent


# ── HRF (baselines only) ─────────────────────────────────────────────────────
def canonical_hrf(tr, length_s=32, delay=6, undershoot=16):
    t = np.arange(0, length_s, tr)
    h = gamma.pdf(t, delay) - gamma.pdf(t, undershoot) / 6.0
    return h / h.sum()


def hrf_convolve(x, hrf):
    return np.convolve(x, hrf, mode="full")[:len(x)]


def nmse(y_true, y_pred):
    v = np.var(y_true)
    return np.mean((y_true - y_pred) ** 2) / v if v > 0 else np.nan


# ── cross-validation ─────────────────────────────────────────────────────────
def mk_block_folds(n, k, m):
    """m repeats of contiguous k-fold; each repeat rolls the block boundaries."""
    folds = []
    base = np.arange(n)
    for rep in range(m):
        roll = (rep * n) // (m * k)
        idx = np.roll(base, roll)
        bnds = np.linspace(0, n, k + 1).astype(int)
        for i in range(k):
            test = idx[bnds[i]:bnds[i + 1]]
            train = np.setdiff1d(base, test, assume_unique=False)
            if len(test) > 3 and len(train) > 5:
                folds.append((train, test))
    return folds


def cv_score(X, y, alphas, folds):
    """Double CV: inner RidgeCV picks lambda on train; outer folds give r + NMSE."""
    r_list, nmse_list = [], []
    for tr, te in folds:
        mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-12
        Xtr = (X[tr] - mu) / sd; Xte = (X[te] - mu) / sd
        model = RidgeCV(alphas=alphas)
        model.fit(Xtr, y[tr])
        pred = model.predict(Xte)
        if np.std(pred) < 1e-9 or np.std(y[te]) < 1e-9:
            continue
        r_list.append(pearsonr(y[te], pred)[0])
        nmse_list.append(nmse(y[te], pred))
    if not r_list:
        return np.nan, np.nan
    return float(np.mean(r_list)), float(np.mean(nmse_list))


def oof_r(X, y, alphas, folds):
    """Honest CV r for a FIXED design: concatenate out-of-fold predictions, then correlate.
    Unlike mean-of-per-fold-r (cv_score), this is not inflated by small test blocks."""
    pred = np.full(len(y), np.nan)
    for tr, te in folds:
        mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-12
        m = RidgeCV(alphas=alphas).fit((X[tr] - mu) / sd, y[tr])
        pred[te] = m.predict((X[te] - mu) / sd)
    ok = ~np.isnan(pred)
    if ok.sum() < 3 or np.std(pred[ok]) < 1e-9:
        return np.nan
    return float(pearsonr(y[ok], pred[ok])[0])


def nested_cv_r(X_list, y, alphas, k, m, cand, inner_m=1):
    """Nested-CV electrode selection to REMOVE selection bias.

    Outer m×k contiguous block folds (shared across candidates). Within each outer
    training set, an inner CV (k folds, inner_m repeats) picks the candidate electrode
    with min NMSE; that electrode is refit on the full outer-training set and predicts
    the held-out outer-test fold. The reported r/NMSE come from the concatenated
    out-of-fold predictions — the electrode is NEVER scored on data used to select it.

    X_list : list of (n, p) design matrices (one per electrode; None allowed).
    cand   : iterable of candidate indices into X_list.
    Returns (r, nmse, chosen_indices_per_outer_fold).
    """
    n = len(y)
    oof = np.full(n, np.nan)
    chosen = []
    for tr, te in mk_block_folds(n, k, m):
        inner = mk_block_folds(len(tr), k, inner_m)
        if not inner:
            continue
        best_ci, best_nm = None, np.inf
        for ci in cand:
            X = X_list[ci]
            if X is None:
                continue
            nms = []
            for itr, ite in inner:
                a, b = tr[itr], tr[ite]
                mu, sd = X[a].mean(0), X[a].std(0) + 1e-12
                mdl = RidgeCV(alphas=alphas).fit((X[a] - mu) / sd, y[a])
                p = mdl.predict((X[b] - mu) / sd)
                if np.std(p) > 1e-9 and np.std(y[b]) > 1e-9:
                    nms.append(nmse(y[b], p))
            if nms and np.mean(nms) < best_nm:
                best_nm, best_ci = np.mean(nms), ci
        if best_ci is None:
            continue
        X = X_list[best_ci]
        mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-12
        mdl = RidgeCV(alphas=alphas).fit((X[tr] - mu) / sd, y[tr])
        oof[te] = mdl.predict((X[te] - mu) / sd)
        chosen.append(best_ci)
    ok = ~np.isnan(oof)
    if ok.sum() < 3 or np.std(oof[ok]) < 1e-9:
        return np.nan, np.nan, chosen
    return float(pearsonr(y[ok], oof[ok])[0]), float(nmse(y[ok], oof[ok])), chosen


# ── assemble design per channel/target/resolution ───────────────────────────
def assemble(runs, ci, target, res, n_delays):
    """Concatenate per-run sliding-delay designs for one channel/target/resolution."""
    bp_key = "bp_tr" if res == "tr" else "bp_hz4"
    tg_key = "tgt_tr" if res == "tr" else "tgt_hz4"
    Xs, ys = [], []
    for rd in runs:
        if target not in rd[tg_key]:
            return None, None
        X, off = make_delay_design(rd[bp_key][ci], n_delays)
        if X.shape[0] == 0:
            continue
        # per-run standardize features (scale-invariant across subjects/cohorts,
        # matching assemble_hrf / assemble_ta so all methods are on equal footing)
        X = (X - X.mean(0)) / (X.std(0) + 1e-12)
        y = zscore(rd[tg_key][target][off:off + X.shape[0]])
        Xs.append(X); ys.append(y)
    if not Xs:
        return None, None
    return np.vstack(Xs), np.concatenate(ys)


def assemble_hrf(runs, ci, target, res, hrf):
    """Fixed-HRF baseline design: HRF-convolved 10 bands (no sliding delay)."""
    bp_key = "bp_tr" if res == "tr" else "bp_hz4"
    tg_key = "tgt_tr" if res == "tr" else "tgt_hz4"
    Xs, ys = [], []
    for rd in runs:
        bp = rd[bp_key][ci]  # (n_bands, n_out)
        Xc = np.column_stack([hrf_convolve(bp[b], hrf) for b in range(bp.shape[0])])
        Xs.append(zscore(Xc, axis=0)); ys.append(zscore(rd[tg_key][target]))
    return np.vstack(Xs), np.concatenate(ys)


def assemble_ta(runs, ci, target, res, hrf):
    """T/A baseline: theta/alpha ratio, HRF-convolved (single regressor)."""
    ta_key = "ta_tr" if res == "tr" else "ta_hz4"
    tg_key = "tgt_tr" if res == "tr" else "tgt_hz4"
    Xs, ys = [], []
    for rd in runs:
        ta = rd[ta_key][ci]  # (2, n_out): theta, alpha
        ratio = ta[0] / (ta[1] + 1e-12)
        Xs.append(zscore(hrf_convolve(ratio, hrf))[:, None]); ys.append(zscore(rd[tg_key][target]))
    return np.vstack(Xs), np.concatenate(ys)


# ── per-subject driver ───────────────────────────────────────────────────────
def process_subject(cfg, sub, cache_dir, out_dir):
    e = cfg["efp"]; tr = cfg["data"]["fmri"]["tr"]
    alphas = np.logspace(np.log10(e["alpha_grid_lo"]), np.log10(e["alpha_grid_hi"]),
                         e["alpha_grid_n"])
    hrf = canonical_hrf(tr, cfg["hrf"]["length_s"], cfg["hrf"]["delay"], cfg["hrf"]["undershoot"])
    occ = set(cfg["baseline"]["occipital_channels"])

    if not (cache_dir / f"{sub}_efp.npz").exists():
        build_subject_features(cfg, sub, cache_dir)
    runs, ch_names = load_subject_features(cache_dir, sub)
    occ_idx = [i for i, c in enumerate(ch_names) if c in occ] or list(range(len(ch_names)))

    rows = []
    for res in e["resolutions"]:
        n_delays = int(round(e["delay_window_s"] / tr)) + 1 if res == "tr" \
            else int(round(e["delay_window_s"] * e["hz4"])) + 1
        n_out_total = sum((rd["n_tr"] if res == "tr" else rd["n_hz4"]) for rd in runs)
        folds = mk_block_folds(n_out_total - len(runs) * (n_delays - 1),
                               e["cv_outer_k"], e["cv_outer_m"])
        k, m = e["cv_outer_k"], e["cv_outer_m"]
        for target in cfg["targets"]:
            # ---- EFP: nested-CV electrode selection (unbiased) ----
            X_list, y = [], None
            for ci in range(len(ch_names)):
                X, yc = assemble(runs, ci, target, res, n_delays)
                X_list.append(X)
                if X is not None and y is None:
                    y = yc
            if y is None:
                continue
            cand = [ci for ci, X in enumerate(X_list) if X is not None]
            if not cand:
                continue
            r_efp, nm_efp, chosen = nested_cv_r(X_list, y, alphas, k, m, cand)
            if not chosen:
                continue
            best_ci = Counter(chosen).most_common(1)[0][0]   # modal selected electrode
            best_ch = ch_names[best_ci]

            # descriptive EFP coefficient matrix: refit modal electrode on all data
            Xb = X_list[best_ci]; mu, sd = Xb.mean(0), Xb.std(0) + 1e-12
            model = RidgeCV(alphas=alphas).fit((Xb - mu) / sd, y)
            n_bands = cfg["efp"]["n_bands"]
            efp = model.coef_.reshape(n_delays, n_bands).T  # [bands x delays]
            band_hz = runs[0]["band_hz"]
            np.savez_compressed(out_dir / f"efp_{sub}_{target}_{res}.npz",
                                efp=efp, band_hz=np.array(band_hz), best_ch=best_ch,
                                n_delays=n_delays, tr=tr, res=res)
            rows.append(dict(subject=sub, target=target, resolution=res, method="EFP",
                             best_ch=best_ch, mean_r=r_efp, mean_nmse=nm_efp))

            # ---- HRF baseline: nested-CV over all electrodes (fair vs EFP) ----
            Xh_list, yh = [], None
            for ci in range(len(ch_names)):
                try:
                    Xc, yc = assemble_hrf(runs, ci, target, res, hrf)
                except Exception:
                    Xc, yc = None, None
                Xh_list.append(Xc)
                if Xc is not None and yh is None:
                    yh = yc
            hrf_cand = [ci for ci, X in enumerate(Xh_list) if X is not None]
            if yh is not None and hrf_cand:
                rh, nmh, hrf_chosen = nested_cv_r(Xh_list, yh, alphas, k, m, hrf_cand)
                hrf_ch = ch_names[Counter(hrf_chosen).most_common(1)[0][0]] if hrf_chosen else best_ch
            else:
                rh, nmh, hrf_ch = np.nan, np.nan, best_ch
            rows.append(dict(subject=sub, target=target, resolution=res, method="HRF",
                             best_ch=hrf_ch, mean_r=rh, mean_nmse=nmh))

            # ---- T/A baseline: nested-CV over occipital electrodes ----
            Xta_list = [None] * len(ch_names)
            yta = None
            for ci in occ_idx:
                Xt, yt = assemble_ta(runs, ci, target, res, hrf)
                Xta_list[ci] = Xt
                if yta is None:
                    yta = yt
            ta_cand = [ci for ci in occ_idx if Xta_list[ci] is not None]
            if yta is not None and ta_cand:
                rt, nmt, ta_chosen = nested_cv_r(Xta_list, yta, alphas, k, m, ta_cand)
                ta_ch = ch_names[Counter(ta_chosen).most_common(1)[0][0]] if ta_chosen else None
            else:
                rt, nmt, ta_ch = np.nan, np.nan, None
            rows.append(dict(subject=sub, target=target, resolution=res, method="TA",
                             best_ch=ta_ch, mean_r=rt, mean_nmse=nmt))
            print(f"  {sub} {target} {res}: EFP r={r_efp:+.3f} (ch {best_ch}) | "
                  f"HRF r={rh:+.3f} | TA r={rt:+.3f}")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subjects", nargs="+", default=None)
    ap.add_argument("--group", action="store_true")
    ap.add_argument("--outdir", default="persubject")
    ap.add_argument("--cache", default=str(PROJ_DIR / "results" / "features_cache"))
    args = ap.parse_args()
    cfg = load_config()
    subs = (cfg["data"]["subjects"]["all"] if args.group else
            (args.subjects or cfg["data"]["subjects"]["pilot"]))
    cache_dir = Path(args.cache)
    out_dir = PROJ_DIR / "results" / args.outdir
    out_dir.mkdir(parents=True, exist_ok=True)
    all_rows = []
    for sub in subs:
        print(f"[{sub}]")
        all_rows += process_subject(cfg, sub, cache_dir, out_dir)
    df = pd.DataFrame(all_rows)
    # per-subject CSV name when a single subject is processed (parallel-safe for SLURM arrays)
    csv_name = f"efp_persubject_{subs[0]}.csv" if len(subs) == 1 else "efp_persubject.csv"
    df.to_csv(out_dir / csv_name, index=False)
    print(f"\nSaved {out_dir / csv_name}  ({len(df)} rows)")
    # quick summary
    if len(df):
        piv = df.pivot_table(index=["target", "resolution"], columns="method",
                             values="mean_r", aggfunc="mean")
        print(piv.round(3))


if __name__ == "__main__":
    main()
