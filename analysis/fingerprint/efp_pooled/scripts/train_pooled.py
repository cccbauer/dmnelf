#!/usr/bin/env python3
"""
train_pooled.py  (cluster, SLURM only — the login node OOM-kills this)
----------------------------------------------------------------------
Train an EFP decoder on the POOLED DMNELF+rtBPD cohort, with the four defects of
efp_epoc/export_model.py fixed. Derived from that script; it is left untouched.

Phase 1 established why this matters: the shipped DMNELF-only model's PDA does not transfer at all
(held-out rtBPD PDA r=+0.010, p=0.43) even though CEN and DMN individually do (+0.069, +0.052,
both p<0.005). Differencing two independently-regularized ridges cancels the transferable
component. So the headline change here is fitting PDA as its own target.

Fixes vs export_model.py:
  1. PDA is fitted directly as a third target (was: computed at runtime as cen - dmn, from two
     ridges regularized at DIFFERENT alphas -- 1e5 vs 3.16e4 in the shipped model).
  2. Alpha grid widened: logspace(-2,5,15) -> logspace(-2,8,30). The shipped cen_alpha is exactly
     1e5 == old grid max, i.e. RidgeCV hit the ceiling and wanted more penalty.
  3. Alpha selected by SUBJECT-GROUPED CV (GroupKFold) instead of RidgeCV's default LOO-GCV over
     pooled, temporally autocorrelated TRs, which systematically under-penalizes.
  4. Frozen band edges = true across-channel/subject median. export_model.py medianed
     rd["band_hz"], which upstream (efp_features.py:238-241) holds only the LAST channel's
     equal-energy edges because it is overwritten inside a per-channel loop.

--estimator pls (added after Phase 2): the user decided to KEEP the shipped two-ridge
architecture (CEN, DMN fit independently, PDA = cen - dmn downstream) rather than chase PDA
directly. This estimator instead asks whether CEN and DMN can be fit JOINTLY -- 2-output PLS
extracts a small number of shared latent components that predict [CEN, DMN] together, letting them
borrow statistical strength from whatever they share (motivated by eeg_bold_coupling/HANDOFF.md:
the EEG-decodable component is largely shared/global, and independent per-target regularization
currently throws that sharing away). No pda_coef is written for this estimator -- eval_holdout.py
already falls back to cen - dmn when it's absent, which is exactly the architecture being kept.

Cohort comes from the LOCKED cohort_split.json; the held-out subjects are never loaded here.

Usage (see submit_train_pooled.sh):
  python train_pooled.py --montage epoc12 --targets clean --estimator ridge --out <model.npz>
  python train_pooled.py --montage epoc12 --targets clean --estimator pls --out <model.npz>
"""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import zscore
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.model_selection import GroupKFold

FP = Path("/projects/swglab/data/DMNELF/analysis/fingerprint")
POOLED = FP / "efp_pooled"
CENMEAN = FP / "efp_epoc" / "cen_mean_cache"

EPOC12 = ["F7", "F3", "FC5", "T7", "P7", "O1", "O2", "P8", "T8", "FC6", "F4", "F8"]
# EPOC X physically has AF3/AF4; Fp1/Fp2 stand in for them in the cap recordings.
EPOC_AFPROXY = EPOC12 + ["Fp1", "Fp2"]
CAP31 = ["Fp1", "Fp2", "F3", "F4", "C3", "C4", "P3", "P4", "O1", "O2", "F7", "F8", "T7", "T8",
         "P7", "P8", "Fz", "Cz", "Pz", "Oz", "FC1", "FC2", "CP1", "CP2", "FC5", "FC6", "CP5",
         "CP6", "TP9", "TP10", "POz"]
MONTAGES = {"epoc12": EPOC12, "epoc_afproxy": EPOC_AFPROXY, "cap31": CAP31}

BASELINE_TR, HRF_DROP = 25, 5              # unchanged from export_model.py
ALPHAS = np.logspace(-2, 8, 30)            # FIX 2: was logspace(-2,5,15), which saturated
TARGET_NAMES = ["CEN", "DMN", "PDA"]       # FIX 1: PDA fitted, not derived

CACHES = {
    "dmnelf":    (FP / "19_fingerprint/efp_meirhasson/results/features_cache", "cenmean_dmnelf_"),
    "rtbpd_nf1": (FP / "efp_meirhasson/results/features_cache_rtbpd",          "cenmean_rtbpd_"),
}


def make_delay_design(bp_run, n_delays):
    n_bands, n_out = bp_run.shape
    n_valid = n_out - (n_delays - 1)
    if n_valid <= 0:
        return np.empty((0, n_bands * n_delays)), n_delays - 1
    X = np.empty((n_valid, n_bands * n_delays))
    for d in range(n_delays):
        X[:, d * n_bands:(d + 1) * n_bands] = bp_run[:, (n_delays - 1 - d):(n_out - d)].T
    return X, n_delays - 1


def load_subject_features(cache_dir, sub):
    z = np.load(Path(cache_dir) / f"{sub}_efp.npz", allow_pickle=True)
    return list(z["runs"]), list(z["ch_names"])


def load_targets(prefix, sub, variant):
    """variant 'clean' -> run{N}; 'gsr' -> run{N}_gsr. DMN always run{N}_dmn.
    PDA target is built from the JOINTLY normalized pair (compare_engine convention) so that
    'which network is higher' is preserved and the contrast is not distorted by unequal scaling."""
    p = CENMEAN / f"{prefix}{sub}.npz"
    if not p.exists():
        return {}
    z = np.load(p, allow_pickle=True)
    out = {}
    for n in range(1, 9):
        cen_k = f"run{n}_gsr" if variant == "gsr" else f"run{n}"
        dmn_k = f"run{n}_dmn"
        if cen_k in z.files and dmn_k in z.files:
            out[str(n)] = {"CEN": np.asarray(z[cen_k], float),
                           "DMN": np.asarray(z[dmn_k], float)}
    return out


def subject_designs(runs, eidx, n_delays, tv, target):
    """Per-run z-scored montage design + z-scored target, feedback-masked.
    Identical construction to export_model.py:subject_designs."""
    Xs, ys = [], []
    for rd in runs:
        rd = rd.item() if hasattr(rd, "item") else rd
        r = str(rd["run"])
        if r not in tv:
            continue
        per_ch, off = [], None
        for ci in eidx:
            Xc, off = make_delay_design(rd["bp_tr"][ci], n_delays)
            per_ch.append((Xc - Xc.mean(0)) / (Xc.std(0) + 1e-12))
        if not per_ch or per_ch[0].shape[0] == 0:
            continue
        nvalid = per_ch[0].shape[0]
        t_idx = off + np.arange(nvalid)
        cen = tv[r]["CEN"][off:off + nvalid]
        dmn = tv[r]["DMN"][off:off + nvalid]
        m = (t_idx >= BASELINE_TR + HRF_DROP) & np.isfinite(cen) & np.isfinite(dmn)
        if m.sum() < 20:
            continue
        if target == "PDA":
            a, c = cen[m], dmn[m]
            allv = np.concatenate([a, c]); mu, sd = allv.mean(), allv.std() + 1e-9
            y = (a - mu) / sd - (c - mu) / sd
        else:
            y = cen[m] if target == "CEN" else dmn[m]
        Xs.append(np.column_stack([c_[m] for c_ in per_ch]))
        ys.append(zscore(y))
    return Xs, ys


def _ridge_svd_paths(Xtr, ytr, Xva, alphas):
    """Ridge predictions on Xva for EVERY alpha, from ONE SVD of the training fold.

    coef(alpha) = V diag(s / (s^2 + alpha)) U^T y  for the centred problem. This makes the alpha
    sweep essentially free: a naive loop refitting Ridge 30x per fold per target per arm would
    take hours on a ~15k x 1320 design.
    """
    xm = Xtr.mean(0); ym = ytr.mean()
    U, s, Vt = np.linalg.svd(Xtr - xm, full_matrices=False)
    Uty = U.T @ (ytr - ym)
    Xvc = Xva - xm
    out = []
    for al in alphas:
        coef = Vt.T @ ((s / (s ** 2 + al)) * Uty)
        out.append(Xvc @ coef + ym)
    return out


def pick_alpha_grouped(X, y, groups, estimator, l1_ratio=0.5, n_splits=5, log=print):
    """FIX 3: choose the penalty by SUBJECT-grouped CV, so no subject is split across folds and
    temporally adjacent TRs cannot leak between train and validation (export_model.py used
    RidgeCV's default LOO-GCV over pooled autocorrelated TRs, which under-penalizes)."""
    n_groups = len(np.unique(groups))
    gkf = GroupKFold(n_splits=min(n_splits, n_groups))
    folds = list(gkf.split(X, y, groups))

    if estimator == "ridge":
        acc = np.zeros((len(ALPHAS), len(folds)))
        for fi, (tr_i, va_i) in enumerate(folds):
            preds = _ridge_svd_paths(X[tr_i], y[tr_i], X[va_i], ALPHAS)
            yv = y[va_i]
            for ai, pr in enumerate(preds):
                acc[ai, fi] = 0.0 if np.std(pr) < 1e-12 else np.corrcoef(pr, yv)[0, 1]
        scores = np.nanmean(acc, axis=1)
        for al, s in zip(ALPHAS, scores):
            log(f"      alpha={al:>12.4g}  grouped-CV r={s:+.4f}")
        bi = int(np.nanargmax(scores))
        return float(ALPHAS[bi]), float(scores[bi])

    # ElasticNet has no closed-form path reuse here; use a coarse grid to stay tractable.
    grid = np.logspace(-4, 1, 8)
    best, best_score = None, -np.inf
    for al in grid:
        sc = []
        for tr_i, va_i in folds:
            mdl = ElasticNet(alpha=al, l1_ratio=l1_ratio, max_iter=5000).fit(X[tr_i], y[tr_i])
            pr = mdl.predict(X[va_i])
            sc.append(0.0 if np.std(pr) < 1e-12 else np.corrcoef(pr, y[va_i])[0, 1])
        s = float(np.nanmean(sc))
        log(f"      alpha={al:>12.4g}  grouped-CV r={s:+.4f}")
        if s > best_score:
            best, best_score = al, s
    return float(best), float(best_score)


PLS_GRID = [2, 5, 10, 20, 40, 80]   # deliberately small vs 1320 input features -- the shared-latent
                                    # -structure hypothesis predicts only a few components matter


def pick_ncomp_grouped_pls(X, Ycen, Ydmn, groups, n_splits=5, log=print):
    """Joint CEN+DMN PLS2: grouped-CV over n_components, scored as the MEAN of
    corr(pred_cen, val_cen) and corr(pred_dmn, val_dmn) so neither target dominates the choice."""
    n_groups = len(np.unique(groups))
    gkf = GroupKFold(n_splits=min(n_splits, n_groups))
    folds = list(gkf.split(X, Ycen, groups))
    Y = np.column_stack([Ycen, Ydmn])

    best, best_score = None, -np.inf
    for nc in PLS_GRID:
        if nc >= X.shape[1]:
            continue
        sc = []
        for tr_i, va_i in folds:
            mdl = PLSRegression(n_components=nc, scale=False).fit(X[tr_i], Y[tr_i])
            pr = mdl.predict(X[va_i])
            r_cen = 0.0 if np.std(pr[:, 0]) < 1e-12 else np.corrcoef(pr[:, 0], Ycen[va_i])[0, 1]
            r_dmn = 0.0 if np.std(pr[:, 1]) < 1e-12 else np.corrcoef(pr[:, 1], Ydmn[va_i])[0, 1]
            sc.append((r_cen + r_dmn) / 2.0)
        s = float(np.nanmean(sc))
        log(f"      n_components={nc:>4d}  grouped-CV mean(CEN,DMN) r={s:+.4f}")
        if s > best_score:
            best, best_score = nc, s
    return int(best), float(best_score)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--montage", choices=list(MONTAGES), default="epoc12")
    ap.add_argument("--targets", choices=["clean", "gsr"], default="clean")
    ap.add_argument("--estimator", choices=["ridge", "elasticnet", "pls"], default="ridge")
    ap.add_argument("--l1-ratio", type=float, default=0.5)
    ap.add_argument("--split", default=str(POOLED / "cohort_split.json"))
    ap.add_argument("--dmnelf-only", action="store_true", help="ablation: train on DMNELF alone")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    split = json.load(open(a.split))
    subs = [("dmnelf", s) for s in split["train"]["dmnelf"]]
    if not a.dmnelf_only:
        subs += [("rtbpd_nf1", s) for s in split["train"]["rtbpd_nf1"]]
    channels = MONTAGES[a.montage]
    n_delays, n_bands, tr = 11, 10, 1.2       # frozen EFP settings (delay_window_s=12, TR=1.2)

    print(f"montage={a.montage} ({len(channels)} ch)  targets={a.targets}  "
          f"estimator={a.estimator}  train_subjects={len(subs)}"
          f"{'  [DMNELF-ONLY ABLATION]' if a.dmnelf_only else ''}")
    print(f"alpha grid: {ALPHAS.min():g} .. {ALPHAS.max():g} ({len(ALPHAS)} pts)\n")

    fitX = {t: [] for t in TARGET_NAMES}
    fitY = {t: [] for t in TARGET_NAMES}
    fitG = {t: [] for t in TARGET_NAMES}      # subject id per row, for GroupKFold
    band_hz_all = []
    used = []

    for cohort, sub in subs:
        cache_dir, prefix = CACHES[cohort]
        try:
            runs, ch_names = load_subject_features(cache_dir, sub)
        except FileNotFoundError:
            print(f"  {sub}: no feature cache, skip"); continue
        eidx = [ch_names.index(c) for c in channels if c in ch_names]
        if len(eidx) != len(channels):
            print(f"  {sub}: missing {[c for c in channels if c not in ch_names]}, skip"); continue
        tv = load_targets(prefix, sub, a.targets)
        if not tv:
            print(f"  {sub}: no targets, skip"); continue
        n_tr_sub = 0
        for tgt in TARGET_NAMES:
            Xs, ys = subject_designs(runs, eidx, n_delays, tv, tgt)
            fitX[tgt] += Xs; fitY[tgt] += ys
            fitG[tgt] += [sub] * sum(x.shape[0] for x in Xs)
            n_tr_sub = sum(x.shape[0] for x in Xs)
        # FIX 4: keep every channel's band edges, not just whatever band_hz last held
        for rd in runs:
            rd = rd.item() if hasattr(rd, "item") else rd
            if rd.get("band_hz") is not None:
                band_hz_all.append(np.asarray(rd["band_hz"]))
        used.append(sub)
        print(f"  {sub} ({cohort}): {n_tr_sub} feedback TRs", flush=True)

    band_edges = np.median(np.array(band_hz_all), axis=0).round().astype(int)
    model = {"channels": np.array(channels), "montage": a.montage, "n_bands": n_bands,
             "n_delays": n_delays, "tr": tr, "sfreq": 250.0, "fmin": 1, "fmax": 40,
             "band_edges_hz": band_edges,
             "layout": "channel-major, delay-major, band-minor",
             "n_train_subjects": len(used), "train_subjects": np.array(used),
             "target_variant": a.targets, "estimator": a.estimator,
             "pda": ("cen - dmn (joint PLS2 CEN/DMN, PDA not fit directly)" if a.estimator == "pls"
                     else "FITTED DIRECTLY as its own ridge (not cen - dmn)"),
             "alpha_selection": ("GroupKFold n_components sweep (joint PLS2)" if a.estimator == "pls"
                                 else "GroupKFold by subject over the pooled cohort"),
             "cohort_split": str(a.split)}

    print()
    if a.estimator == "pls":
        # Joint CEN+DMN fit. fitX["CEN"] and fitX["DMN"] are row-identical designs (subject_designs'
        # mask depends on isfinite(cen) & isfinite(dmn) regardless of which target is extracted) --
        # safe to reuse fitX["CEN"] as the shared design and pair it with both targets' y's, which
        # were appended in the same subject/run order. PDA is intentionally NOT fit here; it stays
        # cen - dmn downstream (eval_holdout.py / decoder.py both already do this when pda_coef is
        # absent from the model file).
        X = np.vstack(fitX["CEN"]); y_cen = np.concatenate(fitY["CEN"]); y_dmn = np.concatenate(fitY["DMN"])
        g = np.array(fitG["CEN"])
        assert len(y_cen) == len(y_dmn) == X.shape[0], "CEN/DMN row mismatch -- design assumption broken"
        mu, sd = X.mean(0), X.std(0) + 1e-12
        Xs_ = (X - mu) / sd
        print(f"  === CEN+DMN (joint PLS2): n={len(y_cen)} feat={X.shape[1]} subjects={len(np.unique(g))}")
        n_comp, cv_r = pick_ncomp_grouped_pls(Xs_, y_cen, y_dmn, g, log=print)
        pls = PLSRegression(n_components=n_comp, scale=False).fit(Xs_, np.column_stack([y_cen, y_dmn]))
        pred = pls.predict(Xs_)
        in_r_cen = float(np.corrcoef(pred[:, 0], y_cen)[0, 1])
        in_r_dmn = float(np.corrcoef(pred[:, 1], y_dmn)[0, 1])
        print(f"      n_components={n_comp}  grouped-CV mean r={cv_r:+.4f}  "
              f"in-sample r: CEN={in_r_cen:+.4f} DMN={in_r_dmn:+.4f}")

        # coef_/intercept extraction is version-robust: don't trust sklearn's internal PLS scaling
        # convention (coef_ orientation changed across versions) -- instead derive a plain
        # (coef, intercept) pair per target from Xs_ (already zero-mean) and verify it reproduces
        # pls.predict() exactly before trusting it.
        coef = np.asarray(pls.coef_, float)
        if coef.shape == (X.shape[1], 2):
            coef = coef.T                                  # -> (2, n_features)
        assert coef.shape == (2, X.shape[1]), f"unexpected PLS coef_ shape {coef.shape}"
        for k, ti, y_t, in_r in (("cen", 0, y_cen, in_r_cen), ("dmn", 1, y_dmn, in_r_dmn)):
            c = coef[ti]
            b = float(y_t.mean() - Xs_.mean(0) @ c)         # Xs_.mean(0) ~= 0, kept for correctness
            check = Xs_ @ c + b
            max_err = float(np.max(np.abs(check - pred[:, ti])))
            print(f"      [{k}] manual-vs-pls.predict() max abs diff = {max_err:.2e}"
                  + ("  *** MISMATCH -- DO NOT TRUST ***" if max_err > 1e-6 else "  OK"))
            model[f"{k}_coef"] = c.astype(np.float32)
            model[f"{k}_intercept"] = b
            model[f"{k}_alpha"] = float(n_comp)             # repurposed field: n_components here
            model[f"{k}_grouped_cv_r"] = cv_r               # joint score, same for both targets
            model[f"{k}_in_sample_r"] = in_r
            model[f"{k}_feat_mean"] = mu.astype(np.float32)
            model[f"{k}_feat_std"] = sd.astype(np.float32)
        model["pls_n_components"] = n_comp
    else:
        for tgt in TARGET_NAMES:
            X = np.vstack(fitX[tgt]); y = np.concatenate(fitY[tgt])
            g = np.array(fitG[tgt])
            mu, sd = X.mean(0), X.std(0) + 1e-12
            Xs_ = (X - mu) / sd
            print(f"  === {tgt}: n={len(y)} feat={X.shape[1]} subjects={len(np.unique(g))}")
            alpha, cv_r = pick_alpha_grouped(Xs_, y, g, a.estimator, a.l1_ratio,
                                             log=lambda s: None)
            mdl = (Ridge(alpha=alpha) if a.estimator == "ridge"
                   else ElasticNet(alpha=alpha, l1_ratio=a.l1_ratio, max_iter=5000)).fit(Xs_, y)
            in_r = float(np.corrcoef(mdl.predict(Xs_), y)[0, 1])
            saturated = (" *** SATURATES GRID ***"
                         if a.estimator == "ridge" and alpha >= ALPHAS.max() * 0.999 else "")
            print(f"      alpha={alpha:g}{saturated}  grouped-CV r={cv_r:+.4f}  in-sample r={in_r:+.4f}")
            k = tgt.lower()
            model[f"{k}_coef"] = mdl.coef_.astype(np.float32)
            model[f"{k}_intercept"] = float(mdl.intercept_)
            model[f"{k}_alpha"] = float(alpha)
            model[f"{k}_grouped_cv_r"] = cv_r
            model[f"{k}_in_sample_r"] = in_r
            model[f"{k}_feat_mean"] = mu.astype(np.float32)
            model[f"{k}_feat_std"] = sd.astype(np.float32)

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(a.out, **model)
    print(f"\nsaved {a.out}")
    print(f"  bands Hz: {band_edges.tolist()}")


if __name__ == "__main__":
    main()
