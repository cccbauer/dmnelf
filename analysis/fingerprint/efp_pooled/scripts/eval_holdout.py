#!/usr/bin/env python3
"""
eval_holdout.py  (cluster or local; numpy only — no mne, runs on a login node)
-----------------------------------------------------------------------------
Score a frozen EFP model on a cohort's cached features. Built to answer the question
`efp_epoc_model.npz` has never been asked: **how does it do on subjects it never saw?**

The shipped model was trained on DMNELF only (efp19_subs.txt is all-dmnelf), so every rtBPD
subject is genuinely held out for it — no refitting needed to get an honest number.

Feature/design construction is copied byte-for-byte in behaviour from efp_epoc/export_model.py
(`subject_designs`) so that scoring is on exactly the features the model was trained on:
  per channel (montage order) -> make_delay_design -> per-run z-score -> channel-major column_stack
  mask t_idx >= BASELINE_TR + HRF_DROP, target z-scored per run.
`make_delay_design` and `load_subject_features` are reimplemented here (identical logic) purely to
avoid importing efp_features.py, which pulls in mne and gets OOM-killed on the login node.

PDA convention follows mindwear/compare_engine.py:_prepare(): observed CEN/DMN are jointly
normalized (common centre+scale, preserving their relationship) then differenced; predicted PDA is
pred_cen - pred_dmn, which is what the deployed decoder actually feeds back.

Usage:
  python eval_holdout.py --model <model.npz> --cohort rtbpd --out results/x.csv
"""
import argparse
import json
from pathlib import Path

import numpy as np

FP = Path("/projects/swglab/data/DMNELF/analysis/fingerprint")
BASELINE_TR, HRF_DROP = 25, 5          # must match export_model.py
CENMEAN = FP / "efp_epoc" / "cen_mean_cache"

COHORTS = {
    # name        -> (feature cache dir,                                   cenmean filename prefix)
    "dmnelf":    (FP / "19_fingerprint/efp_meirhasson/results/features_cache",     "cenmean_dmnelf_"),
    "rtbpd":     (FP / "efp_meirhasson/results/features_cache_rtbpd",              "cenmean_rtbpd_"),
    "rtbpd_nf2": (FP / "efp_meirhasson/results/features_cache_rtbpd_nf2",          "cenmean_rtbpd_nf2_"),
}


def make_delay_design(bp_run, n_delays):
    """Sliding-delay design for one run. bp_run (n_bands, n_out) -> X (n_valid, n_bands*n_delays).
    Row t uses lags [t, t-1, ..., t-(n_delays-1)]. Returns (X, offset=n_delays-1)."""
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


def zscore(a):
    a = np.asarray(a, float)
    return (a - np.nanmean(a)) / (np.nanstd(a) + 1e-12)


def joint_norm(a, c):
    """compare_engine's normalization: common centre+scale over both tracks."""
    allv = np.concatenate([a, c])
    mu, sd = np.nanmean(allv), np.nanstd(allv) + 1e-9
    return (a - mu) / sd, (c - mu) / sd


def mean_ci(mean, sem, n):
    """95% CI for a MEAN of per-subject correlations.

    NOT Fisher: Fisher's 1/sqrt(n-3) applies to a single r from n paired observations, and using it
    with n = number of subjects yields absurdly wide intervals. The quantity here is a sample mean
    across subjects, so the interval is t-based on the between-subject SEM.
    """
    if not np.isfinite(mean) or not np.isfinite(sem) or n < 3:
        return np.nan, np.nan
    try:
        from scipy.stats import t as tdist
        crit = float(tdist.ppf(0.975, n - 1))
    except Exception:
        crit = 1.96
    return float(mean - crit * sem), float(mean + crit * sem)


def sign_flip_p(rs, n_perm=10000, seed=0):
    """Sign-flip permutation on per-subject r's (efp_meirhasson convention)."""
    rs = np.asarray([r for r in rs if np.isfinite(r)], float)
    if rs.size < 3:
        return np.nan
    rng = np.random.default_rng(seed)
    obs = rs.mean()
    null = (rng.choice([-1.0, 1.0], size=(n_perm, rs.size)) * rs).mean(axis=1)
    return float((np.sum(np.abs(null) >= abs(obs)) + 1) / (n_perm + 1))


def run_designs(runs, ch_names, eidx, n_delays, targets, skips=None):
    """Yield (run_id, X_masked, observed CEN, observed DMN) for each scorable run."""
    for rd in runs:
        rd = rd.item() if hasattr(rd, "item") else rd
        r = str(rd["run"])
        if r not in targets:
            if skips is not None:
                skips.append(f"run{r}: no target (have {sorted(targets)})")
            continue
        per_ch, off = [], None
        for ci in eidx:
            Xc, off = make_delay_design(rd["bp_tr"][ci], n_delays)
            per_ch.append((Xc - Xc.mean(0)) / (Xc.std(0) + 1e-12))
        if not per_ch or per_ch[0].shape[0] == 0:
            if skips is not None:
                skips.append(f"run{r}: empty design")
            continue
        nvalid = per_ch[0].shape[0]
        t_idx = off + np.arange(nvalid)
        y_cen = np.asarray(targets[r]["CEN"], float)[off:off + nvalid]
        y_dmn = np.asarray(targets[r]["DMN"], float)[off:off + nvalid]
        m = (t_idx >= BASELINE_TR + HRF_DROP) & np.isfinite(y_cen) & np.isfinite(y_dmn)
        if m.sum() < 20:
            if skips is not None:
                skips.append(f"run{r}: only {int(m.sum())} usable TRs (<20)")
            continue
        X = np.column_stack([c[m] for c in per_ch])
        yield r, X, y_cen[m], y_dmn[m]


def load_targets(prefix, sub, key_variant):
    """key_variant: 'clean' -> run{N}/run{N}_dmn ; 'gsr' -> run{N}_gsr/run{N}_dmn."""
    p = CENMEAN / f"{prefix}{sub}.npz"
    if not p.exists():
        return {}
    z = np.load(p, allow_pickle=True)
    out = {}
    for n in range(1, 9):
        cen_k = f"run{n}_gsr" if key_variant == "gsr" else f"run{n}"
        dmn_k = f"run{n}_dmn"
        if cen_k in z.files and dmn_k in z.files:
            # Key by str: the feature cache stores rd["run"] as a string ('1', '2', ...).
            out[str(n)] = {"CEN": z[cen_k], "DMN": z[dmn_k]}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=str(FP / "efp_epoc" / "efp_epoc_model.npz"))
    ap.add_argument("--cohort", choices=list(COHORTS), default="rtbpd")
    ap.add_argument("--subs", default=None, help="comma list; default = every subject in the cache")
    ap.add_argument("--targets", choices=["clean", "gsr"], default="clean")
    ap.add_argument("--out", default=None)
    ap.add_argument("--label", default="", help="free-text note for the CSV/JSON")
    a = ap.parse_args()

    M = np.load(a.model, allow_pickle=True)
    channels = [str(c) for c in M["channels"]]
    n_delays = int(M["n_delays"])
    has_pda = "pda_coef" in M.files
    cache_dir, prefix = COHORTS[a.cohort]

    if a.subs:
        subs = [s.strip() for s in a.subs.split(",") if s.strip()]
    else:
        subs = sorted(p.name.replace("_efp.npz", "") for p in Path(cache_dir).glob("*_efp.npz"))

    print(f"model      : {a.model}")
    print(f"             montage={M['montage'] if 'montage' in M.files else '?'} "
          f"n_ch={len(channels)} n_delays={n_delays} "
          f"n_train_subjects={int(M['n_train_subjects']) if 'n_train_subjects' in M.files else '?'} "
          f"pda_coef={'YES' if has_pda else 'no (uses cen-dmn)'}")
    print(f"cohort     : {a.cohort}  ({len(subs)} subjects)   targets={a.targets}")
    print(f"features   : {cache_dir}")
    print()

    rows = []
    for sub in subs:
        try:
            runs, ch_names = load_subject_features(cache_dir, sub)
        except FileNotFoundError:
            print(f"  {sub}: no feature cache, skip"); continue
        eidx = [ch_names.index(c) for c in channels if c in ch_names]
        if len(eidx) != len(channels):
            missing = [c for c in channels if c not in ch_names]
            print(f"  {sub}: missing channels {missing}, skip"); continue
        targets = load_targets(prefix, sub, a.targets)
        if not targets:
            print(f"  {sub}: no cenmean targets, skip"); continue

        skips = []
        n_before = len(rows)
        for r, X, y_cen, y_dmn in run_designs(runs, ch_names, eidx, n_delays, targets, skips):
            Xs_cen = (X - M["cen_feat_mean"]) / M["cen_feat_std"]
            Xs_dmn = (X - M["dmn_feat_mean"]) / M["dmn_feat_std"]
            p_cen = Xs_cen @ M["cen_coef"] + float(M["cen_intercept"])
            p_dmn = Xs_dmn @ M["dmn_coef"] + float(M["dmn_intercept"])
            if has_pda:
                Xs_pda = (X - M["pda_feat_mean"]) / M["pda_feat_std"]
                p_pda = Xs_pda @ M["pda_coef"] + float(M["pda_intercept"])
            else:
                p_pda = p_cen - p_dmn
            o_cen_j, o_dmn_j = joint_norm(y_cen, y_dmn)
            obs = {"CEN": zscore(y_cen), "DMN": zscore(y_dmn), "PDA": o_cen_j - o_dmn_j}
            pred = {"CEN": p_cen, "DMN": p_dmn, "PDA": p_pda}
            row = {"subject": sub, "run": r, "n": int(X.shape[0])}
            for k in ("CEN", "DMN", "PDA"):
                row[k] = float(np.corrcoef(pred[k], obs[k])[0, 1])
            rows.append(row)
            print(f"  {sub} run{r}: n={row['n']:3d}  CEN {row['CEN']:+.3f}  "
                  f"DMN {row['DMN']:+.3f}  PDA {row['PDA']:+.3f}")
        if len(rows) == n_before:
            print(f"  {sub}: NO runs scored -> {'; '.join(skips) if skips else 'cache had no runs'}")

    if not rows:
        print("\nNo scorable runs."); return

    print(f"\n{'='*72}\nSUMMARY — {a.cohort} (n_runs={len(rows)})")
    subs_seen = sorted({r["subject"] for r in rows})
    summary = {}
    for k in ("CEN", "DMN", "PDA"):
        per_sub = [float(np.mean([r[k] for r in rows if r["subject"] == s])) for s in subs_seen]
        per_sub = [v for v in per_sub if np.isfinite(v)]
        mean = float(np.mean(per_sub)); sem = float(np.std(per_sub, ddof=1) / np.sqrt(len(per_sub)))
        lo, hi = mean_ci(mean, sem, len(per_sub))
        p = sign_flip_p(per_sub)
        summary[k] = {"subject_mean_r": mean, "sem": sem, "n_subjects": len(per_sub),
                      "ci95": [lo, hi], "sign_flip_p": p,
                      "per_subject_min": float(np.min(per_sub)), "per_subject_max": float(np.max(per_sub))}
        print(f"  {k}: subject-mean r = {mean:+.3f} ± {sem:.3f} (SEM, n={len(per_sub)})   "
              f"CI95 [{lo:+.3f},{hi:+.3f}]   sign-flip p = {p:.4f}   "
              f"range [{np.min(per_sub):+.3f},{np.max(per_sub):+.3f}]")

    if a.out:
        out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
        import csv
        with open(out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["subject", "run", "n", "CEN", "DMN", "PDA"])
            w.writeheader(); w.writerows(rows)
        meta = {"model": a.model, "cohort": a.cohort, "targets": a.targets, "label": a.label,
                "held_out": a.cohort.startswith("rtbpd"), "summary": summary}
        Path(str(out).replace(".csv", "_summary.json")).write_text(json.dumps(meta, indent=2))
        print(f"\nwrote {out} and {str(out).replace('.csv','_summary.json')}")


if __name__ == "__main__":
    main()
