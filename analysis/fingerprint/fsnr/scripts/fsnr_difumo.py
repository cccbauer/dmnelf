#!/usr/bin/env python3
"""
fsnr_difumo.py
--------------
DMN-as-noise investigation: variability quenching + DiFuMo-resolved f-SNR.

Phase-1 (fsnr_fmri.py) found f-SNR tracks PDA/CEN regulation but NOT DMN *mean*
suppression. The f-SNR framework calls DMN the *noise*, so the relevant quantity is DMN
*variance* (endogenous fluctuation) and its reduction during regulation — variability
quenching. This script tests that, with DiFuMo ROIs (nilearn Yeo-7 labels) for
network-specificity, and asks whether DMN decluttering couples to f-SNR / regulation.

z = within-run {rest 0:25, feedback 30:125 (HRF-lag drop)}.
Personalized DMN=col64, CEN=col65 (scale-retained) for absolute variance; the 64 DiFuMo
components (z-scored per run → all on unit-variance footing) for the ROI quench profile.
"""
from pathlib import Path
import numpy as np, pandas as pd, re
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from fsnr_fmri import (canonical_hrf, task_regressor, glm_fsnr,
                       BASELINE_TR, HRF_DROP, DMN_I, CEN_I)

PROJ = Path(__file__).resolve().parent.parent
DATA = PROJ / "data"
RES = PROJ / "results"; RES.mkdir(exist_ok=True)


def difumo_groups():
    """Yeo-7 network -> 0-based DiFuMo-64 component indices."""
    from nilearn.datasets import fetch_atlas_difumo
    labs = fetch_atlas_difumo(dimension=64, resolution_mm=2).labels
    df = pd.DataFrame(labs)
    y = df["yeo_networks7"].astype(str).values
    idx = lambda pred: np.array([i for i, v in enumerate(y) if pred(v)])
    return {
        "DMN":  idx(lambda v: v.startswith("Default")),
        "CEN":  idx(lambda v: v.startswith("Cont")),
        "ATTN": idx(lambda v: "Attn" in v),
        "VIS":  idx(lambda v: v.startswith("Vis")),
    }, df["difumo_names"].values, y


def var_split(r, n_tr):
    rest = r[:BASELINE_TR]
    fb = r[BASELINE_TR + HRF_DROP:n_tr]
    return rest.var(), fb.var()


def quench_db(vr, vf):
    return 10 * np.log10(vr / vf) if (vf > 1e-12 and vr > 0) else np.nan


def sign_flip_mean(x, n=10000, seed=0):
    """One-sample sign-flip: p that mean(x) > 0."""
    x = np.asarray([v for v in x if np.isfinite(v)])
    if len(x) < 3:
        return np.nan, np.nan
    obs = x.mean(); rng = np.random.default_rng(seed)
    null = (rng.choice([-1, 1], size=(n, len(x))) * np.abs(x)).mean(1)
    return obs, float((np.sum(null >= obs) + 1) / (n + 1))


def main():
    groups, names, yeo = difumo_groups()
    print("DiFuMo groups:", {k: len(v) for k, v in groups.items()})

    rows, roi_prof = [], []
    for f in sorted(DATA.glob("sub-*_task-feedback_run-*_features.npz")):
        m = re.match(r"sub-(\w+?)_task-feedback_run-(\d+)_features", f.stem)
        sub, run = m.group(1), int(m.group(2))
        z = np.load(f, allow_pickle=True)
        fm = np.asarray(z["fmri_features"], float); pda = np.asarray(z["pda"], float)
        n_tr = fm.shape[0]
        if n_tr < BASELINE_TR + HRF_DROP + 10:
            continue
        dmn, cen = fm[:, DMN_I], fm[:, CEN_I]
        row = dict(subject=sub, run=run)
        # personalized network quench (scale-retained)
        for nm, r in [("DMN", dmn), ("CEN", cen), ("PDA", pda)]:
            vr, vf = var_split(r, n_tr)
            row[f"varrest_{nm}"] = vr; row[f"varfb_{nm}"] = vf
            row[f"quench_{nm}"] = vr - vf; row[f"quenchdb_{nm}"] = quench_db(vr, vf)
        # DiFuMo network-group quench (z-scored components; per-comp var ratio, mean over group)
        comp_qdb = np.array([quench_db(*var_split(fm[:, i], n_tr)) for i in range(64)])
        roi_prof.append(comp_qdb)
        for g, ix in groups.items():
            row[f"grpquenchdb_{g}"] = np.nanmean(comp_qdb[ix]) if len(ix) else np.nan
            # feedback-window noise magnitude for this group (mean var during fb)
            row[f"grpnoisefb_{g}"] = np.mean([var_split(fm[:, i], n_tr)[1] for i in ix]) if len(ix) else np.nan
        row["grpquenchdb_WHOLE"] = np.nanmean(comp_qdb)
        # GLM regulation signal + beta (reuse phase-1)
        x = task_regressor(n_tr)
        db_p, beta_p, r2_p = glm_fsnr(pda, x)
        db_c, beta_c, r2_c = glm_fsnr(cen, x)
        row["glm_PDA_db"] = db_p; row["beta_PDA"] = beta_p; row["beta_CEN"] = beta_c
        # noise-reduction f-SNR: task signal over DMN feedback-noise (personalized)
        ss_sig = r2_p / (1 - r2_p) if (r2_p is not None and r2_p < 1) else np.nan   # PDA signal/resid
        row["fsnr_quench_PDA_db"] = quench_db(row["varrest_DMN"], row["varfb_DMN"])  # DMN declutter index
        rows.append(row)

    df = pd.DataFrame(rows).sort_values(["subject", "run"]).reset_index(drop=True)
    df.to_csv(RES / "fsnr_difumo.csv", index=False)
    prof = np.nanmean(np.array(roi_prof), 0)
    pd.DataFrame({"component": np.arange(1, 65), "difumo_name": names, "yeo7": yeo,
                  "quench_db": prof}).sort_values("quench_db", ascending=False)\
        .to_csv(RES / "difumo_quench_profile.csv", index=False)
    print(f"\n{len(df)} runs, {df.subject.nunique()} subjects\n")

    # ---- 1. is DMN variance quenched rest->feedback? DMN-specific? ----
    print("=== variability quenching (var_rest - var_fb; >0 = decluttered during feedback) ===")
    for nm in ["DMN", "CEN", "PDA"]:
        obs, p = sign_flip_mean(df[f"quench_{nm}"])
        print(f"  {nm:4s} (personalized)  mean={obs:+.3f}  p={p:.4f}  (>0 in {100*(df[f'quench_{nm}']>0).mean():.0f}% runs)  quenchdB={df[f'quenchdb_{nm}'].mean():+.2f}")
    for g in ["DMN", "CEN", "ATTN", "VIS"]:
        obs, p = sign_flip_mean(df[f"grpquenchdb_{g}"])
        print(f"  {g:4s} (DiFuMo grp)     quenchdB mean={obs:+.2f}  p={p:.4f}")
    ow, pw = sign_flip_mean(df["grpquenchdb_WHOLE"])
    print(f"  WHOLE-brain DiFuMo        quenchdB mean={ow:+.2f}  p={pw:.4f}")

    print("\n=== DMN-specificity: DMN quench vs whole-brain / vs CEN (paired, dB) ===")
    for other in ["WHOLE", "CEN", "ATTN"]:
        diff = df["grpquenchdb_DMN"] - df[f"grpquenchdb_{other}"]
        obs, p = sign_flip_mean(diff)
        print(f"  DMN − {other:5s}: {obs:+.2f} dB  p={p:.4f}  ({'DMN quenches MORE' if obs>0 else 'not more'})")

    # ---- 2. does DMN decluttering couple to f-SNR / regulation success? ----
    def corr(a, b):
        d = df[[a, b]].replace([np.inf, -np.inf], np.nan).dropna()
        return np.corrcoef(d[a], d[b])[0, 1] if len(d) > 3 else np.nan
    print("\n=== hypothesis: DMN quench (dB) couples to f-SNR / regulation? (across runs) ===")
    print(f"  corr(DMN quench, GLM f-SNR PDA) = {corr('grpquenchdb_DMN','glm_PDA_db'):+.2f}")
    print(f"  corr(DMN quench, β_PDA regulation) = {corr('grpquenchdb_DMN','beta_PDA'):+.2f}")
    print(f"  corr(personalized DMN quench, β_PDA) = {corr('quenchdb_DMN','beta_PDA'):+.2f}")
    # partial out whole-brain quench
    d = df[["grpquenchdb_DMN", "grpquenchdb_WHOLE", "beta_PDA"]].replace([np.inf,-np.inf],np.nan).dropna()
    if len(d) > 5:
        from numpy.linalg import lstsq
        def resid(y, x):
            X = np.column_stack([np.ones(len(x)), x]); b,*_=lstsq(X,y,rcond=None); return y - X@b
        rq = resid(d["grpquenchdb_DMN"].values, d["grpquenchdb_WHOLE"].values)
        rb = resid(d["beta_PDA"].values, d["grpquenchdb_WHOLE"].values)
        print(f"  partial corr(DMN quench, β_PDA | whole-brain) = {np.corrcoef(rq,rb)[0,1]:+.2f}")

    # ---- 3. reliability of DMN quench ----
    def icc(col):
        dd = df[["subject", col]].replace([np.inf,-np.inf],np.nan).dropna()
        g = dd.groupby("subject")[col]; b,w = g.mean().var(), g.var().mean()
        return b/(b+w) if (b+w)>0 else np.nan
    print(f"\n=== reliability: DMN quench ICC = {icc('grpquenchdb_DMN'):.2f} "
          f"(personalized {icc('quenchdb_DMN'):.2f}) ===")

    # ---- 4. spatial profile: top decluttering vs amplifying ROIs ----
    p = pd.read_csv(RES / "difumo_quench_profile.csv")
    print("\n=== top DECLUTTERING ROIs (var drops in feedback) ===")
    for _, r in p.head(6).iterrows():
        print(f"  {r.quench_db:+.2f} dB  {r.yeo7:16s} {r.difumo_name}")
    print("=== top AMPLIFYING ROIs (var rises in feedback) ===")
    for _, r in p.tail(6).iloc[::-1].iterrows():
        print(f"  {r.quench_db:+.2f} dB  {r.yeo7:16s} {r.difumo_name}")


if __name__ == "__main__":
    main()
