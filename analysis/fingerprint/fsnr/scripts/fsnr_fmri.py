#!/usr/bin/env python3
"""
fsnr_fmri.py
------------
Purely-fMRI functional signal-to-noise ratio (f-SNR) proxy for DMNELF neurofeedback,
faithful to the law-of-total-variance decomposition (Laukkonen 2026; Nath 2026):

    Var(r) = Var_z(E[r|z])   +   E_z(Var(r|z))
             \___ signal ___/       \__ noise __/     ,   f-SNR = signal / noise

where the goal-relevant cause z = {rest, feedback} is the WITHIN-RUN condition:
each feedback run = 25 TR rest baseline + 100 TR feedback (drop the first HRF_DROP
feedback TRs for the ~5 s hemodynamic lag). f-SNR reported in dB (10*log10), matching
Nath's ERP-SNR. r = personalized DMN (col 64), CEN (col 65), PDA (= CEN - DMN).

Variants:
  fSNR_{PDA,CEN,DMN} : per-network law-of-total-variance ratio.
  fSNR_CENDMN        : CEN task-signal over DMN endogenous noise (the "DMN = noise"
                       reading): signal_CEN / noise_DMN.

Also records directional regulation (delta = mean_feedback - mean_rest) to TEST the
hypothesis that higher PDA / lower DMN goes with higher f-SNR (non-circular: f-SNR is a
variance ratio, delta is a mean).

Reads local npz pulled from cyclic_features (fmri_features [n_tr,66], key 'pda').
Writes results/fsnr_fmri.csv (per run) + prints validation.
"""
from pathlib import Path
import numpy as np, pandas as pd, re
from scipy.stats import gamma

PROJ = Path(__file__).resolve().parent.parent
DATA = PROJ / "data"
RES = PROJ / "results"; RES.mkdir(exist_ok=True)
DMN_I, CEN_I = 64, 65
BASELINE_TR = 25      # first 25 TR = rest baseline
HRF_DROP = 5          # drop first 5 feedback TRs (HRF lag ~6 s at TR 1.2)
TR = 1.2


def canonical_hrf(tr=TR, length_s=32, delay=6, undershoot=16):
    t = np.arange(0, length_s, tr)
    h = gamma.pdf(t, delay) - gamma.pdf(t, undershoot) / 6.0
    return h / h.sum()


def task_regressor(n_tr, baseline_tr=BASELINE_TR):
    """HRF-convolved rest->feedback boxcar (the pseudo-target): 0 during rest, 1 during
    feedback. This is E[r|z] for continuous z; GLM f-SNR = explained/residual variance."""
    box = np.zeros(n_tr); box[baseline_tr:] = 1.0
    hrf = canonical_hrf()
    x = np.convolve(box, hrf, mode="full")[:n_tr]
    return x - x.mean()


def glm_fsnr(r, x):
    """Regress r on [1, x]; return (fsnr_db, beta, r2). signal=explained var, noise=resid."""
    X = np.column_stack([np.ones_like(x), x])
    beta, *_ = np.linalg.lstsq(X, r, rcond=None)
    fitted = X @ beta
    ss_model = np.sum((fitted - r.mean()) ** 2)
    ss_resid = np.sum((r - fitted) ** 2)
    fsnr = ss_model / ss_resid if ss_resid > 1e-12 else np.nan
    r2 = ss_model / (ss_model + ss_resid) if (ss_model + ss_resid) > 0 else np.nan
    return (10 * np.log10(fsnr) if fsnr and fsnr > 0 else np.nan), float(beta[1]), r2


def lotv(r, zmask):
    """Law-of-total-variance decomposition of series r under binary condition labels.
    zmask: boolean array, True=feedback, False=rest (only these TRs are used).
    Returns dict(signal, noise, fsnr, fsnr_db, mean_rest, mean_fb, delta)."""
    rest, fb = r[~zmask], r[zmask]
    n0, n1 = len(rest), len(fb); n = n0 + n1
    p0, p1 = n0 / n, n1 / n
    m0, m1 = rest.mean(), fb.mean()
    grand = p0 * m0 + p1 * m1
    signal = p0 * (m0 - grand) ** 2 + p1 * (m1 - grand) ** 2       # Var_z(E[r|z])
    noise = p0 * rest.var() + p1 * fb.var()                        # E_z(Var(r|z))
    fsnr = signal / noise if noise > 1e-12 else np.nan
    return dict(signal=signal, noise=noise, fsnr=fsnr,
                fsnr_db=10 * np.log10(fsnr) if fsnr and fsnr > 0 else np.nan,
                mean_rest=m0, mean_fb=m1, delta=m1 - m0)


def run_fsnr(fm, pda):
    n_tr = fm.shape[0]
    dmn, cen = fm[:, DMN_I], fm[:, CEN_I]
    # condition mask over the TRs we keep: rest = 0:BASELINE_TR, feedback = BASELINE_TR+HRF_DROP:end
    keep = np.zeros(n_tr, bool); zmask = np.zeros(n_tr, bool)
    keep[:BASELINE_TR] = True
    keep[BASELINE_TR + HRF_DROP:] = True
    zmask[BASELINE_TR + HRF_DROP:] = True
    kz, kk = zmask[keep], keep
    out = {}
    for nm, r in [("PDA", pda), ("CEN", cen), ("DMN", dmn)]:
        d = lotv(r[kk], kz)
        out[f"fsnr_{nm}_db"] = d["fsnr_db"]
        out[f"signal_{nm}"] = d["signal"]; out[f"noise_{nm}"] = d["noise"]
        out[f"delta_{nm}"] = d["delta"]; out[f"meanfb_{nm}"] = d["mean_fb"]
    # DMN-as-noise variant: CEN task-signal over DMN endogenous noise
    sC = out["signal_CEN"]; nD = out["noise_DMN"]
    out["fsnr_CENDMN_db"] = 10 * np.log10(sC / nD) if (nD > 1e-12 and sC > 0) else np.nan

    # ---- GLM / pseudo-target version (continuous z; uses all TRs, proper HRF) ----
    x = task_regressor(n_tr)
    ss_resid = {}
    for nm, r in [("PDA", pda), ("CEN", cen), ("DMN", dmn)]:
        db, beta, r2 = glm_fsnr(r, x)
        out[f"glm_{nm}_db"] = db          # explained/residual variance in dB
        out[f"beta_{nm}"] = beta          # signed HRF-aware regulation (replaces crude delta)
        out[f"r2_{nm}"] = r2
        # residual variance for the CEN/DMN glm variant
        Xd = np.column_stack([np.ones_like(x), x]); b, *_ = np.linalg.lstsq(Xd, r, rcond=None)
        ss_resid[nm] = np.sum((r - Xd @ b) ** 2) / len(r)
    # CEN signal (glm explained var) over DMN residual (endogenous) noise
    Xd = np.column_stack([np.ones_like(x), x]); bC, *_ = np.linalg.lstsq(Xd, cen, rcond=None)
    ssC = np.sum((Xd @ bC - cen.mean()) ** 2) / len(cen)
    out["glm_CENDMN_db"] = 10 * np.log10(ssC / ss_resid["DMN"]) if ss_resid["DMN"] > 1e-12 and ssC > 0 else np.nan
    return out


def main():
    rows = []
    for f in sorted(DATA.glob("sub-*_task-feedback_run-*_features.npz")):
        m = re.match(r"sub-(\w+?)_task-feedback_run-(\d+)_features", f.stem)
        sub, run = m.group(1), int(m.group(2))
        z = np.load(f, allow_pickle=True)
        fm = np.asarray(z["fmri_features"], float)
        pda = np.asarray(z["pda"], float)
        if fm.shape[0] < BASELINE_TR + HRF_DROP + 10:
            continue
        rows.append(dict(subject=sub, run=run, n_tr=fm.shape[0], **run_fsnr(fm, pda)))
    df = pd.DataFrame(rows).sort_values(["subject", "run"]).reset_index(drop=True)
    df.to_csv(RES / "fsnr_fmri.csv", index=False)
    print(f"{len(df)} runs, {df.subject.nunique()} subjects -> {RES/'fsnr_fmri.csv'}\n")

    def icc(col):
        d = df[["subject", col]].replace([np.inf, -np.inf], np.nan).dropna()
        g = d.groupby("subject")[col]
        btw, wth = g.mean().var(), g.var().mean()
        return btw / (btw + wth) if (btw + wth) > 0 else np.nan

    def corr(a, b):
        d = df[[a, b]].replace([np.inf, -np.inf], np.nan).dropna()
        return np.corrcoef(d[a], d[b])[0, 1] if len(d) > 3 else np.nan

    # ---- head-to-head: discrete (2-bin LoTV) vs GLM (HRF pseudo-target) ----
    print("=== DISCRETE (2-bin) vs GLM (HRF pseudo-target) — per network ===")
    print(f"  {'metric':10s} {'discrete dB':>12s} {'GLM dB':>10s} {'ICC disc':>9s} {'ICC glm':>8s}")
    for nm in ["PDA", "CEN", "DMN", "CENDMN"]:
        dc = f"fsnr_{nm}_db" if nm != "CENDMN" else "fsnr_CENDMN_db"
        gc = f"glm_{nm}_db"
        vd = df[dc].replace([np.inf, -np.inf], np.nan).dropna()
        vg = df[gc].replace([np.inf, -np.inf], np.nan).dropna()
        print(f"  {nm:10s} {vd.mean():+8.2f}±{vd.std():4.1f} {vg.mean():+7.2f}±{vg.std():4.1f}"
              f" {icc(dc):9.2f} {icc(gc):8.2f}")
    print("  (per-subject 4-run reliability ~ 4*ICC/(1+3*ICC))")

    print("\n=== directional regulation: crude Δmean vs HRF-aware β (group) ===")
    for nm in ["PDA", "CEN", "DMN"]:
        dcol, bcol = f"delta_{nm}", f"beta_{nm}"
        print(f"  {nm:4s}  Δmean={df[dcol].mean():+.3f} ({100*(df[dcol]>0).mean():.0f}% >0)"
              f"   β={df[bcol].mean():+.3f} ({100*(df[bcol]>0).mean():.0f}% >0)")

    print("\n=== non-redundancy with PDA level: corr(f-SNR, mean feedback PDA) ===")
    for nm in ["PDA", "CEN", "CENDMN"]:
        print(f"  {nm:6s}  discrete={corr(f'fsnr_{nm}_db','meanfb_PDA'):+.2f}"
              f"   glm={corr(f'glm_{nm}_db','meanfb_PDA'):+.2f}   (want |r|<0.8)")

    print("\n=== hypothesis test: f-SNR vs regulation direction (GLM β; across runs) ===")
    for nm in ["PDA", "CEN", "CENDMN"]:
        gc = f"glm_{nm}_db"
        print(f"  {gc:12s} vs βDMN r={corr(gc,'beta_DMN'):+.2f} (expect <0)"
              f"   vs βPDA r={corr(gc,'beta_PDA'):+.2f} (expect >0)")


if __name__ == "__main__":
    main()
