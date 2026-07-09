#!/usr/bin/env python3
"""
eeg_fsnr_bandpower.py  —  Stream B, Flavor 1
--------------------------------------------
A PURE EEG f-SNR from band power, matched to the fMRI PDA and fMRI running f-SNR.

EEG features per channel c x band b (HRF-convolved band power from the eeg_bold_coupling
cache, so already in BOLD-time):
  raw   : the band power itself (the coupling baseline)
  fsnr  : running EEG f-SNR = trailing_mean/trailing_std  (the f-SNR analog of the fMRI one)

Matching (leak-free, nested single-feature selection over the 31x5 sites):
  select the site maximizing |corr| with the target on inner-train folds, score the
  concatenated out-of-fold correlation. Targets: fMRI PDA(t) and fMRI running f-SNR(t).
Also: does EEG band-power VARIANCE quench rest->feedback (cross-modal analog of the BOLD
result)? And an a-priori construct index (posterior-alpha f-SNR), no site fitting.
"""
from pathlib import Path
import numpy as np, glob, re
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "fsnr" / "scripts"))
from fsnr_proxy import running_fsnr, W as PW
from fsnr_fmri import BASELINE_TR, HRF_DROP

PROJ = Path(__file__).resolve().parent.parent
DATA = PROJ / "data"; RES = PROJ / "results"; RES.mkdir(exist_ok=True)
BANDS = ["delta", "theta", "alpha", "beta", "gamma"]
POST = ["P3", "P4", "P7", "P8", "O1", "O2", "Oz", "Pz", "POz", "PO3", "PO4"]  # posterior
FRONTAL = ["Fp1", "Fp2", "F3", "F4", "F7", "F8", "Fz", "FC1", "FC2", "FC5", "FC6"]  # portable headset


def cols_for(feat, chset):
    """Indices of feature columns whose channel is in chset."""
    return [j for j, (b, c) in enumerate(feat) if c in chset]


def apriori_match(runs, chs, band, chset):
    """Construct-driven (no fitting): running f-SNR of `band` averaged over `chset`,
    correlated with PDA per run (mean over runs)."""
    pi = [i for i, c in enumerate(chs) if c in chset]
    if not pi:
        return np.nan
    rs = []
    for rd in runs:
        pda = zs(np.asarray(rd["targets"]["PDA"], float))
        ef = zs(np.nanmean(np.column_stack([running_fsnr(rd["bp"][band][:, i])[1] for i in pi]), 1))
        m = np.isfinite(pda) & np.isfinite(ef)
        if m.sum() > 20:
            rs.append(np.corrcoef(pda[m], ef[m])[0, 1])
    return np.nanmean(rs) if rs else np.nan


def zs(x):
    return (x - np.nanmean(x)) / (np.nanstd(x) + 1e-12)


def subject_runs(f):
    z = np.load(f, allow_pickle=True)
    chs = [str(c) for c in z["ch_names"]]
    return list(z["runs_data"]), chs


def build_features(runs, chs):
    """Per run -> (Xraw, Xfsnr [T, 31*5], ypda, yfsnr). Concatenate across runs."""
    Xr, Xf, ypda, yfsnr, feat = [], [], [], [], None
    for rd in runs:
        n = rd["n_tr"]
        pda = np.asarray(rd["targets"]["PDA"], float)
        fs = running_fsnr(pda)[1]                       # fMRI running f-SNR (dB)
        raw = np.column_stack([rd["bp"][b][:, ci] for b in BANDS for ci in range(len(chs))])
        fsnr = np.column_stack([running_fsnr(rd["bp"][b][:, ci])[1]
                                for b in BANDS for ci in range(len(chs))])
        Xr.append(raw); Xf.append(fsnr); ypda.append(zs(pda)); yfsnr.append(zs(fs))
        if feat is None:
            feat = [(b, chs[ci]) for b in BANDS for ci in range(len(chs))]
    return (np.vstack(Xr), np.vstack(Xf), np.concatenate(ypda), np.concatenate(yfsnr), feat)


def folds(n, k=5):
    e = np.linspace(0, n, k + 1).astype(int)
    return [(np.r_[0:e[i], e[i+1]:n], np.arange(e[i], e[i+1])) for i in range(k)]


def match_nested(X, y, k=5):
    """Leak-free single-site match: select best site on inner-train, OOF-correlate."""
    ok = np.isfinite(y) & np.all(np.isfinite(X), 1)
    X, y = X[ok], y[ok]
    if len(y) < 30:
        return np.nan, None
    pred = np.full(len(y), np.nan); chosen = []
    for tr, te in folds(len(y), k):
        c = np.array([abs(np.corrcoef(X[tr, j], y[tr])[0, 1]) for j in range(X.shape[1])])
        j = np.nanargmax(c)
        s = np.sign(np.corrcoef(X[tr, j], y[tr])[0, 1])
        pred[te] = s * zs(X[te, j]); chosen.append(j)
    return float(np.corrcoef(y, pred)[0, 1]), chosen


def main():
    files = sorted(glob.glob(str(DATA / "*_bandpower.npz")))
    rows = []
    apriori = {"post_alpha": [], "frontal_alpha": [], "frontal_theta": []}
    for f in files:
        sub = re.search(r"(dmnelf\w+)_bandpower", f).group(1)
        runs, chs = subject_runs(f)
        Xr, Xf, ypda, yfsnr, feat = build_features(runs, chs)
        fc = cols_for(feat, FRONTAL); pc = cols_for(feat, POST)
        r = dict(subject=sub)
        r["raw_vs_PDA_all"], _ = match_nested(Xr, ypda)
        r["raw_vs_PDA_frontal"], _ = match_nested(Xr[:, fc], ypda)
        r["raw_vs_PDA_post"], _ = match_nested(Xr[:, pc], ypda)
        r["fsnr_vs_PDA_all"], _ = match_nested(Xf, ypda)
        r["fsnr_vs_PDA_frontal"], _ = match_nested(Xf[:, fc], ypda)
        r["fsnr_vs_fMRIfsnr_all"], _ = match_nested(Xf, yfsnr)
        rows.append(r)
        # NOTE: EEG variability-quench is NOT computed here — the cached band power is
        # HRF-convolved (~27 s onset ramp contaminates the 25-TR baseline); done in Flavor 2.
        # a-priori construct indices (no fitting): band f-SNR over a montage, matched to PDA
        apriori["post_alpha"].append(apriori_match(runs, chs, "alpha", POST))
        apriori["frontal_alpha"].append(apriori_match(runs, chs, "alpha", FRONTAL))
        apriori["frontal_theta"].append(apriori_match(runs, chs, "theta", FRONTAL))

    import pandas as pd
    df = pd.DataFrame(rows); df.to_csv(RES / "eeg_fsnr_bandpower_match.csv", index=False)
    print(f"{len(df)} subjects (leak-free nested single-site match, causal window {PW} TR)\n")

    def sflip(x, n=10000):
        x = np.asarray([v for v in x if np.isfinite(v)]); rng = np.random.default_rng(0)
        obs = x.mean(); null = (rng.choice([-1, 1], (n, len(x)))*np.abs(x)).mean(1)
        return obs, (np.sum(null >= obs)+1)/(n+1)

    print("=== leak-free nested single-site match to PDA — by montage ===")
    for c in ["raw_vs_PDA_all", "raw_vs_PDA_frontal", "raw_vs_PDA_post",
              "fsnr_vs_PDA_all", "fsnr_vs_PDA_frontal", "fsnr_vs_fMRIfsnr_all"]:
        o, p = sflip(df[c].values)
        print(f"  {c:22s} r={o:+.3f}  p={p:.4f}")
    print("\n=== a-priori construct indices (no fitting) vs PDA ===")
    for k in ["post_alpha", "frontal_alpha", "frontal_theta"]:
        o, p = sflip(np.array(apriori[k]))
        print(f"  {k:16s} r={o:+.3f}  p={p:.4f}")
    print("\n(EEG variability-quench deferred to Flavor 2: cached band power is HRF-convolved.)")


if __name__ == "__main__":
    main()
