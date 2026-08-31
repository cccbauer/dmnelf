"""
refine_007.py
-------------
Pressure-test the within-subject infraslow->PDA decode for one subject:
  1. circular-shift permutation null (preserves autocorrelation) for honest p
     -- the parametric pearson p is anti-conservative for slow autocorrelated signals.
  2. per-channel rest->feedback decode topography (where is the signal?)
  3. HRF-convolved infraslow vs the integer-TR-lag shift (is the ~13s lag really
     a hemodynamic response, or longer-than-HRF infraslow coupling?)

Lag is selected on REST (within-run split) and applied to FEEDBACK; the null is
computed on the held-out feedback prediction, so significance is not circular.

Usage: python refine_007.py --subject dmnelf007 --config config.yaml
"""
import argparse, warnings
from pathlib import Path
import numpy as np, yaml, mne
from scipy.stats import pearsonr, gamma
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler
warnings.simplefilter("ignore"); mne.set_log_level("ERROR")

ALPHAS = np.logspace(-2, 6, 40); LAGS = range(0, 16)


def load_config(p):
    cfg = yaml.safe_load(open(p)); d = cfg["data"]
    d["features_dir"] = (d["features_dir_cluster"] if Path("/projects/swglab").exists()
                         else d["features_dir_local"])
    return cfg


def block_mean(raw, spt, n):
    x = raw.get_data(picks="eeg"); x = x[:, :n*spt].reshape(x.shape[0], n, spt).mean(2).T
    return ((x - x.mean(0)) / (x.std(0) + 1e-8)).astype(np.float32)


def gather(cfg, subj, task, desc):
    fdir = Path(cfg["data"]["features_dir"]) / f"sub-{subj}"
    eroot = Path(cfg["data"]["eeg_preproc_dir"]); ses = cfg["data"]["session"]
    spt = cfg["data"]["eeg"]["samples_per_tr"]; runs = []; chs = None
    for npz in sorted(fdir.glob(f"sub-{subj}_task-{task}_run-*_features.npz")):
        d = np.load(npz, allow_pickle=True); pda = np.asarray(d["pda"], float)
        run = npz.name.split("run-")[1][0]
        fif = eroot/f"sub-{subj}"/ses/"eeg"/f"sub-{subj}_{ses}_task-{task}_run-{int(run):02d}_desc-{desc}_eeg.fif"
        if not fif.exists(): continue
        raw = mne.io.read_raw_fif(str(fif), preload=True, verbose=False)
        runs.append((pda, block_mean(raw, spt, len(pda)))); chs = raw.ch_names[:runs[-1][1].shape[1]]
    return runs, chs


def lag(X, y, L): return (X[:-L], y[L:]) if L > 0 else (X, y)


def fit_pred(Xtr, ytr, Xte):
    sc = StandardScaler().fit(Xtr)
    m = RidgeCV(alphas=ALPHAS).fit(sc.transform(Xtr), ytr)
    return m.predict(sc.transform(Xte))


def stack(runs, sel, L):
    Xs, ys = [], []
    for pda, X in runs:
        Xx = X if sel is None else X[:, sel]
        a, b = lag(Xx, pda, L); Xs.append(a); ys.append(b)
    return np.vstack(Xs), np.concatenate(ys)


def best_lag_rest(rest, sel):
    """select lag by within-rest leave-one-run-out CV (Fisher-avg r)."""
    best, bestr = 0, -9
    for L in LAGS:
        rs = []
        for i in range(len(rest)):
            tr = [rest[j] for j in range(len(rest)) if j != i]
            Xtr, ytr = stack(tr, sel, L); Xte, yte = stack([rest[i]], sel, L)
            pred = fit_pred(Xtr, ytr, Xte)
            if pred.std() > 0: rs.append(np.arctanh(np.clip(pearsonr(pred, yte)[0], -.999, .999)))
        r = np.tanh(np.mean(rs)) if rs else np.nan
        if not np.isnan(r) and r > bestr: bestr, best = r, L
    return best, bestr


def circshift_null(pred, true, nperm=5000, min_shift=15):
    robs = pearsonr(pred, true)[0]; n = len(true); rng = np.random.default_rng(0)
    cnt = 0
    for _ in range(nperm):
        k = int(rng.integers(min_shift, n - min_shift))
        if abs(pearsonr(pred, np.roll(true, k))[0]) >= abs(robs): cnt += 1
    return robs, (cnt + 1) / (nperm + 1)


def hrf_kernel(tr, length=40):
    t = np.arange(0, length, tr)
    h = gamma.pdf(t, 6) - 0.35 * gamma.pdf(t, 16)
    return h / h.sum()


def hrf_conv(runs, sel, tr):
    h = hrf_kernel(tr); out = []
    for pda, X in runs:
        Xx = X if sel is None else X[:, sel]
        Xc = np.column_stack([np.convolve(Xx[:, c], h)[:len(Xx)] for c in range(Xx.shape[1])])
        out.append((pda, Xc))
    return out


def evaluate(tag, rest, feed, sel):
    L, cvr = best_lag_rest(rest, sel)
    Xtr, ytr = stack(rest, sel, L); Xte, yte = stack(feed, sel, L)
    pred = fit_pred(Xtr, ytr, Xte)
    robs, pnull = circshift_null(pred, yte)
    print(f"[{tag:18s}] lag={L:2d}TR restCV r={cvr:+.3f} | feedback r={robs:+.3f} "
          f"circshift_p={pnull:.4f} (parametric_p={pearsonr(pred,yte)[1]:.1e})")
    return robs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subject", required=True); ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--desc-is", default="preproc500HzISp01")
    a = ap.parse_args(); cfg = load_config(a.config); tr = cfg["data"]["fmri"]["tr"]
    rest, chs = gather(cfg, a.subject, "rest", a.desc_is)
    feed, _ = gather(cfg, a.subject, "feedback", a.desc_is)
    cz = chs.index("Cz")
    print(f"=== {a.subject} refine  (rest={len(rest)} feedback={len(feed)} runs) ===")
    print("\n-- honest significance (circular-shift null preserves autocorrelation) --")
    evaluate("infraslow_all", rest, feed, None)
    evaluate("infraslow_Cz", rest, feed, [cz])
    # baseline control (1-40Hz) at Cz, same machinery
    base_rest, _ = gather(cfg, a.subject, "rest", cfg["data"]["eeg"].get("desc", "preproc500Hz"))
    base_feed, _ = gather(cfg, a.subject, "feedback", cfg["data"]["eeg"].get("desc", "preproc500Hz"))
    if base_rest and base_feed:
        evaluate("baseline_Cz(ctrl)", base_rest, base_feed, [cz])

    print("\n-- HRF-convolved infraslow (does a canonical HRF explain the ~13s lag?) --")
    rest_h = hrf_conv(rest, [cz], tr); feed_h = hrf_conv(feed, [cz], tr)
    evaluate("infraslow_Cz_HRF", rest_h, feed_h, None)
    rest_ha = hrf_conv(rest, None, tr); feed_ha = hrf_conv(feed, None, tr)
    evaluate("infraslow_all_HRF", rest_ha, feed_ha, None)

    print("\n-- per-channel feedback decode topography (lag from each channel's rest CV) --")
    rows = []
    for c in range(len(chs)):
        L, _ = best_lag_rest(rest, [c])
        Xtr, ytr = stack(rest, [c], L); Xte, yte = stack(feed, [c], L)
        pred = fit_pred(Xtr, ytr, Xte)
        rows.append((chs[c], pearsonr(pred, yte)[0], L))
    for ch, r, L in sorted(rows, key=lambda z: -abs(z[1]))[:10]:
        print(f"   {ch:5s} r={r:+.3f} lag={L}TR")


if __name__ == "__main__":
    main()
