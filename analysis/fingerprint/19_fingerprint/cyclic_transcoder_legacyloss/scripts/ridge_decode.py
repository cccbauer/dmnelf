"""
ridge_decode.py
---------------
Within-subject PDA decoding with Ridge (replaces the over-regularizing ElasticNet
that collapsed to a constant). Uses the infraslow features and an EEG-leads-BOLD
lag selected by within-rest leave-one-run-out CV.

Reports, for each feature set:
  - best lag (chosen by within-rest LORO CV)
  - within-rest CV r  (train rest run A -> predict rest run B, averaged both ways)
  - rest -> feedback r (train all rest -> predict all feedback) at that lag

Feature sets: baseline (1-40Hz block-means from npz), infraslow (0.01-40Hz from
desc-preproc500HzISp01 fif, all channels), infraslow-Cz (single best coupling ch).

Usage:
  python ridge_decode.py --subject dmnelf007 --config config.yaml
"""
import argparse, warnings
from pathlib import Path
import numpy as np, yaml, mne
from scipy.stats import pearsonr
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler
warnings.simplefilter("ignore"); mne.set_log_level("ERROR")

ALPHAS = np.logspace(-2, 6, 40)
LAGS = range(0, 16)


def load_config(path):
    cfg = yaml.safe_load(open(path)); d = cfg["data"]
    d["features_dir"] = (d["features_dir_cluster"] if Path("/projects/swglab").exists()
                         else d["features_dir_local"])
    return cfg


def block_mean(raw, spt, n):
    x = raw.get_data(picks="eeg"); x = x[:, :n*spt].reshape(x.shape[0], n, spt).mean(2).T
    return ((x - x.mean(0)) / (x.std(0) + 1e-8)).astype(np.float32)


def gather(cfg, subject, task, desc_is):
    fdir = Path(cfg["data"]["features_dir"]) / f"sub-{subject}"
    eroot = Path(cfg["data"]["eeg_preproc_dir"]); ses = cfg["data"]["session"]
    spt = cfg["data"]["eeg"]["samples_per_tr"]; runs = []; chs = None
    for npz in sorted(fdir.glob(f"sub-{subject}_task-{task}_run-*_features.npz")):
        d = np.load(npz, allow_pickle=True); pda = np.asarray(d["pda"], float); n = len(pda)
        run = npz.name.split("run-")[1][0]
        fif = eroot/f"sub-{subject}"/ses/"eeg"/f"sub-{subject}_{ses}_task-{task}_run-{int(run):02d}_desc-{desc_is}_eeg.fif"
        if not fif.exists():
            print(f"  [skip {task} run-{run}] no fif"); continue
        raw = mne.io.read_raw_fif(str(fif), preload=True, verbose=False)
        runs.append((pda, np.asarray(d["eeg_block"], float), block_mean(raw, spt, n)))
        chs = raw.ch_names[:runs[-1][1].shape[1]]
    return runs, chs


def lagged(X, y, lag):
    return (X[:-lag], y[lag:]) if lag > 0 else (X, y)


def fit_predict(Xtr, ytr, Xte):
    sc = StandardScaler().fit(Xtr)
    m = RidgeCV(alphas=ALPHAS).fit(sc.transform(Xtr), ytr)
    return m.predict(sc.transform(Xte))


def cols(runs, idx, sel):
    """idx: 1=baseline,2=infraslow; sel: channel indices or None=all."""
    return [(r[0], r[idx] if sel is None else r[idx][:, sel]) for r in runs]


def within_rest_cv(rest, lag):
    """LORO across rest runs; returns Fisher-averaged Pearson r."""
    rs = []
    for i in range(len(rest)):
        tr = [rest[j] for j in range(len(rest)) if j != i]; te = rest[i]
        Xtr = np.vstack([lagged(p[1], p[0], lag)[0] for p in tr])
        ytr = np.concatenate([lagged(p[1], p[0], lag)[1] for p in tr])
        Xte, yte = lagged(te[1], te[0], lag)
        pred = fit_predict(Xtr, ytr, Xte)
        if pred.std() > 0:
            rs.append(np.arctanh(np.clip(pearsonr(pred, yte)[0], -.999, .999)))
    return np.tanh(np.mean(rs)) if rs else np.nan


def rest_to_feedback(rest, feed, lag):
    Xtr = np.vstack([lagged(p[1], p[0], lag)[0] for p in rest])
    ytr = np.concatenate([lagged(p[1], p[0], lag)[1] for p in rest])
    Xte = np.vstack([lagged(p[1], p[0], lag)[0] for p in feed])
    yte = np.concatenate([lagged(p[1], p[0], lag)[1] for p in feed])
    pred = fit_predict(Xtr, ytr, Xte)
    r, p = pearsonr(pred, yte)
    return r, p, float(pred.std()), float(yte.std()), len(yte)


def run_set(name, rest, feed):
    # pick lag by within-rest CV
    cvr = {L: within_rest_cv(rest, L) for L in LAGS}
    best = max(cvr, key=lambda L: (cvr[L] if not np.isnan(cvr[L]) else -9))
    r, p, ps, ts, n = rest_to_feedback(rest, feed, best)
    print(f"[{name:16s}] best_lag={best:2d}TR  within-rest CV r={cvr[best]:+.3f}  "
          f"| rest->feedback r={r:+.3f} (p={p:.1e}) pred_std={ps:.3f} n={n}")
    print(f"   {'':16s} CV r by lag: " + " ".join(f"L{L}:{cvr[L]:+.2f}" for L in LAGS))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subject", required=True)
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--desc-is", default="preproc500HzISp01")
    a = ap.parse_args()
    cfg = load_config(a.config)
    rest, chs = gather(cfg, a.subject, "rest", a.desc_is)
    feed, _ = gather(cfg, a.subject, "feedback", a.desc_is)
    print(f"=== {a.subject} Ridge decode  (rest runs={len(rest)}, feedback runs={len(feed)}) ===")
    if not rest or not feed:
        print("insufficient data"); return
    cz = chs.index("Cz") if "Cz" in chs else int(np.argmax([0]))
    run_set("baseline_all",   cols(rest,1,None),  cols(feed,1,None))
    run_set("infraslow_all",  cols(rest,2,None),  cols(feed,2,None))
    run_set("infraslow_Cz",   cols(rest,2,[cz]),  cols(feed,2,[cz]))


if __name__ == "__main__":
    main()
