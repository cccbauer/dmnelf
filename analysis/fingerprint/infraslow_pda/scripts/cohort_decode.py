"""
cohort_decode.py
----------------
Run the within-subject infraslow->PDA Ridge decode across all subjects in the
config and tabulate. For each subject: train on OWN rest, predict OWN feedback
PDA; report Pearson r with a circular-shift permutation p (preserves
autocorrelation). Compares infraslow (all ch + Cz) vs baseline 1-40Hz (control).

Lag is selected on rest (leave-one-run-out CV if >=2 rest runs, else within-run
70/30 split) and applied to the held-out feedback runs, so significance is not
circular.

Usage: python cohort_decode.py --config config.yaml [--out results/cohort_decode.csv]
"""
import argparse, warnings, csv
from pathlib import Path
import numpy as np, yaml, mne
from scipy.stats import pearsonr
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
        if not fif.exists():
            continue
        raw = mne.io.read_raw_fif(str(fif), preload=True, verbose=False)
        runs.append((pda, block_mean(raw, spt, len(pda)))); chs = raw.ch_names[:runs[-1][1].shape[1]]
    return runs, chs


def lag(X, y, L): return (X[:-L], y[L:]) if L > 0 else (X, y)


def stack(runs, sel, L):
    Xs, ys = [], []
    for pda, X in runs:
        Xx = X if sel is None else X[:, sel]
        a, b = lag(Xx, pda, L); Xs.append(a); ys.append(b)
    return np.vstack(Xs), np.concatenate(ys)


def fit_pred(Xtr, ytr, Xte):
    sc = StandardScaler().fit(Xtr)
    m = RidgeCV(alphas=ALPHAS).fit(sc.transform(Xtr), ytr)
    return m.predict(sc.transform(Xte))


def cv_r(rest, sel, L):
    """within-rest CV r at lag L: LORO if >=2 runs else within-run 70/30."""
    rs = []
    if len(rest) >= 2:
        for i in range(len(rest)):
            tr = [rest[j] for j in range(len(rest)) if j != i]
            Xtr, ytr = stack(tr, sel, L); Xte, yte = stack([rest[i]], sel, L)
            pred = fit_pred(Xtr, ytr, Xte)
            if pred.std() > 0: rs.append(pearsonr(pred, yte)[0])
    else:
        pda, X = rest[0]; Xx = X if sel is None else X[:, sel]
        a, b = lag(Xx, pda, L); k = int(len(b) * 0.7)
        if k > 10 and len(b) - k > 10:
            pred = fit_pred(a[:k], b[:k], a[k:])
            if pred.std() > 0: rs.append(pearsonr(pred, b[k:])[0])
    return np.nanmean(rs) if rs else np.nan


def best_lag(rest, sel):
    best, br = 0, -9
    for L in LAGS:
        r = cv_r(rest, sel, L)
        if not np.isnan(r) and r > br: br, best = r, L
    return best, br


def circshift_null(pred, true, nperm=2000, min_shift=15):
    robs = pearsonr(pred, true)[0]; n = len(true); rng = np.random.default_rng(0); cnt = 0
    for _ in range(nperm):
        k = int(rng.integers(min_shift, n - min_shift))
        if abs(pearsonr(pred, np.roll(true, k))[0]) >= abs(robs): cnt += 1
    return robs, (cnt + 1) / (nperm + 1)


def decode(rest, feed, sel):
    L, cvr = best_lag(rest, sel)
    Xtr, ytr = stack(rest, sel, L); Xte, yte = stack(feed, sel, L)
    pred = fit_pred(Xtr, ytr, Xte)
    r, p = circshift_null(pred, yte)
    return dict(lag=L, cv_r=cvr, r=r, p=p, n=len(yte))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--out", default="results/cohort_decode.csv")
    a = ap.parse_args(); cfg = load_config(a.config)
    subs = [s for s in cfg["data"]["subjects"]["all"]
            if s not in set(cfg["data"]["subjects"].get("exclude", []))]
    di = cfg["data"]["eeg"]["desc_infraslow"]; db = cfg["data"]["eeg"]["desc"]

    print(f"{'subject':11s} {'IS_all r':>9s} {'p':>7s} {'lag':>4s} | "
          f"{'IS_Cz r':>8s} {'p':>7s} {'lag':>4s} | {'base r':>7s} {'p':>7s}")
    print("-" * 78)
    rows = []
    for s in subs:
        rest_i, chs = gather(cfg, s, "rest", di); feed_i, _ = gather(cfg, s, "feedback", di)
        rest_b, _ = gather(cfg, s, "rest", db); feed_b, _ = gather(cfg, s, "feedback", db)
        if not rest_i or not feed_i:
            print(f"{s:11s}  [no infraslow data]"); continue
        cz = chs.index("Cz") if chs and "Cz" in chs else 0
        ia = decode(rest_i, feed_i, None)
        iz = decode(rest_i, feed_i, [cz])
        bb = decode(rest_b, feed_b, None) if rest_b and feed_b else dict(r=np.nan, p=np.nan, lag=-1, cv_r=np.nan, n=0)
        print(f"{s:11s} {ia['r']:+9.3f} {ia['p']:7.4f} {ia['lag']:4d} | "
              f"{iz['r']:+8.3f} {iz['p']:7.4f} {iz['lag']:4d} | {bb['r']:+7.3f} {bb['p']:7.4f}")
        rows.append(dict(subject=s,
                         is_all_r=ia['r'], is_all_p=ia['p'], is_all_lag=ia['lag'], is_all_cvr=ia['cv_r'],
                         is_cz_r=iz['r'], is_cz_p=iz['p'], is_cz_lag=iz['lag'], is_cz_cvr=iz['cv_r'],
                         base_all_r=bb['r'], base_all_p=bb['p'], n=ia['n']))

    if rows:
        ia = np.array([r['is_all_r'] for r in rows]); bb = np.array([r['base_all_r'] for r in rows])
        nsig = sum(1 for r in rows if r['is_all_p'] < 0.05 and r['is_all_r'] > 0)
        print("-" * 78)
        print(f"infraslow_all: mean r={np.nanmean(ia):+.3f}  significant(p<.05,r>0)={nsig}/{len(rows)}  "
              f"| baseline mean r={np.nanmean(bb):+.3f}")
        outp = Path(cfg["project"]["base_dir"]) / a.out if "/projects/swglab" in str(Path(cfg["project"]["base_dir"])) else Path(a.out)
        outp.parent.mkdir(parents=True, exist_ok=True)
        with open(outp, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
        print(f"saved: {outp}")


if __name__ == "__main__":
    main()
