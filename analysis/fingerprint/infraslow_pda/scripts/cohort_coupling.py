"""
cohort_coupling.py
------------------
Cohort-wide WITHIN-REST infraslow->PDA coupling (no decoder, no cross-task gap).
For each subject, cross-correlate every EEG channel's per-TR block-mean with the
rest PDA across EEG-leads-BOLD lags, and take the best |r| over channels x lags.

Significance via a MAX-STATISTIC circular-shift null: circularly shift the rest
PDA (per run, preserving autocorrelation), recompute the max|r| over ALL
channels x lags, repeat; p = P(null max >= observed max). This properly accounts
for selecting the best channel/lag AND for autocorrelation.

Compares infraslow (0.01-40Hz) vs baseline (1-40Hz). If infraslow couples within
rest for many subjects, the cohort decode failure is a rest->feedback transfer
problem, not the feature.

Usage: python cohort_coupling.py --config config.yaml [--nperm 1000]
"""
import argparse, warnings, csv
from pathlib import Path
import numpy as np, yaml, mne
warnings.simplefilter("ignore"); mne.set_log_level("ERROR")

LAGS = list(range(0, 18))


def load_config(p):
    cfg = yaml.safe_load(open(p)); d = cfg["data"]
    d["features_dir"] = (d["features_dir_cluster"] if Path("/projects/swglab").exists()
                         else d["features_dir_local"])
    return cfg


def block_mean(raw, spt, n):
    x = raw.get_data(picks="eeg"); x = x[:, :n*spt].reshape(x.shape[0], n, spt).mean(2).T
    return ((x - x.mean(0)) / (x.std(0) + 1e-8)).astype(np.float64)


def gather_rest(cfg, subj, desc):
    fdir = Path(cfg["data"]["features_dir"]) / f"sub-{subj}"
    eroot = Path(cfg["data"]["eeg_preproc_dir"]); ses = cfg["data"]["session"]
    spt = cfg["data"]["eeg"]["samples_per_tr"]; runs = []; chs = None
    for npz in sorted(fdir.glob(f"sub-{subj}_task-rest_run-*_features.npz")):
        d = np.load(npz, allow_pickle=True); pda = np.asarray(d["pda"], float)
        run = npz.name.split("run-")[1][0]
        fif = eroot/f"sub-{subj}"/ses/"eeg"/f"sub-{subj}_{ses}_task-rest_run-{int(run):02d}_desc-{desc}_eeg.fif"
        if not fif.exists():
            continue
        raw = mne.io.read_raw_fif(str(fif), preload=True, verbose=False)
        runs.append((pda, block_mean(raw, spt, len(pda)))); chs = raw.ch_names[:runs[-1][1].shape[1]]
    return runs, chs


def zc(a, axis=0):
    return (a - a.mean(axis, keepdims=True)) / (a.std(axis, keepdims=True) + 1e-12)


def max_stat(runs, pda_shifted):
    """max |corr| over channels x lags, given (possibly shifted) per-run pda."""
    best = 0.0; best_ch = -1; best_lag = -1
    for L in LAGS:
        Xs, ys = [], []
        for (pda, X), ps in zip(runs, pda_shifted):
            if L > 0: Xs.append(X[:-L]); ys.append(ps[L:])
            else:     Xs.append(X);     ys.append(ps)
        Xcat = np.vstack(Xs); ycat = np.concatenate(ys)
        r = (zc(Xcat).T @ zc(ycat)) / len(ycat)     # (nch,)
        i = int(np.argmax(np.abs(r)))
        if abs(r[i]) > abs(best):
            best, best_ch, best_lag = r[i], i, L
    return best, best_ch, best_lag


def coupling_with_null(runs, nperm, rng):
    pda0 = [p for p, _ in runs]
    obs, ci, li = max_stat(runs, pda0)
    n_ge = 0
    for _ in range(nperm):
        shifted = []
        for pda, _ in runs:
            n = len(pda); k = int(rng.integers(15, n - 15)) if n > 40 else 1
            shifted.append(np.roll(pda, k))
        m, _, _ = max_stat(runs, shifted)
        if abs(m) >= abs(obs): n_ge += 1
    return obs, ci, li, (n_ge + 1) / (nperm + 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--nperm", type=int, default=1000)
    ap.add_argument("--out", default="results/cohort_coupling.csv")
    a = ap.parse_args(); cfg = load_config(a.config); tr = cfg["data"]["fmri"]["tr"]
    subs = [s for s in cfg["data"]["subjects"]["all"]
            if s not in set(cfg["data"]["subjects"].get("exclude", []))]
    di = cfg["data"]["eeg"]["desc_infraslow"]; db = cfg["data"]["eeg"]["desc"]
    rng = np.random.default_rng(0)

    print(f"{'subject':11s} | {'INFRASLOW best|r| ch lag    p(maxstat)':38s} | {'BASELINE best|r|   p':22s}")
    print("-" * 84)
    rows = []
    for s in subs:
        ri, chs = gather_rest(cfg, s, di); rb, _ = gather_rest(cfg, s, db)
        if not ri:
            print(f"{s:11s} | no infraslow rest"); continue
        oi, ci, li, pi = coupling_with_null(ri, a.nperm, np.random.default_rng(0))
        if rb:
            ob, cb, lb, pb = coupling_with_null(rb, a.nperm, np.random.default_rng(0))
        else:
            ob, pb = np.nan, np.nan
        print(f"{s:11s} | r={oi:+.3f} {chs[ci]:>4s} L{li:<2d}({li*tr:4.1f}s) p={pi:.3f}      "
              f"| r={ob:+.3f} p={pb:.3f}")
        rows.append(dict(subject=s, is_r=oi, is_ch=chs[ci], is_lag=li, is_p=pi,
                         base_r=ob, base_p=pb))
    if rows:
        nis = sum(1 for r in rows if r['is_p'] < 0.05)
        nb = sum(1 for r in rows if not np.isnan(r['base_p']) and r['base_p'] < 0.05)
        print("-" * 84)
        print(f"INFRASLOW within-rest coupling significant (maxstat p<.05): {nis}/{len(rows)}  "
              f"| BASELINE: {nb}/{len(rows)}")
        outp = Path(cfg["project"]["base_dir"]) / a.out
        outp.parent.mkdir(parents=True, exist_ok=True)
        with open(outp, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
        print(f"saved: {outp}")


if __name__ == "__main__":
    main()
