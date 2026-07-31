"""
within_rest_coupling.py
-----------------------
Assumption-light coupling test (no decoder): for one subject's rest runs,
cross-correlate each EEG channel's per-TR block-mean with the rest PDA across
EEG-leads-BOLD lags. Compares baseline (1-40Hz, desc-preproc500Hz) vs infraslow
(0.01-40Hz, desc-preproc500HzISp01). Answers "does infraslow couple to PDA at
all?" before committing to any model.

Usage: python within_rest_coupling.py --subject dmnelf007 --config config.yaml
"""
import argparse, warnings
from pathlib import Path
import numpy as np, yaml, mne
from scipy.stats import pearsonr
warnings.simplefilter("ignore"); mne.set_log_level("ERROR")

LAGS = range(0, 18)


def load_config(p):
    cfg = yaml.safe_load(open(p)); d = cfg["data"]
    d["features_dir"] = (d["features_dir_cluster"] if Path("/projects/swglab").exists()
                         else d["features_dir_local"])
    return cfg


def block_mean(raw, spt, n):
    x = raw.get_data(picks="eeg"); x = x[:, :n*spt].reshape(x.shape[0], n, spt).mean(2).T
    return (x - x.mean(0)) / (x.std(0) + 1e-8)


def gather_rest(cfg, subj, desc):
    fdir = Path(cfg["data"]["features_dir"]) / f"sub-{subj}"
    eroot = Path(cfg["data"]["eeg_preproc_dir"]); ses = cfg["data"]["session"]
    spt = cfg["data"]["eeg"]["samples_per_tr"]
    PDA, X, chs = [], [], None
    for npz in sorted(fdir.glob(f"sub-{subj}_task-rest_run-*_features.npz")):
        d = np.load(npz, allow_pickle=True); pda = np.asarray(d["pda"], float)
        run = npz.name.split("run-")[1][0]
        fif = eroot/f"sub-{subj}"/ses/"eeg"/f"sub-{subj}_{ses}_task-rest_run-{int(run):02d}_desc-{desc}_eeg.fif"
        if not fif.exists():
            print(f"  [skip rest run-{run}] no fif desc-{desc}"); continue
        raw = mne.io.read_raw_fif(str(fif), preload=True, verbose=False)
        PDA.append(pda); X.append(block_mean(raw, spt, len(pda))); chs = raw.ch_names[:X[-1].shape[1]]
    return PDA, X, chs


def coupling(PDA, X):
    out = {}
    for L in LAGS:
        rs = []
        for c in range(X[0].shape[1]):
            xs, ys = [], []
            for Xr, pda in zip(X, PDA):
                if L > 0: xs.append(Xr[:-L, c]); ys.append(pda[L:])
                else:     xs.append(Xr[:, c]);   ys.append(pda)
            rs.append(pearsonr(np.concatenate(xs), np.concatenate(ys))[0])
        out[L] = np.array(rs)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subject", required=True); ap.add_argument("--config", default="config.yaml")
    a = ap.parse_args(); cfg = load_config(a.config)
    descs = {"baseline": cfg["data"]["eeg"]["desc"],
             "infraslow": cfg["data"]["eeg"]["desc_infraslow"]}
    for name, desc in descs.items():
        PDA, X, chs = gather_rest(cfg, a.subject, desc)
        if not X:
            print(f"[{name}] no data"); continue
        N = sum(len(p) for p in PDA); thr = 2/np.sqrt(N)
        out = coupling(PDA, X)
        bl = max(out, key=lambda L: np.max(np.abs(out[L]))); rs = out[bl]; ci = int(np.argmax(np.abs(rs)))
        print(f"[{name:9s}] n={N} |r|>{thr:.3f}~p<.05  best lag={bl}TR ({bl*cfg['data']['fmri']['tr']:.1f}s) "
              f"ch={chs[ci]} r={rs[ci]:+.3f}  nsig={int(np.sum(np.abs(rs)>thr))}/{len(chs)}")
        print("   max|r| by lag: " + " ".join(f"L{L}:{np.max(np.abs(out[L])):.3f}" for L in LAGS))


if __name__ == "__main__":
    main()
