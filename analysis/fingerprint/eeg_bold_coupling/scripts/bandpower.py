"""
bandpower.py
------------
Shared feature module: per-channel band-limited power (BLP) per TR, HRF-convolved.

For a preprocessed EEG run we band-pass filter, take the Hilbert envelope, average
the envelope over the 600 samples of each 1.2 s TR, log-transform, then convolve
each channel's per-TR power timeseries with a canonical (double-gamma) HRF to model
the ~5-6 s neurovascular lag. Output: [N_TR x n_ch] per band.

This is the standard EEG-fMRI coupling feature (Goldman 2002, Laufs 2003,
Mantini 2007), replacing the per-TR raw-amplitude block-mean that averaged
oscillations to ~0 and kept only infraslow drift.
"""
from pathlib import Path
import numpy as np, yaml, mne
from scipy.stats import gamma
from scipy.signal import hilbert
mne.set_log_level("ERROR")


def load_config(p):
    cfg = yaml.safe_load(open(p)); d = cfg["data"]
    d["features_dir"] = (d["features_dir_cluster"] if Path("/projects/swglab").exists()
                         else d["features_dir_local"])
    return cfg


def canonical_hrf(tr, length_s=32, delay=6, undershoot=16):
    """SPM-style double-gamma HRF sampled at the TR. Sums to 1."""
    t = np.arange(0, length_s, tr)
    h = gamma.pdf(t, delay) - gamma.pdf(t, undershoot) / 6.0
    return h / h.sum()


def hrf_convolve(x, hrf):
    """Causal convolution; truncated to len(x)."""
    return np.convolve(x, hrf, mode="full")[:len(x)]


def band_power_run(raw, bands, spt, n_tr, hrf):
    """Return dict band -> [n_tr x n_ch] HRF-convolved log power, and ch_names."""
    picks = mne.pick_types(raw.info, eeg=True, exclude=[])
    chs = [raw.ch_names[i] for i in picks]
    out = {}
    for name, (lo, hi) in bands.items():
        r = raw.copy().filter(lo, hi, picks=picks, verbose=False)
        env = np.abs(hilbert(r.get_data(picks=picks), axis=-1))  # |analytic signal|
        # per-TR mean of the envelope -> log power
        env = env[:, :n_tr * spt].reshape(env.shape[0], n_tr, spt).mean(2)  # (ch, n_tr)
        lp = np.log(env.T + 1e-12)                                          # (n_tr, ch)
        lp = np.column_stack([hrf_convolve(lp[:, c], hrf) for c in range(lp.shape[1])])
        out[name] = lp
    return out, chs


def zscore(x):
    return (x - x.mean(0)) / (x.std(0) + 1e-12)


def gather_subject(cfg, subj, hrf):
    """Per feedback run: (targets dict, bandpower dict, parcels[n_tr x 64], chs).
    targets: DMN, CEN, PDA timeseries (raw, will be handled by caller)."""
    d = cfg["data"]; ses = d["session"]; task = d["task"]
    spt = d["eeg"]["samples_per_tr"]; desc = d["eeg"]["desc"]
    dmn_i = d["fmri"]["dmn_idx"]; cen_i = d["fmri"]["cen_idx"]; ndi = d["fmri"]["n_difumo"]
    fdir = Path(d["features_dir"]) / f"sub-{subj}"
    eroot = Path(d["eeg_preproc_dir"]); runs = []
    for npz in sorted(fdir.glob(f"sub-{subj}_task-{task}_run-*_features.npz")):
        z = np.load(npz, allow_pickle=True)
        fm = np.asarray(z["fmri_features"], float)          # (n_tr, 66)
        n_tr = fm.shape[0]; run = npz.name.split("run-")[1][0]
        fif = (eroot/f"sub-{subj}"/ses/"eeg" /
               f"sub-{subj}_{ses}_task-{task}_run-{int(run):02d}_desc-{desc}_eeg.fif")
        if not fif.exists():
            continue
        raw = mne.io.read_raw_fif(str(fif), preload=True, verbose=False)
        bp, chs = band_power_run(raw, cfg["bands"], spt, n_tr, hrf)
        targets = dict(DMN=fm[:, dmn_i], CEN=fm[:, cen_i], PDA=fm[:, cen_i] - fm[:, dmn_i])
        runs.append(dict(run=run, n_tr=n_tr, targets=targets,
                         bp=bp, parcels=fm[:, :ndi], chs=chs))
    return runs
