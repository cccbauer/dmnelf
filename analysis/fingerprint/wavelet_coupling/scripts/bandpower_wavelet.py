"""
bandpower_wavelet.py
--------------------
Wavelet-based EEG feature extraction for fMRI network decoding.
Drop-in replacement for bandpower.py's Hilbert envelope approach.

Methods:
  - morlet:  Morlet CWT at center frequencies matching δ/θ/α/β/γ bands
  - dwt:     DWT with db4, coefficient energy per level mapped to bands
  - dwt_stats: DWT + statistical features (mean, std, entropy, energy per band)

Output shape matches Hilbert: dict band -> (n_tr, n_ch) HRF-convolved log power.
"""
from pathlib import Path
import numpy as np, yaml, mne, pywt
from scipy.stats import entropy as sp_entropy
mne.set_log_level("ERROR")


def load_config(p):
    cfg = yaml.safe_load(open(p)); d = cfg["data"]
    suffix = "_cluster" if Path("/projects/swglab").exists() else "_local"
    for key in ("features_dir", "eeg_preproc_dir", "confounds_dir"):
        d[key] = str(Path(d[key + suffix]).expanduser())
    return cfg


def canonical_hrf(tr, length_s=32, delay=6, undershoot=16):
    from scipy.stats import gamma
    t = np.arange(0, length_s, tr)
    h = gamma.pdf(t, delay) - gamma.pdf(t, undershoot) / 6.0
    return h / h.sum()


def hrf_convolve(x, hrf):
    return np.convolve(x, hrf, mode="full")[:len(x)]


def zscore(x):
    return (x - x.mean(0)) / (x.std(0) + 1e-12)


# ── Morlet CWT ──

def _freq_to_scale(freq, wavelet="cmor1.5-1.0", sfreq=500):
    """Convert center frequency to CWT scale for a given wavelet."""
    central_freq = pywt.central_frequency(wavelet)
    return central_freq * sfreq / freq


def morlet_power_run(raw, bands, spt, n_tr, hrf):
    """Morlet CWT power at band center frequencies → per-TR mean → log → HRF."""
    picks = mne.pick_types(raw.info, eeg=True, exclude=[])
    chs = [raw.ch_names[i] for i in picks]
    data = raw.get_data(picks=picks)  # (n_ch, n_samples)
    sfreq = raw.info["sfreq"]
    wavelet = "cmor1.5-1.0"

    band_centers = {name: (lo + hi) / 2.0 for name, (lo, hi) in bands.items()}
    scales = np.array([_freq_to_scale(fc, wavelet, sfreq) for fc in band_centers.values()])
    band_names = list(bands.keys())

    out = {}
    for bi, bname in enumerate(band_names):
        scale = np.array([scales[bi]])
        band_power = np.zeros((len(chs), n_tr))
        for ci in range(len(chs)):
            coefs, _ = pywt.cwt(data[ci, :n_tr * spt], scale, wavelet,
                                sampling_period=1.0 / sfreq)
            power = np.abs(coefs[0]) ** 2
            # Per-TR mean
            band_power[ci] = power[:n_tr * spt].reshape(n_tr, spt).mean(axis=1)
        lp = np.log(band_power.T + 1e-12)  # (n_tr, n_ch)
        lp = np.column_stack([hrf_convolve(lp[:, c], hrf) for c in range(lp.shape[1])])
        out[bname] = lp
    return out, chs


# ── DWT ──

def _dwt_level_map(sfreq=500, n_levels=5):
    """Map DWT decomposition levels to approximate frequency bands at given sfreq.
    At 500 Hz with 5 levels (db4):
      cD1: 125-250 Hz (noise, skip or use as gamma proxy)
      cD2: 62.5-125 Hz (high gamma, skip)
      cD3: 31.25-62.5 Hz (gamma 30-40 range)
      cD4: 15.625-31.25 Hz (beta 13-30 range)
      cD5: 7.8125-15.625 Hz (alpha 8-13 range)
      cA5: 0-7.8125 Hz (delta+theta)

    For better band mapping at 500 Hz, use 7 levels:
      cD7: 1.95-3.9 Hz ≈ delta
      cD6: 3.9-7.8 Hz ≈ theta
      cD5: 7.8-15.6 Hz ≈ alpha
      cD4: 15.6-31.25 Hz ≈ beta
      cD3: 31.25-62.5 Hz ≈ gamma
    """
    return {
        "delta": ("cD7", 7),  # ~2-4 Hz
        "theta": ("cD6", 6),  # ~4-8 Hz
        "alpha": ("cD5", 5),  # ~8-16 Hz
        "beta":  ("cD4", 4),  # ~16-31 Hz
        "gamma": ("cD3", 3),  # ~31-63 Hz
    }


def dwt_power_run(raw, bands, spt, n_tr, hrf, wavelet="db4", n_levels=7):
    """DWT coefficient energy per band → per-TR mean → log → HRF."""
    picks = mne.pick_types(raw.info, eeg=True, exclude=[])
    chs = [raw.ch_names[i] for i in picks]
    data = raw.get_data(picks=picks)  # (n_ch, n_samples)
    level_map = _dwt_level_map(raw.info["sfreq"], n_levels)
    band_names = list(bands.keys())

    out = {}
    for bname in band_names:
        if bname not in level_map:
            continue
        _, level_idx = level_map[bname]
        band_power = np.zeros((len(chs), n_tr))

        for ci in range(len(chs)):
            sig = data[ci, :n_tr * spt]
            coeffs = pywt.wavedec(sig, wavelet, level=n_levels)
            # coeffs[0] = cA_n, coeffs[1] = cD_n, ..., coeffs[n] = cD_1
            detail_idx = n_levels - level_idx + 1
            if detail_idx < 0 or detail_idx >= len(coeffs):
                band_power[ci] = 0
                continue
            detail = coeffs[detail_idx]
            # Upsample detail coefficients back to original length
            ratio = len(sig) / len(detail)
            detail_up = np.repeat(detail, int(np.ceil(ratio)))[:len(sig)]
            power = detail_up ** 2
            band_power[ci] = power.reshape(n_tr, spt).mean(axis=1)

        lp = np.log(band_power.T + 1e-12)
        lp = np.column_stack([hrf_convolve(lp[:, c], hrf) for c in range(lp.shape[1])])
        out[bname] = lp
    return out, chs


def dwt_stats_run(raw, bands, spt, n_tr, hrf, wavelet="db4", n_levels=7):
    """DWT + statistical features per band: power, mean, std, entropy.
    Returns 4× features per band (n_tr, n_ch * 4 per band)."""
    picks = mne.pick_types(raw.info, eeg=True, exclude=[])
    chs = [raw.ch_names[i] for i in picks]
    data = raw.get_data(picks=picks)
    level_map = _dwt_level_map(raw.info["sfreq"], n_levels)
    band_names = list(bands.keys())

    out = {}
    for bname in band_names:
        if bname not in level_map:
            continue
        _, level_idx = level_map[bname]
        n_ch = len(chs)
        # 4 features per channel: power, mean, std, entropy
        features = np.zeros((n_tr, n_ch * 4))

        for ci in range(n_ch):
            sig = data[ci, :n_tr * spt]
            coeffs = pywt.wavedec(sig, wavelet, level=n_levels)
            detail_idx = n_levels - level_idx + 1
            if detail_idx < 0 or detail_idx >= len(coeffs):
                continue
            detail = coeffs[detail_idx]
            ratio = len(sig) / len(detail)
            detail_up = np.repeat(detail, int(np.ceil(ratio)))[:len(sig)]

            # Reshape to (n_tr, spt) blocks
            blocks = detail_up.reshape(n_tr, spt)

            # Feature 1: energy (sum of squared coefficients)
            energy = np.sum(blocks ** 2, axis=1)
            features[:, ci] = np.log(energy + 1e-12)

            # Feature 2: mean absolute value
            features[:, n_ch + ci] = np.mean(np.abs(blocks), axis=1)

            # Feature 3: std
            features[:, 2 * n_ch + ci] = np.std(blocks, axis=1)

            # Feature 4: entropy (Shannon entropy of normalized coefficients)
            for ti in range(n_tr):
                b = np.abs(blocks[ti])
                b_norm = b / (b.sum() + 1e-12)
                features[ti, 3 * n_ch + ci] = sp_entropy(b_norm + 1e-12)

        # HRF convolve all features
        for fi in range(features.shape[1]):
            features[:, fi] = hrf_convolve(features[:, fi], hrf)

        out[bname] = features
    return out, chs


# ── Unified interface ──

def wavelet_power_run(raw, bands, spt, n_tr, hrf, method="morlet"):
    """Drop-in replacement for band_power_run(). Returns dict band -> features, ch_names."""
    if method == "morlet":
        return morlet_power_run(raw, bands, spt, n_tr, hrf)
    elif method == "dwt":
        return dwt_power_run(raw, bands, spt, n_tr, hrf)
    elif method == "dwt_stats":
        return dwt_stats_run(raw, bands, spt, n_tr, hrf)
    else:
        raise ValueError(f"Unknown wavelet method: {method}")


def gather_subject_wavelet(cfg, subj, hrf, method="morlet"):
    """Same as bandpower.gather_subject() but with wavelet features."""
    d = cfg["data"]; ses = d["session"]; task = d["task"]
    spt = d["eeg"]["samples_per_tr"]; desc = d["eeg"]["desc"]
    dmn_i = d["fmri"]["dmn_idx"]; cen_i = d["fmri"]["cen_idx"]; ndi = d["fmri"]["n_difumo"]
    fdir = Path(d["features_dir"]) / f"sub-{subj}"
    eroot = Path(d["eeg_preproc_dir"]); runs = []
    for npz in sorted(fdir.glob(f"sub-{subj}_task-{task}_run-*_features.npz")):
        z = np.load(npz, allow_pickle=True)
        fm = np.asarray(z["fmri_features"], float)
        n_tr = fm.shape[0]; run = npz.name.split("run-")[1][0]
        fif = (eroot/f"sub-{subj}"/ses/"eeg" /
               f"sub-{subj}_{ses}_task-{task}_run-{int(run):02d}_desc-{desc}_eeg.fif")
        if not fif.exists():
            continue
        raw = mne.io.read_raw_fif(str(fif), preload=True, verbose=False)
        bp, chs = wavelet_power_run(raw, cfg["bands"], spt, n_tr, hrf, method=method)
        targets = dict(DMN=fm[:, dmn_i], CEN=fm[:, cen_i], PDA=fm[:, cen_i] - fm[:, dmn_i])
        runs.append(dict(run=run, n_tr=n_tr, targets=targets,
                         bp=bp, parcels=fm[:, :ndi], chs=chs))
    return runs
