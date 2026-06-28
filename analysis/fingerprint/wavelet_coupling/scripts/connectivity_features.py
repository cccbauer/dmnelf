"""
connectivity_features.py
------------------------
Phase-Locking Value (PLV) connectivity features between EEG channel pairs.
PLV captures inter-channel phase synchrony — the signature of distributed
networks like DMN that may not be visible in local power.

Output: per-TR PLV for each channel pair × band, HRF-convolved.
"""
from pathlib import Path
import numpy as np, mne
from scipy.signal import hilbert
from itertools import combinations
mne.set_log_level("ERROR")


def compute_plv_run(raw, bands, spt, n_tr, hrf, channel_pairs=None):
    """Compute PLV between channel pairs per band per TR → HRF convolve.

    Args:
        raw: MNE Raw object
        bands: dict band_name -> (lo, hi) Hz
        spt: samples per TR
        n_tr: number of TRs
        hrf: HRF kernel
        channel_pairs: list of (ch_idx_a, ch_idx_b) tuples, or None for all pairs

    Returns:
        dict band -> (n_tr, n_pairs) HRF-convolved PLV
        list of pair labels [(ch_a, ch_b), ...]
    """
    picks = mne.pick_types(raw.info, eeg=True, exclude=[])
    chs = [raw.ch_names[i] for i in picks]
    data = raw.get_data(picks=picks)  # (n_ch, n_samples)
    sfreq = raw.info["sfreq"]
    n_ch = len(chs)

    if channel_pairs is None:
        channel_pairs = list(combinations(range(n_ch), 2))

    pair_labels = [(chs[a], chs[b]) for a, b in channel_pairs]
    n_pairs = len(channel_pairs)

    out = {}
    for bname, (lo, hi) in bands.items():
        # Bandpass filter all channels
        filtered = mne.filter.filter_data(data, sfreq, lo, hi, verbose=False)

        # Get instantaneous phase via Hilbert transform
        analytic = hilbert(filtered, axis=-1)
        phase = np.angle(analytic)  # (n_ch, n_samples)

        plv_matrix = np.zeros((n_tr, n_pairs))

        for pi, (ch_a, ch_b) in enumerate(channel_pairs):
            phase_diff = phase[ch_a, :n_tr * spt] - phase[ch_b, :n_tr * spt]
            # Reshape to (n_tr, spt) blocks
            pd_blocks = phase_diff.reshape(n_tr, spt)
            # PLV = |mean(exp(i * phase_diff))| per TR
            plv_matrix[:, pi] = np.abs(np.mean(np.exp(1j * pd_blocks), axis=1))

        # HRF convolve
        from bandpower_wavelet import hrf_convolve
        for pi in range(n_pairs):
            plv_matrix[:, pi] = hrf_convolve(plv_matrix[:, pi], hrf)

        out[bname] = plv_matrix

    return out, pair_labels


def get_dmn_relevant_pairs(ch_names):
    """Select channel pairs relevant to DMN connectivity.
    DMN involves frontal-posterior and midline connectivity.

    Frontal: Fp1, Fp2, F3, F4, Fz, FC1, FC2
    Posterior: P3, P4, Pz, O1, O2, Oz, POz
    Midline: Fz, Cz, Pz, Oz, POz
    """
    frontal = {"Fp1", "Fp2", "F3", "F4", "Fz", "FC1", "FC2", "F7", "F8"}
    posterior = {"P3", "P4", "Pz", "O1", "O2", "Oz", "POz", "P7", "P8"}
    midline = {"Fz", "Cz", "Pz", "Oz", "POz"}
    temporal = {"T7", "T8", "TP9", "TP10"}

    pairs = []
    ch_idx = {ch: i for i, ch in enumerate(ch_names)}

    # Frontal-posterior pairs (DMN long-range)
    for f in frontal:
        for p in posterior:
            if f in ch_idx and p in ch_idx:
                pairs.append((ch_idx[f], ch_idx[p]))

    # Midline pairs (DMN midline axis)
    midline_list = [ch for ch in midline if ch in ch_idx]
    for a, b in combinations([ch_idx[ch] for ch in midline_list], 2):
        pairs.append((a, b))

    # Temporal-frontal pairs
    for t in temporal:
        for f in frontal:
            if t in ch_idx and f in ch_idx:
                pairs.append((ch_idx[t], ch_idx[f]))

    return list(set(pairs))  # deduplicate


def gather_subject_plv(cfg, subj, hrf, desc="preproc500Hz", use_dmn_pairs=True):
    """Load EEG, compute PLV features for one subject's feedback runs."""
    d = cfg["data"]; ses = d["session"]; task = d["task"]
    spt = d["eeg"]["samples_per_tr"]
    dmn_i = d["fmri"]["dmn_idx"]; cen_i = d["fmri"]["cen_idx"]; ndi = d["fmri"]["n_difumo"]
    fdir = Path(d["features_dir"]) / f"sub-{subj}"
    eroot = Path(d["eeg_preproc_dir"]); runs = []

    bands = cfg["bands"]

    for npz in sorted(fdir.glob(f"sub-{subj}_task-{task}_run-*_features.npz")):
        z = np.load(npz, allow_pickle=True)
        fm = np.asarray(z["fmri_features"], float)
        n_tr = fm.shape[0]; run = npz.name.split("run-")[1][0]
        fif = (eroot/f"sub-{subj}"/ses/"eeg" /
               f"sub-{subj}_{ses}_task-{task}_run-{int(run):02d}_desc-{desc}_eeg.fif")
        if not fif.exists():
            continue
        raw = mne.io.read_raw_fif(str(fif), preload=True, verbose=False)

        picks = mne.pick_types(raw.info, eeg=True, exclude=[])
        chs = [raw.ch_names[i] for i in picks]

        if use_dmn_pairs:
            pairs = get_dmn_relevant_pairs(chs)
        else:
            pairs = None

        plv, pair_labels = compute_plv_run(raw, bands, spt, n_tr, hrf, pairs)
        targets = dict(DMN=fm[:, dmn_i], CEN=fm[:, cen_i], PDA=fm[:, cen_i] - fm[:, dmn_i])
        runs.append(dict(run=run, n_tr=n_tr, targets=targets,
                         bp=plv, parcels=fm[:, :ndi], chs=pair_labels))
    return runs
