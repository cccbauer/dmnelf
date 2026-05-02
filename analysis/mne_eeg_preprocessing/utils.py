# utils.py — DMNELF microstate PDA pipeline
# SSH/SCP helpers + validated data loader + TESS machinery

import subprocess
import numpy as np
import pandas as pd
from pathlib import Path
import mne
import py_compile

from config import (
    CLUSTER_SSH, CLUSTER_BASE, EEG_ROOT, DIFUMO_ROOT,
    SUBJECTS, RUN_MAP, SFREQ, TR_SAMPLES, ALIGNMENT_TOL,
    DMN_IDX, CEN_IDX, BASELINE_VOLS, PSA_WINDOW, TR, N_MICROSTATES
)

# ══════════════════════════════════════════════════════════════
# 1.  SSH / SCP HELPERS
# ══════════════════════════════════════════════════════════════

def run_ssh(cmd, verbose=True):
    """Run a command on the cluster via SSH."""
    full = "/usr/bin/ssh " + CLUSTER_SSH + " 'bash -l -c \"" + cmd + "\"'"
    result = subprocess.run(
        full, shell=True, capture_output=True, text=True
    )
    if verbose:
        if result.stdout:
            print(result.stdout)
        if result.stderr:
            # Filter out known harmless cluster warnings
            filtered = "\n".join([
                line for line in result.stderr.split("\n")
                if not any(x in line for x in [
                    "flatpak",
                    "libcrypto",
                    "OPENSSL",
                    "Loading matlab",
                    "Loading requirement",
                    "OpenJDK",
                ])
            ]).strip()
            if filtered:
                print(filtered)
    return result


def scp_to(local_path, remote_path, verbose=True):
    """Copy a local file to the cluster."""
    cmd = ("/usr/bin/scp '" + str(local_path)
           + "' " + CLUSTER_SSH + ":" + remote_path)
    result = subprocess.run(
        cmd, shell=True, capture_output=True, text=True
    )
    if verbose:
        if result.stdout: print(result.stdout)
        if result.stderr: print(result.stderr)
    # Verify transfer
    verify = run_ssh("ls " + remote_path + " 2>/dev/null || echo MISSING",
                     verbose=False)
    if "MISSING" in verify.stdout:
        print("WARNING: SCP failed for " + str(local_path))
    elif verbose:
        print("Verified: " + remote_path)
    return result


def scp_from(remote_path, local_path, verbose=True):
    """Copy a file from the cluster to local."""
    cmd = ("/usr/bin/scp " + CLUSTER_SSH + ":" + remote_path
           + " " + str(local_path))
    result = subprocess.run(
        cmd, shell=True, capture_output=True, text=True
    )
    if verbose:
        if result.stdout: print(result.stdout)
        if result.stderr: print(result.stderr)
    return result


def deploy_script(local_path, verbose=True):
    """Syntax-check then SCP a script to cluster scripts dir."""
    local_path = Path(local_path)
    try:
        py_compile.compile(str(local_path), doraise=True)
        print("Syntax OK: " + local_path.name)
    except py_compile.PyCompileError as e:
        print("SYNTAX ERROR: " + str(e))
        raise
    remote = CLUSTER_BASE + "/scripts/" + local_path.name
    scp_to(local_path, remote, verbose=verbose)
    return remote


def make_cluster_dirs():
    """Create all required directories on the cluster."""
    dirs = [
        CLUSTER_BASE,
        CLUSTER_BASE + "/scripts",
        CLUSTER_BASE + "/features",
        CLUSTER_BASE + "/targets",
        CLUSTER_BASE + "/models",
        CLUSTER_BASE + "/results",
        CLUSTER_BASE + "/logs",
        CLUSTER_BASE + "/qc",
    ]
    cmd = "mkdir -p " + " ".join(dirs)
    return run_ssh(cmd)


# ══════════════════════════════════════════════════════════════
# 2.  DATA LOADING  (validated data_loader.py)
# ══════════════════════════════════════════════════════════════

def load_eeg(fif_path, target_sfreq=SFREQ):
    """
    Load preprocessed .fif.
    Drops non-EEG channels, average reference, resamples.
    Returns: (n_ch, n_samples) float32 uV, list of ch_names
    """
    raw = mne.io.read_raw_fif(str(fif_path), preload=True, verbose=False)
    drop = [ch for ch in raw.ch_names
            if any(x in ch.upper() for x in
                   ("ECG", "EKG", "EMG", "EOG", "STIM", "STATUS"))]
    if drop:
        raw.drop_channels(drop)
    ref_check = float(np.abs(raw.get_data().mean(axis=0)).mean())
    if ref_check > 1e-7:
        raw.set_eeg_reference("average", projection=False, verbose=False)
    if raw.info["sfreq"] != target_sfreq:
        raw.resample(target_sfreq, verbose=False)
    data = (raw.get_data() * 1e6).astype(np.float32)
    return data, list(raw.ch_names)


def load_fmri(tsv_path):
    """
    Load DiFuMo-64 TSV, select ROI_* columns only.
    Returns: (64, n_volumes) float32
    """
    df = pd.read_csv(str(tsv_path), sep="\t")
    roi_cols = [c for c in df.columns if c.startswith("ROI_")]
    if len(roi_cols) != 64:
        raise ValueError("Expected 64 ROI cols, got " + str(len(roi_cols)))
    return df[roi_cols].values.T.astype(np.float32)


def find_matched_runs(subject, task):
    """
    Return list of run IDs where both EEG .fif and DiFuMo .tsv exist.
    Checks against RUN_MAP for the task.
    """
    matched = []
    for run in RUN_MAP.get(task, []):
        fif_name = (subject + "_ses-dmnelf_task-" + task
                    + "_run-" + run + "_desc-preproc_eeg.fif")
        tsv_name = (subject + "_ses-dmnelf_task-" + task
                    + "_run-" + run + "_desc-difumo64_timeseries.tsv")
        fif = (Path(EEG_ROOT) / subject / "ses-dmnelf" / "eeg" / fif_name)
        tsv = Path(DIFUMO_ROOT) / tsv_name
        if fif.exists() and tsv.exists():
            matched.append(run)
    return matched


# ══════════════════════════════════════════════════════════════
# 3.  MURFI-ALIGNED PDA  (validated)
# ══════════════════════════════════════════════════════════════

def baseline_zscore(x, n_baseline=BASELINE_VOLS):
    """
    Z-score relative to first n_baseline volumes.
    Matches MURFI real-time normalization (Hinds 2011, Bloom 2023).
    Args:  x (n_vols,) float
    Returns: z (n_vols,) float32
    """
    n  = len(x)
    nb = min(n_baseline, n)
    mu  = float(x[:nb].mean())
    sig = float(x[:nb].std())
    if sig < 1e-8:
        sig = 1.0
    return ((x - mu) / sig).astype(np.float32)


def rolling_zscore(x, window=PSA_WINDOW):
    """
    Causal rolling z-score using stride tricks.
    Used for PSA DMN suppression multiplier.
    Args:  x (n,) float
    Returns: z (n,) float32
    """
    n      = len(x)
    xf     = x.astype(np.float64)
    padded = np.full(n + window - 1, np.nan)
    padded[window - 1:] = xf
    shape   = (n, window)
    strides = (padded.strides[0], padded.strides[0])
    wins = np.lib.stride_tricks.as_strided(
        padded, shape=shape, strides=strides
    )
    with np.errstate(all="ignore"):
        ma_mean = np.nanmean(wins, axis=1)
        ma_std  = np.nanstd(wins,  axis=1)
    safe = np.where(ma_std < 1e-6, 1.0, ma_std)
    z    = np.where(ma_std < 1e-6, 0.0, (xf - ma_mean) / safe)
    return z.astype(np.float32)


def compute_pda(fmri, dmn_idx=DMN_IDX, cen_idx=CEN_IDX,
                n_baseline=BASELINE_VOLS):
    """
    MURFI-aligned PDA.
    Z-score each parcel relative to first n_baseline vols,
    then PDA(t) = mean(CEN_z, t) - mean(DMN_z, t).
    Args:  fmri (64, n_vols)
    Returns: pda (n_vols,) float32
    """
    fmri_z = np.array([baseline_zscore(fmri[i], n_baseline)
                        for i in range(fmri.shape[0])])
    cen = fmri_z[cen_idx, :].mean(axis=0)
    dmn = fmri_z[dmn_idx, :].mean(axis=0)
    return (cen - dmn).astype(np.float32)


def compute_psa(fmri, psa_window=PSA_WINDOW, dmn_idx=DMN_IDX,
                cen_idx=CEN_IDX, n_baseline=BASELINE_VOLS):
    """
    PSA(t) = PDA(t) x clip(1 - DMN_zscore_rolling(t), 0, 1)
    Returns: pda (n_vols,), psa (n_vols,) float32
    """
    pda    = compute_pda(fmri, dmn_idx, cen_idx, n_baseline)
    fmri_z = np.array([baseline_zscore(fmri[i], n_baseline)
                        for i in range(fmri.shape[0])])
    dmn_z  = fmri_z[dmn_idx, :].mean(axis=0)
    mult   = np.clip(1.0 - rolling_zscore(dmn_z, psa_window),
                     0.0, 1.0).astype(np.float32)
    return pda, (pda * mult).astype(np.float32)


# ══════════════════════════════════════════════════════════════
# 4.  GFP / GMD
# ══════════════════════════════════════════════════════════════

def compute_gfp(eeg):
    """
    Global Field Power = spatial std across channels at each sample.
    Args:  eeg (n_ch, n_samples)
    Returns: gfp (n_samples,) float32
    """
    return eeg.std(axis=0).astype(np.float32)


def compute_gmd(eeg):
    """
    Global Map Dissimilarity between consecutive samples.
    Normalized by GFP. First sample = 0.
    Args:  eeg (n_ch, n_samples)
    Returns: gmd (n_samples,) float32
    """
    norm = eeg / (eeg.std(axis=0, keepdims=True) + 1e-10)
    diff = np.diff(norm, axis=1)
    gmd  = np.sqrt((diff ** 2).mean(axis=0)).astype(np.float32)
    return np.concatenate([[0.0], gmd]).astype(np.float32)


def get_gfp_peaks(gfp, outlier_sd=3.0):
    """
    Find local GFP maxima, reject outliers > outlier_sd standard deviations.
    Used for k-means fitting (standard microstate analysis practice).
    Args:  gfp (n_samples,) float32
    Returns: peak_indices (n_peaks,) int
    """
    from scipy.signal import argrelmax
    peaks = argrelmax(gfp, order=1)[0]
    if outlier_sd is not None:
        mu  = gfp[peaks].mean()
        sig = gfp[peaks].std()
        peaks = peaks[gfp[peaks] < mu + outlier_sd * sig]
    return peaks


# ══════════════════════════════════════════════════════════════
# 5.  TESS  (Custo 2014 / 2017)
# ══════════════════════════════════════════════════════════════

def normalize_map(m):
    """L2-normalize a topographic map (n_ch,)."""
    n = np.linalg.norm(m)
    return m / n if n > 1e-10 else m


def polarity_invariant_corr(a, b):
    """
    Absolute spatial correlation between two maps.
    Ignores polarity (standard for microstate analysis).
    """
    a = a - a.mean()
    b = b - b.mean()
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    if denom < 1e-10:
        return 0.0
    return float(abs(np.dot(a, b) / denom))


def tess_project(eeg, templates):
    """
    TESS Stage 1: spatial GLM projection.
    For each time sample, project instantaneous topography onto
    each template map via pseudoinverse, yielding continuous
    T-hat coefficients (Custo 2014).

    This preserves amplitude and within-TR temporal dynamics —
    superior to binary winner-takes-all labeling.

    Args:
        eeg:       (n_ch, n_samples) float32
        templates: (n_maps, n_ch)    float32  — L2-normalized
    Returns:
        t_hat:     (n_maps, n_samples) float32
    """
    # Pseudoinverse of template matrix: (n_maps, n_ch)
    pinv = np.linalg.pinv(templates)          # (n_ch, n_maps)
    t_hat = templates @ eeg                    # (n_maps, n_samples)
    return t_hat.astype(np.float32)


def hrf_canonical(tr=TR, peak_delay=5.0, undershoot_delay=15.0,
                  peak_disp=1.0, undershoot_disp=1.0,
                  peak_undershoot_ratio=6.0, onset=0.0,
                  duration=32.0):
    """
    Canonical double-gamma HRF (Glover 1999).
    Returns: (n_samples,) normalized to peak=1
    """
    from scipy.stats import gamma
    dt      = tr
    t       = np.arange(onset, onset + duration, dt)
    peak    = gamma.pdf(t, peak_delay      / peak_disp,
                        scale=peak_disp)
    under   = gamma.pdf(t, undershoot_delay / undershoot_disp,
                        scale=undershoot_disp)
    hrf     = peak - under / peak_undershoot_ratio
    hrf    /= hrf.max()
    return hrf.astype(np.float32)


def convolve_hrf(signal, hrf):
    """
    Convolve a 1D signal with HRF and trim to original length.
    Args:  signal (n,), hrf (m,)
    Returns: convolved (n,) float32
    """
    conv = np.convolve(signal, hrf, mode="full")
    return conv[:len(signal)].astype(np.float32)


def downsample_to_tr(signal, tr_samples=TR_SAMPLES):
    """
    Downsample EEG-rate signal to fMRI TR rate by averaging within TR.
    Args:  signal (n_eeg_samples,), tr_samples: EEG samples per TR
    Returns: (n_vols,) float32
    """
    n_vols = len(signal) // tr_samples
    out    = np.zeros(n_vols, dtype=np.float32)
    for i in range(n_vols):
        out[i] = signal[i * tr_samples:(i + 1) * tr_samples].mean()
    return out


def compute_tess_features(eeg, templates, tr_samples=TR_SAMPLES, tr=TR):
    """
    Full TESS feature pipeline for one EEG run.

    Steps:
      1. Project EEG onto templates → T-hat (n_maps, n_eeg_samples)
      2. Convolve each T-hat with canonical HRF
      3. Downsample to TR
      4. Compute GFP and GMD, downsample to TR
      5. Stack: (n_vols, n_maps + 2)

    Feature columns:
      0..N_MICROSTATES-1  : T-hat per microstate (HRF-convolved)
      N_MICROSTATES       : GFP (downsampled)
      N_MICROSTATES + 1   : GMD (downsampled)

    Args:
        eeg:       (n_ch, n_samples) float32
        templates: (n_maps, n_ch)    float32
        tr_samples: EEG samples per TR
        tr:         TR in seconds
    Returns:
        features: (n_vols, n_maps + 2) float32
    """
    hrf   = hrf_canonical(tr=tr)
    n_maps = templates.shape[0]

    # T-hat at EEG rate
    t_hat = tess_project(eeg, templates)   # (n_maps, n_samples)

    # HRF convolve + downsample each T-hat
    t_hat_tr = np.zeros(
        (n_maps, len(eeg[0]) // tr_samples), dtype=np.float32
    )
    for m in range(n_maps):
        convolved     = convolve_hrf(t_hat[m], hrf)
        t_hat_tr[m]   = downsample_to_tr(convolved, tr_samples)

    # GFP and GMD at EEG rate → downsample
    gfp    = compute_gfp(eeg)
    gmd    = compute_gmd(eeg)
    gfp_tr = downsample_to_tr(gfp, tr_samples)
    gmd_tr = downsample_to_tr(gmd, tr_samples)

    # Trim to minimum length (alignment safety)
    n_vols = min(t_hat_tr.shape[1], len(gfp_tr), len(gmd_tr))
    feats  = np.zeros((n_vols, n_maps + 2), dtype=np.float32)
    feats[:, :n_maps]   = t_hat_tr[:, :n_vols].T
    feats[:, n_maps]    = gfp_tr[:n_vols]
    feats[:, n_maps + 1] = gmd_tr[:n_vols]

    return feats