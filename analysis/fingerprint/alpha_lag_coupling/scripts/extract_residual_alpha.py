#!/usr/bin/env python3
"""
extract_residual_alpha.py  —  per-TR residual alpha power at Pz, resting-state runs
-------------------------------------------------------------------------------------
Adapts fsnr_eeg/scripts/eeg_fsnr_specparam.py's FOOOF-based periodic/aperiodic split to a
single posterior electrode (Pz) on DMNELF's resting-state ("task-rest") EEG, following
Jacob et al. 2025's method: per TR, a 2 s Hanning-window PSD is fit with FOOOF (aperiodic 1/f
+ oscillatory peaks) over 1-40 Hz (capped from their 1-50 Hz — see config.yaml), and residual
alpha = the summed oscillatory peak power with center frequency in the alpha band (8-12 Hz),
i.e. alpha power *above* the aperiodic background, not raw band power.

Output: results/residual_alpha/<sub>_alpha.npz — one array per run: residual_alpha[n_tr],
offset[n_tr], exponent[n_tr] (aperiodic), r2[n_tr] (FOOOF fit quality).

Usage:  python extract_residual_alpha.py --subjects dmnelf010
        python extract_residual_alpha.py --subjects all      # every subject in config.yaml
"""
import argparse
import warnings
from pathlib import Path

import numpy as np
import yaml

warnings.filterwarnings("ignore")
import mne  # noqa: E402

mne.set_log_level("ERROR")
from mne.time_frequency import psd_array_welch  # noqa: E402
from fooof import FOOOF  # noqa: E402

HERE = Path(__file__).resolve().parent.parent
CFG = yaml.safe_load((HERE / "config.yaml").read_text())
D = CFG["data"]
A = CFG["alpha"]

EEG_DIR = Path(D["eeg_preproc_dir_local"]).expanduser()
FEAT_DIR = Path(D["features_dir_local"]).expanduser()
SES, DESC, TASK = D["session"], D["desc"], D["task"]
CHANNEL = D["eeg"]["channel"]
TR = float(D["fmri"]["tr"])
WINDOW_SEC = float(A["window_sec"])
FIT_LO, FIT_HI = A["aperiodic_fit_range_hz"]
BAND_LO, BAND_HI = A["band_hz"]
MAX_N_PEAKS = int(A["fooof_max_n_peaks"])
PEAK_WIDTH_LIMITS = tuple(A["fooof_peak_width_limits"])

OUT = HERE / "results" / "residual_alpha"


def _fit_tr(freqs, psd):
    """One FOOOF fit for a single TR's PSD -> (offset, exponent, r2, residual_alpha)."""
    fm = FOOOF(max_n_peaks=MAX_N_PEAKS, peak_width_limits=PEAK_WIDTH_LIMITS,
              aperiodic_mode="fixed", verbose=False)
    fm.fit(freqs, psd, [FIT_LO, FIT_HI])
    offset, exponent = fm.get_params("aperiodic_params")
    r2 = float(fm.get_params("r_squared"))
    pk = np.atleast_2d(fm.get_params("peak_params"))          # [n_peaks, 3]: CF, PW, BW
    if pk.size == 0 or np.isnan(pk).all():
        alpha = 0.0
    else:
        pk = pk[~np.isnan(pk).any(axis=1)]
        in_band = (pk[:, 0] >= BAND_LO) & (pk[:, 0] <= BAND_HI)
        alpha = float(pk[in_band, 1].sum()) if in_band.any() else 0.0
    return float(offset), float(exponent), r2, alpha


def residual_alpha_run(fif_path: Path, n_tr: int) -> dict:
    raw = mne.io.read_raw_fif(str(fif_path), preload=True, verbose="ERROR")
    raw.pick([CHANNEL])
    data = raw.get_data()[0]                                   # [n_samples], Volts
    sf = float(raw.info["sfreq"])
    half = int(round(WINDOW_SEC * sf / 2))
    n_samp_per_tr = sf * TR
    n_samples = data.shape[0]

    offset = np.full(n_tr, np.nan); exponent = np.full(n_tr, np.nan)
    r2 = np.full(n_tr, np.nan); residual_alpha = np.full(n_tr, np.nan)
    for t in range(n_tr):
        center = int(round((t + 0.5) * n_samp_per_tr))
        a, b = max(0, center - half), min(n_samples, center + half)
        seg = data[a:b]
        if seg.size < int(sf * 1.0):                          # need >= 1s of data to fit
            continue
        psd, freqs = psd_array_welch(seg[np.newaxis, :], sf, fmin=FIT_LO, fmax=FIT_HI,
                                     n_fft=seg.size, verbose=False)
        try:
            offset[t], exponent[t], r2[t], residual_alpha[t] = _fit_tr(freqs, psd[0])
        except Exception:
            continue
    return dict(offset=offset, exponent=exponent, r2=r2, residual_alpha=residual_alpha,
               sfreq=sf, channel=CHANNEL, tr=TR)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subjects", nargs="+", required=True,
                    help="subject ids, or 'all' for every subject in config.yaml")
    ap.add_argument("--max-tr", type=int, default=None,
                    help="debug: only process the first N TRs of each run")
    a = ap.parse_args()
    subs = D["subjects"]["all"] if a.subjects == ["all"] else a.subjects
    OUT.mkdir(parents=True, exist_ok=True)

    for sub in subs:
        runs = {}
        for run in range(1, D["n_runs"] + 1):
            feat_npz = FEAT_DIR / f"sub-{sub}" / f"sub-{sub}_task-{TASK}_run-{run}_features.npz"
            fif = EEG_DIR / f"sub-{sub}" / SES / "eeg" / f"sub-{sub}_{SES}_task-{TASK}_run-{run:02d}_desc-{DESC}_eeg.fif"
            if not feat_npz.exists() or not fif.exists():
                print(f"  {sub} run {run}: missing feat or fif, skipping"); continue
            n_tr = int(np.load(feat_npz, allow_pickle=True)["fmri_features"].shape[0])
            if a.max_tr:
                n_tr = min(n_tr, a.max_tr)
            print(f"  {sub} run {run}: extracting residual alpha ({n_tr} TR)...", flush=True)
            runs[f"run{run}"] = residual_alpha_run(fif, n_tr)
        if runs:
            save = {}
            for rk, rv in runs.items():
                for k, v in rv.items():
                    save[f"{rk}_{k}"] = v
            save["_runs"] = np.array(list(runs.keys()))
            np.savez_compressed(OUT / f"{sub}_alpha.npz", **save)
            print(f"  saved {sub}_alpha.npz")
        else:
            print(f"  {sub}: no runs found")


if __name__ == "__main__":
    main()
