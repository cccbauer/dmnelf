#!/usr/bin/env python3
"""
eeg_fsnr_specparam.py  —  Stream B, Flavor 2 (extraction; runs on cluster)
--------------------------------------------------------------------------
Per-TR sliding-window PSD -> specparam (FOOOF) per channel: separates the periodic
(oscillatory = signal) component from the aperiodic 1/f (noise) background. Also computes
NON-convolved band power (for a clean EEG variability-quench, unlike the HRF-convolved
cache). These are instantaneous EEG features; HRF alignment to BOLD is handled in matching.

Per feedback run, per TR t (centered ~8 s window), per channel:
  offset, exponent  (aperiodic 1/f: the noise background)
  periodic_total    (summed oscillatory peak power above 1/f = the signal)
  alpha_periodic    (oscillatory power with CF in 8-13 Hz)
  bandpow[5]        (raw integrated power, non-convolved: delta..gamma)
EEG oscillatory/aperiodic f-SNR is then periodic_total / offset (computed in matching).

Output: results/specparam/<sub>_specparam.npz   (per run arrays [n_tr, 31] + bandpow[n_tr,31,5])

Usage (cluster):  python eeg_fsnr_specparam.py --subjects dmnelf001
"""
import argparse, warnings, glob, re
from pathlib import Path
import numpy as np
warnings.filterwarnings("ignore")
import mne; mne.set_log_level("ERROR")
from mne.time_frequency import psd_array_welch
from fooof import FOOOFGroup

EEG_DIR = Path("/projects/swglab/data/DMNELF/derivatives/eeg_preprocessed")
FEAT_DIR = Path("/projects/swglab/data/DMNELF/derivatives/cyclic_features")
OUT = Path(__file__).resolve().parent.parent / "results" / "specparam"
SES = "ses-dmnelf"; DESC = "preproc500Hz"; SF = 500.0; SPT = 600
WIN_S = 8.0; FMIN, FMAX = 1.0, 45.0
BANDS = dict(delta=(1, 4), theta=(4, 8), alpha=(8, 13), beta=(13, 30), gamma=(30, 45))


def run_features(fif, n_tr):
    raw = mne.io.read_raw_fif(str(fif), preload=True, verbose=False)
    picks = mne.pick_types(raw.info, eeg=True, exclude=[])
    chs = [raw.ch_names[i] for i in picks]
    data = raw.get_data(picks=picks)                      # [nch, N]
    N = data.shape[1]; half = int(WIN_S * SF / 2); nch = len(chs)
    off = np.full((n_tr, nch), np.nan); exp = off.copy()
    per = off.copy(); alpha = off.copy(); r2 = off.copy()
    bp = np.full((n_tr, nch, len(BANDS)), np.nan)
    for t in range(n_tr):
        c = int((t + 0.5) * SPT)
        a, b = max(0, c - half), min(N, c + half)
        seg = data[:, a:b]
        if seg.shape[1] < SF * 2:
            continue
        psd, freqs = psd_array_welch(seg, SF, fmin=FMIN, fmax=FMAX,
                                     n_fft=int(SF * 2), n_overlap=int(SF), verbose=False)
        # non-convolved band power
        for bi, (lo, hi) in enumerate(BANDS.values()):
            m = (freqs >= lo) & (freqs < hi)
            bp[t, :, bi] = np.log(psd[:, m].mean(1) + 1e-30)
        fg = FOOOFGroup(max_n_peaks=6, peak_width_limits=(1, 12), aperiodic_mode="fixed", verbose=False)
        try:
            fg.fit(freqs, psd, [FMIN, FMAX])
        except Exception:
            continue
        ap = fg.get_params("aperiodic_params")     # [nch, 2] offset, exponent
        off[t] = ap[:, 0]; exp[t] = ap[:, 1]
        r2[t] = fg.get_params("r_squared")
        pk = fg.get_params("peak_params")          # [n_peaks, 4]: CF, PW, BW, ch_idx
        if pk.ndim == 2 and len(pk):
            idx = pk[:, 3].astype(int)
            for ci in range(nch):
                sel = idx == ci
                per[t, ci] = pk[sel, 1].sum() if sel.any() else 0.0
                am = sel & (pk[:, 0] >= 8) & (pk[:, 0] <= 13)
                alpha[t, ci] = pk[am, 1].sum() if am.any() else 0.0
    return dict(offset=off, exponent=exp, periodic=per, alpha=alpha, r2=r2, bandpow=bp, chs=chs)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--subjects", nargs="+", required=True)
    args = ap.parse_args(); OUT.mkdir(parents=True, exist_ok=True)
    for sub in args.subjects:
        runs = {}
        for npz in sorted(glob.glob(str(FEAT_DIR / f"sub-{sub}" / f"sub-{sub}_task-feedback_run-*_features.npz"))):
            run = int(re.search(r"run-(\d+)_features", npz).group(1))
            n_tr = int(np.load(npz, allow_pickle=True)["fmri_features"].shape[0])
            fif = EEG_DIR / f"sub-{sub}" / SES / "eeg" / f"sub-{sub}_{SES}_task-feedback_run-{run:02d}_desc-{DESC}_eeg.fif"
            if not fif.exists():
                print(f"  {sub} run {run}: no fif"); continue
            runs[f"run{run}"] = run_features(fif, n_tr)
            print(f"  {sub} run {run}: done ({n_tr} TR)")
        if runs:
            save = {k: np.array(v, dtype=object) for k, v in runs.items()}
            save["_runs"] = np.array(list(runs.keys()))
            np.savez_compressed(OUT / f"{sub}_specparam.npz", **save)
            print(f"saved {sub}_specparam.npz")


if __name__ == "__main__":
    main()
