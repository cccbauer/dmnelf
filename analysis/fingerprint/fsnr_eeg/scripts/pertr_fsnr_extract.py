#!/usr/bin/env python3
"""
pertr_fsnr_extract.py  (cluster)  —  per-TR EEG f-SNR from the within-TR EEG
----------------------------------------------------------------------------
The running/trailing f-SNR spans many TRs and only captured the rest->feedback state
step. Here we compute a genuinely INSTANTANEOUS per-TR f-SNR from the EEG inside each TR,
so it can track within-feedback fluctuations of PDA/CEN/DMN.

Per feedback run, per TR t, per channel (31):
  TEMPORAL f-SNR @ 1.2 s (exactly one TR = 600 samples):
    tsnr[t,ch,band] = envelope_mean / envelope_std  of the band-limited Hilbert envelope
                      inside that TR (a within-TR amplitude signal-to-noise, per band).
  SPECTRAL features @ 2.4 s and @ 3.6 s windows (2 / 3 TR, centered on the TR):
    offset, exponent  (aperiodic 1/f = noise), periodic_total (oscillatory = signal),
    alpha_periodic, r2, and log band power (delta..gamma).
    -> oscillatory/aperiodic f-SNR = periodic/offset and band/broadband are formed in decoding.

Output: results/pertr_fsnr/<sub>_pertr.npz  (per run: tsnr[n_tr,31,5]; per spectral window
        offset/exponent/periodic/alpha/r2 [n_tr,31] and bandpow [n_tr,31,5]; chs; n_tr)

Usage (cluster): python pertr_fsnr_extract.py --cohort {dmnelf,rtbpd} --subjects dmnelf001
"""
import argparse, warnings, glob, re
from pathlib import Path
import numpy as np
warnings.filterwarnings("ignore")
import mne; mne.set_log_level("ERROR")
from mne.time_frequency import psd_array_welch
from scipy.signal import butter, filtfilt, hilbert
from fooof import FOOOFGroup

SF = 500.0
TR = 1.2
SPT = int(round(SF * TR))                      # 600 samples / TR
SPEC_WINS = {"w2400": 2.4, "w3600": 3.6}       # spectral windows (s) = 2 TR, 3 TR
FMIN, FMAX = 1.0, 45.0
BANDS = dict(delta=(1, 4), theta=(4, 8), alpha=(8, 13), beta=(13, 30), gamma=(30, 40))

COH = {
    "dmnelf": dict(eeg="/projects/swglab/data/DMNELF/derivatives/eeg_preprocessed",
                   feat="/projects/swglab/data/DMNELF/derivatives/cyclic_features",
                   ses="ses-dmnelf"),
    "rtbpd":  dict(eeg="/projects/swglab/data/rtBPD/derivatives/eeg_preprocessed",
                   feat="/projects/swglab/data/rtBPD/derivatives/cyclic_features",
                   ses="ses-nf"),
}
OUT = Path(__file__).resolve().parent.parent / "results" / "pertr_fsnr"


def band_env(data, lo, hi):
    """Full-run band-limited Hilbert envelope [nch, N]."""
    b, a = butter(4, [lo / (SF / 2), min(hi, SF / 2 - 1) / (SF / 2)], btype="band")
    return np.abs(hilbert(filtfilt(b, a, data, axis=1), axis=1))


def temporal_tsnr(data, n_tr):
    """Per-TR within-window envelope mean/std, per band. [n_tr, nch, nband]."""
    nch = data.shape[0]; out = np.full((n_tr, nch, len(BANDS)), np.nan)
    for bi, (lo, hi) in enumerate(BANDS.values()):
        env = band_env(data, lo, hi)                       # [nch, N]
        for t in range(n_tr):
            seg = env[:, t * SPT:(t + 1) * SPT]
            if seg.shape[1] < SPT // 2:
                continue
            out[t, :, bi] = seg.mean(1) / (seg.std(1) + 1e-12)
    return out


def spectral(data, n_tr, win_s):
    N = data.shape[1]; nch = data.shape[0]; half = int(win_s * SF / 2)
    off = np.full((n_tr, nch), np.nan); exp = off.copy(); per = off.copy()
    alpha = off.copy(); r2 = off.copy(); bp = np.full((n_tr, nch, len(BANDS)), np.nan)
    nfft = int(SF * 1.5); nov = int(SF * 0.75)
    for t in range(n_tr):
        c = int((t + 0.5) * SPT); a0, b0 = max(0, c - half), min(N, c + half)
        seg = data[:, a0:b0]
        if seg.shape[1] < SF * 1.5:
            continue
        psd, freqs = psd_array_welch(seg, SF, fmin=FMIN, fmax=FMAX, n_fft=nfft,
                                     n_overlap=nov, verbose=False)
        for bi, (lo, hi) in enumerate(BANDS.values()):
            m = (freqs >= lo) & (freqs < hi)
            if m.any():
                bp[t, :, bi] = np.log(psd[:, m].mean(1) + 1e-30)
        fg = FOOOFGroup(max_n_peaks=6, peak_width_limits=(1, 12), aperiodic_mode="fixed", verbose=False)
        try:
            fg.fit(freqs, psd, [FMIN, FMAX])
        except Exception:
            continue
        ap = fg.get_params("aperiodic_params")
        off[t] = ap[:, 0]; exp[t] = ap[:, 1]; r2[t] = fg.get_params("r_squared")
        pk = fg.get_params("peak_params")
        if pk.ndim == 2 and len(pk):
            idx = pk[:, 3].astype(int)
            for ci in range(nch):
                sel = idx == ci
                per[t, ci] = pk[sel, 1].sum() if sel.any() else 0.0
                am = sel & (pk[:, 0] >= 8) & (pk[:, 0] <= 13)
                alpha[t, ci] = pk[am, 1].sum() if am.any() else 0.0
    return dict(offset=off, exponent=exp, periodic=per, alpha=alpha, r2=r2, bandpow=bp)


def run_features(fif, n_tr):
    raw = mne.io.read_raw_fif(str(fif), preload=True, verbose=False)
    picks = mne.pick_types(raw.info, eeg=True, exclude=[])
    chs = [raw.ch_names[i] for i in picks]
    data = raw.get_data(picks=picks)
    out = dict(chs=chs, n_tr=n_tr, tsnr=temporal_tsnr(data, n_tr))
    for key, w in SPEC_WINS.items():
        out[key] = spectral(data, n_tr, w)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohort", required=True, choices=list(COH))
    ap.add_argument("--subjects", nargs="+", required=True)
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args(); out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    c = COH[a.cohort]; ses = c["ses"]
    for sub in a.subjects:
        runs = {}
        for npz in sorted(glob.glob(f"{c['feat']}/sub-{sub}/sub-{sub}_task-feedback_run-*_features.npz")):
            run = int(re.search(r"run-(\d+)_features", npz).group(1))
            n_tr = int(np.load(npz, allow_pickle=True)["fmri_features"].shape[0])
            fif = Path(c["eeg"]) / f"sub-{sub}" / ses / "eeg" / \
                f"sub-{sub}_{ses}_task-feedback_run-{run:02d}_desc-preproc500Hz_eeg.fif"
            if not fif.exists():
                print(f"  {sub} run {run}: no fif", flush=True); continue
            runs[f"run{run}"] = run_features(fif, n_tr)
            print(f"  {sub} run {run}: done ({n_tr} TR)", flush=True)
        if runs:
            save = {k: np.array(v, dtype=object) for k, v in runs.items()}
            save["_runs"] = np.array(list(runs.keys()))
            np.savez_compressed(out / f"{sub}_pertr.npz", **save)
            print(f"saved {sub}_pertr.npz ({len(runs)} runs)", flush=True)


if __name__ == "__main__":
    main()
