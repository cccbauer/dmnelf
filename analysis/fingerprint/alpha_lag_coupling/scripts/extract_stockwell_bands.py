#!/usr/bin/env python3
"""
extract_stockwell_bands.py  —  per-TR Stockwell equal-energy band power at Pz, resting-state
--------------------------------------------------------------------------------------------------
Alternative feature extraction to extract_residual_alpha.py's FOOOF-based residual alpha: reuses
efp_meirhasson's OWN Stockwell + data-driven equal-energy-band feature construction directly
(`channel_bandpower`/`bin_average` from efp_meirhasson/scripts/efp_features.py) — the same S-
transform and banding this project's best-performing decoder is built on — rather than a fixed
canonical-band reimplementation, applied here to resting-state ("task-rest") data instead of
"feedback". n_bands/freq_min/freq_max come from efp_meirhasson's own config.yaml (10 bands, 1-40 Hz).

This lets lagged_coupling_stockwell.py test whether Stockwell-derived band power shows different
— plausibly stronger or more cross-run-stable — lagged coupling with DMN/CEN than the residual-
alpha (FOOOF) pipeline found.

Output: results/stockwell_bands/<sub>_stockwell.npz — one array per run: bandpower[n_tr, n_bands]
(equal-energy bands, subject-specific edges — see band_hz_lo/band_hz_hi for each run's actual
Hz edges, since equal-energy banding is fit per run, not fixed across subjects).

Usage:  python extract_stockwell_bands.py --subjects dmnelf010
        python extract_stockwell_bands.py --subjects all
"""
import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import yaml

warnings.filterwarnings("ignore")
import mne  # noqa: E402

mne.set_log_level("ERROR")

HERE = Path(__file__).resolve().parent.parent
EFP_SCRIPTS = HERE.parent / "efp_meirhasson" / "scripts"
sys.path.insert(0, str(EFP_SCRIPTS))
from efp_features import channel_bandpower  # noqa: E402  (reuses efp_meirhasson's own feature code)

CFG = yaml.safe_load((HERE / "config.yaml").read_text())
D, S = CFG["data"], CFG["stockwell"]

EEG_DIR = Path(D["eeg_preproc_dir_local"]).expanduser()
FEAT_DIR = Path(D["features_dir_local"]).expanduser()
SES, DESC, TASK = D["session"], D["desc"], D["task"]
CHANNEL = D["eeg"]["channel"]
TR = float(D["fmri"]["tr"])
FMIN, FMAX, N_BANDS = int(S["freq_min"]), int(S["freq_max"]), int(S["n_bands"])

OUT = HERE / "results" / "stockwell_bands"


def _bin_average(power: np.ndarray, n_out: int) -> np.ndarray:
    """Same contiguous-bin downsampling as efp_features.bin_average, to TR resolution."""
    n_bands, n_samp = power.shape
    edges = np.linspace(0, n_samp, n_out + 1).astype(int)
    out = np.empty((n_bands, n_out))
    for i in range(n_out):
        a, b = edges[i], max(edges[i + 1], edges[i] + 1)
        out[:, i] = power[:, a:b].mean(axis=1)
    return out


def stockwell_bandpower_run(fif_path: Path, n_tr: int) -> dict:
    raw = mne.io.read_raw_fif(str(fif_path), preload=True, verbose="ERROR")
    raw.pick([CHANNEL])
    data = raw.get_data()[0]                                # [n_samples], Volts
    sf = float(raw.info["sfreq"])
    bp_full, band_hz, _ = channel_bandpower(data, sf, FMIN, FMAX, N_BANDS)   # [n_bands, n_samples]
    # raw (linear) power, no log — matches efp_meirhasson's own build_subject_features exactly
    # (bin_average is called directly on channel_bandpower's output there, no log transform).
    bandpower = _bin_average(bp_full, n_tr).T                               # [n_tr, n_bands]
    return dict(bandpower=bandpower, band_hz_lo=np.array([lo for lo, _ in band_hz]),
               band_hz_hi=np.array([hi for _, hi in band_hz]), sfreq=sf, channel=CHANNEL, tr=TR)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subjects", nargs="+", required=True,
                    help="subject ids, or 'all' for every subject in config.yaml")
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
            print(f"  {sub} run {run}: extracting Stockwell band power ({n_tr} TR)...", flush=True)
            runs[f"run{run}"] = stockwell_bandpower_run(fif, n_tr)
        if runs:
            save = {}
            for rk, rv in runs.items():
                for k, v in rv.items():
                    save[f"{rk}_{k}"] = v
            save["_runs"] = np.array(list(runs.keys()))
            save["_n_bands"] = N_BANDS
            np.savez_compressed(OUT / f"{sub}_stockwell.npz", **save)
            print(f"  saved {sub}_stockwell.npz")
        else:
            print(f"  {sub}: no runs found")


if __name__ == "__main__":
    main()
