#!/usr/bin/env python3
"""
fig_extract_preproc.py  (cluster, env eeg_preproc)  —  Fig 1 assets
-------------------------------------------------------------------
Runs the real preprocessing pipeline (eeg_preproc.preprocess_run) in capture mode for the
exemplar subjects and dumps per-stage EEG traces + PSDs + rejected-ICA topographies, so
Fig 1 shows the full chain on REAL signal. Stage 0 (raw gradient-contaminated) is loaded
separately from the BrainVision .vhdr.

Stages: 0 raw-BVA (gradient) | 1 gradient-corrected EDF | 2 filtered 1-40 | 3 post-BCG |
        4 post-ICA | 5 final (CAR+interp).
Output: ~/figassets/preproc_<sub>.npz
"""
import sys, glob, re
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path.home() / "figscripts"))   # our capture-instrumented copy
import mne
mne.set_log_level("ERROR")
import eeg_preproc as ep

RAW_ROOT = Path("/projects/swglab/data/DMNELF/rawdata_eeg")
BVA_ROOT = Path("/projects/swglab/data/DMNELF/analysis/MNE/bids/rawdata")
OUT = Path.home() / "figassets"; OUT.mkdir(exist_ok=True)
CHANS = ["Pz", "Fz", "Fp1", "Cz"]                 # traces to store; plotter picks
WIN = 6.0                                          # seconds shown per stage


def psd(x, sf, fmax=45):
    from scipy.signal import welch
    f, p = welch(x, sf, nperseg=int(min(len(x), sf * 2)))
    m = f <= fmax
    return f[m], 10 * np.log10(p[m] + 1e-20)


def pick_trace(raw, t0, chans=CHANS):
    sf = raw.info["sfreq"]; i0 = int(t0 * sf); n = int(WIN * sf)
    out = {}
    for c in chans:
        if c in raw.ch_names:
            x = raw.get_data(picks=c)[0]
            out[c] = x[i0:i0 + n]
    t = np.arange(n) / sf
    return t, out, sf


def find_run(sub):
    """First available feedback EDF (else any) for capture."""
    d = RAW_ROOT / sub / "ses-dmnelf" / "eeg"
    for task in ("feedback", "rest", "shortrest"):
        fs = sorted(glob.glob(str(d / f"{sub}_ses-dmnelf_task-{task}_run-*_desc-bvaAC1kHz_eeg.edf")))
        if fs:
            run = re.search(r"run-(\w+?)_desc", Path(fs[0]).name).group(1)
            return task, run
    return None, None


def stage0_bva(sub, t0):
    """Raw gradient-contaminated trace from BrainVision .vhdr (during scanning)."""
    fs = sorted(glob.glob(str(BVA_ROOT / sub / "ses-dmnelf" / "eeg" / "*_raw_eeg.vhdr")))
    if not fs:
        return None
    raw = mne.io.read_raw_brainvision(fs[0], preload=True, verbose="ERROR")
    # pick a window well into scanning where gradient artifact is present
    t, tr, sf = pick_trace(raw, min(t0, raw.times[-1] - WIN - 1))
    ps = {c: psd(tr[c], sf) for c in tr}
    return dict(t=t, tr=tr, sf=sf, psd=ps)


def main():
    subs = sys.argv[1:] or ["sub-dmnelf1002", "sub-dmnelf009"]
    for sub in subs:
        task, run = find_run(sub)
        if run is None:
            print(f"{sub}: no EDF"); continue
        print(f"{sub}: capturing task-{task} run-{run}", flush=True)
        cap = {}
        try:
            ep.preprocess_run(sub, task, run, overwrite=True, sfreq_target=500.0, capture=cap)
        except Exception as e:
            print(f"  pipeline error: {e}")
        t0 = 60.0
        blob = {"subject": sub, "task": task, "run": run}
        # stages 1-5 (aligned, from capture)
        stage_keys = [("gradient_corrected", "s1_gradient_corrected"),
                      ("filtered", "s2_filtered"), ("post_bcg", "s3_post_bcg"),
                      ("post_ica", "s4_post_ica"), ("final", "s5_final")]
        for capk, name in stage_keys:
            if capk in cap:
                raw = cap[capk]; t0s = min(t0, raw.times[-1] - WIN - 1)
                t, tr, sf = pick_trace(raw, t0s)
                blob[name + "_t"] = t; blob[name + "_sf"] = sf
                for c, x in tr.items():
                    blob[f"{name}_{c}"] = x
                    f, pxx = psd(x, sf); blob[f"{name}_{c}_psdf"] = f; blob[f"{name}_{c}_psd"] = pxx
        # stage 0 raw BVA
        s0 = stage0_bva(sub, t0)
        if s0:
            blob["s0_t"] = s0["t"]; blob["s0_sf"] = s0["sf"]
            for c, x in s0["tr"].items():
                blob[f"s0_gradient_{c}"] = x
                blob[f"s0_gradient_{c}_psdf"], blob[f"s0_gradient_{c}_psd"] = s0["psd"][c]
        # ICA rejected-component topographies + labels
        if "ica" in cap and cap["ica"] is not None:
            ica = cap["ica"]; comps = ica.get_components()   # [n_ch, n_comp]
            rej = cap.get("artifact_components", [])[:6]
            labels = cap.get("ica_labels", None)
            blob["ica_ch_names"] = np.array(ica.ch_names)
            blob["ica_rejected"] = np.array(rej)
            blob["ica_topos"] = np.array([comps[:, i] for i in rej]) if rej else np.zeros((0,))
            blob["ica_rej_labels"] = np.array([labels[i] if labels else "artifact" for i in rej])
        # ECG + R-peaks for the BCG panel
        if cap.get("ecg") is not None:
            blob["ecg"] = cap["ecg"]; blob["ecg_sf"] = cap.get("sfreq_1k", 1000.0)
            if cap.get("rpeaks") is not None:
                blob["rpeaks"] = np.asarray(cap["rpeaks"])
        np.savez_compressed(OUT / f"preproc_{sub}.npz", **blob)
        print(f"  saved {OUT}/preproc_{sub}.npz  ({len([k for k in blob if '_s' in k or k.startswith('s')])} arrays)")


if __name__ == "__main__":
    main()
