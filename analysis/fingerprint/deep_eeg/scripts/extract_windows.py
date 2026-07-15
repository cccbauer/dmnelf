#!/usr/bin/env python3
"""
extract_windows.py  (cluster)  —  raw-EEG windows for R-EEGNet (EEG -> CEN/DMN BOLD)
------------------------------------------------------------------------------------
Per feedback TR t, take the 15 s of EEG PRECEDING it (HRF lag 0-15 s, per Stabile 2025),
from the 500 Hz fif band-passed 1-40 Hz (or <20 Hz variant) and resampled to 80 Hz ->
window [31 ch, 1200 samp]. Aligned to CLEAN confound-regressed targets (cenmean: CEN=run{N},
DMN=run{N}_dmn), feedback block only, target z-scored per run.

Output: {sub}_windows[_lt20].npz  with X [N,31,1200] float32, y_cen[N], y_dmn[N], run[N], ch_names.
Usage: python extract_windows.py --cohort {dmnelf,rtbpd,rtbpd_nf2} --subject SUB --clean-dir DIR --out DIR [--hi 20]
"""
import argparse, glob, warnings
from pathlib import Path
import numpy as np
warnings.filterwarnings("ignore")
import mne; mne.set_log_level("ERROR")

COH = {"dmnelf": dict(eeg="/projects/swglab/data/DMNELF/derivatives/eeg_preprocessed", ses="ses-dmnelf", pfx="cenmean_dmnelf_"),
       "rtbpd": dict(eeg="/projects/swglab/data/rtBPD/derivatives/eeg_preprocessed", ses="ses-nf", pfx="cenmean_rtbpd_"),
       "rtbpd_nf2": dict(eeg="/projects/swglab/data/rtBPD/derivatives/eeg_preprocessed", ses="ses-nf2", pfx="cenmean_rtbpd_nf2_")}
TR = 1.2; SF = 80.0; WIN_S = 15.0; W = int(WIN_S * SF)   # 1200
BASELINE_TR, HRF_DROP = 25, 5


def zs(x): return (x - np.nanmean(x)) / (np.nanstd(x) + 1e-12)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohort", required=True, choices=list(COH))
    ap.add_argument("--subject", required=True)
    ap.add_argument("--clean-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--lo", type=float, default=1.0); ap.add_argument("--hi", type=float, default=40.0)
    a = ap.parse_args(); out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    c = COH[a.cohort]; sub = a.subject; ses = c["ses"]
    cmf = Path(a.clean_dir) / f"{c['pfx']}{sub}.npz"
    if not cmf.exists():
        print(f"{sub}: no clean targets"); return
    cm = np.load(cmf, allow_pickle=True)
    Xs, ycen, ydmn, runid = [], [], [], []; ch_names = None
    for fif in sorted(glob.glob(f"{c['eeg']}/sub-{sub}/{ses}/eeg/sub-{sub}_{ses}_task-feedback_run-*_desc-preproc500Hz_eeg.fif")):
        run = int(fif.split("run-")[1][:2])
        ck, dk = f"run{run}", f"run{run}_dmn"
        if ck not in cm.files or dk not in cm.files:
            continue
        cen = np.asarray(cm[ck], float); dmn = np.asarray(cm[dk], float); n_tr = len(cen)
        raw = mne.io.read_raw_fif(fif, preload=True, verbose=False)
        raw.pick("eeg").filter(a.lo, a.hi, verbose=False).resample(SF, verbose=False)
        if ch_names is None:
            ch_names = raw.ch_names
        data = raw.get_data()                       # [31, T80]
        Tt = data.shape[1]
        fb = list(range(BASELINE_TR + HRF_DROP, n_tr))
        cz, dz = zs(cen[fb]), zs(dmn[fb])           # per-run z-score over feedback block
        for k, t in enumerate(fb):
            end = int(round(t * TR * SF))            # window ends at TR t onset
            if end - W < 0 or end > Tt:
                continue
            Xs.append(data[:, end - W:end].astype(np.float32))
            ycen.append(cz[k]); ydmn.append(dz[k]); runid.append(run)
    if not Xs:
        print(f"{sub}: no windows"); return
    suf = "" if a.hi >= 40 else "_lt20"
    np.savez_compressed(out / f"{sub}_windows{suf}.npz",
                        X=np.stack(Xs), y_cen=np.array(ycen, np.float32),
                        y_dmn=np.array(ydmn, np.float32), run=np.array(runid),
                        ch_names=np.array(ch_names))
    print(f"{sub}: saved {len(Xs)} windows [31,{W}] ({a.lo}-{a.hi}Hz)", flush=True)


if __name__ == "__main__":
    main()
