#!/usr/bin/env python3
"""
eeg_fsnr_calm.py  —  clinical-outcome link (rtBPD)
--------------------------------------------------
Relate the per-run self-reported CALM rating (slider_calm, 1-9; behavioural TSVs) to
(i) the pure EEG f-SNR marker (frontal-theta running signal-to-noise) and (ii) the offline
BOLD PDA regulation, in the adolescents-with-elevated-BPD-traits cohort. Tests whether the
calibration-free EEG biomarker tracks a clinical/behavioural outcome.

EEG f-SNR per run = mean over the feedback block of the frontal-theta running f-SNR
(trailing_mean/trailing_std of theta band power, averaged over frontal electrodes).
"""
from pathlib import Path
import numpy as np, pandas as pd, glob
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from eeg_fsnr_bandpower import running_fsnr, FRONTAL, zs
from fsnr_fmri import BASELINE_TR, HRF_DROP

PROJ = Path(__file__).resolve().parent.parent
DATA = PROJ / "data"; RES = PROJ / "results"


def run_eeg_fsnr(rd, chs):
    pi = [i for i, c in enumerate(chs) if c in FRONTAL]
    n = rd["n_tr"]
    ef = np.nanmean(np.column_stack([running_fsnr(rd["bp"]["theta"][:, i])[1] for i in pi]), 1)
    fb = slice(BASELINE_TR + HRF_DROP, n)
    pda = np.asarray(rd["targets"]["PDA"], float)
    reg = pda[fb].mean() - pda[:BASELINE_TR].mean()   # offline PDA regulation (fb - baseline)
    return float(np.nanmean(ef[fb])), float(reg)


def main():
    sl = pd.read_csv(DATA / "rtbpd_sliders.csv")
    rows = []
    for ses, folder in [("nf1", DATA / "rtbpd_nf1"), ("nf2", DATA / "rtbpd_nf2")]:
        for f in sorted(glob.glob(str(folder / "*_bandpower.npz"))):
            sub = Path(f).name.split("_")[0]
            z = np.load(f, allow_pickle=True); chs = [str(c) for c in z["ch_names"]]
            for rd in z["runs_data"]:
                run = int(rd["run"])
                fsnr, reg = run_eeg_fsnr(rd, chs)
                rows.append(dict(subject=sub, session=ses, run=run, eeg_fsnr=fsnr, pda_reg=reg))
    e = pd.DataFrame(rows)
    m = e.merge(sl[["subject", "session", "run", "slider_calm", "slider_difficulty", "rt_pda_mean"]],
                on=["subject", "session", "run"], how="inner").dropna(subset=["slider_calm"])
    m.to_csv(RES / "eeg_fsnr_calm.csv", index=False)
    print(f"{len(m)} runs merged (EEG+calm), {m.subject.nunique()} subjects\n")

    def report(x, y, lab):
        d = m.dropna(subset=[x, y])
        rp = np.corrcoef(d[x], d[y])[0, 1]
        g = d.groupby("subject").agg(a=(x, "mean"), b=(y, "mean"))
        rs = np.corrcoef(g.a, g.b)[0, 1] if len(g) > 3 else np.nan
        wc = [np.corrcoef(gg[x], gg[y])[0, 1] for _, gg in d.groupby("subject")
              if len(gg) >= 3 and gg[x].std() > 0 and gg[y].std() > 0]
        print(f"  {lab:34s} pooled r={rp:+.2f} (n={len(d)})  between-subj r={rs:+.2f}  within-subj r={np.nanmean(wc):+.2f}")

    print("=== does the marker track self-reported CALM? ===")
    report("eeg_fsnr", "slider_calm", "EEG frontal-theta f-SNR  ~ calm")
    report("pda_reg", "slider_calm", "offline BOLD PDA regulation ~ calm")
    report("rt_pda_mean", "slider_calm", "real-time PDA ~ calm")
    print("\n=== sanity ===")
    report("eeg_fsnr", "pda_reg", "EEG f-SNR ~ offline PDA reg")
    report("eeg_fsnr", "slider_difficulty", "EEG f-SNR ~ difficulty")


if __name__ == "__main__":
    main()
