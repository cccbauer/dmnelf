#!/usr/bin/env python3
"""
cen_ceiling_extract.py  (cluster)  —  noise ceiling of the CEN-BOLD target
--------------------------------------------------------------------------
The max achievable r(EEG, CEN) is bounded by the reliability of the CEN-BOLD timecourse
itself: r <= sqrt(reliability). Here we estimate that reliability by SPLIT-HALF of the
personalized CEN-mask voxels: split voxels into two random halves, take each half's mean
timecourse, correlate them, Spearman-Brown correct. Done within-feedback and full-run, at
increasing temporal smoothing (1/3/5 TR) — because the ceiling rises as you average.

Per subject x feedback run: personalized CEN mask voxel timeseries from the feedback BOLD,
denoised (motion-full + WM/CSF + cosine, detrend, z-score; NO low-pass to preserve the
fluctuations EEG must track). Saves reliability rows + the mask-mean timecourse (to validate
against the existing target).

Usage: python cen_ceiling_extract.py --cohort {dmnelf,rtbpd} --subject SUB --out DIR
"""
import argparse, os, glob, warnings
from pathlib import Path
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
from nilearn.maskers import NiftiMasker
from nilearn.interfaces.fmriprep import load_confounds

COH = {"dmnelf": dict(root="/projects/swglab/data/DMNELF", ses="ses-dmnelf"),
       "rtbpd": dict(root="/projects/swglab/data/rtBPD", ses="ses-nf1"),
       "rtbpd_nf2": dict(root="/projects/swglab/data/rtBPD", ses="ses-nf2")}
TR = 1.2
BASELINE_TR, HRF_DROP = 25, 5
SMOOTH = [1, 3, 5]
NREP = 50
RNG = np.random.default_rng(0)


def smooth(x, w):
    if w <= 1:
        return x
    k = np.ones(w) / w
    return np.convolve(x, k, mode="same")


def corr(a, b):
    if np.std(a) < 1e-9 or np.std(b) < 1e-9:
        return np.nan
    return float(np.corrcoef(a, b)[0, 1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohort", required=True, choices=list(COH))
    ap.add_argument("--subject", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(); out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    c = COH[a.cohort]; sub = a.subject; ses = c["ses"]
    fp = f"{c['root']}/derivatives/fmriprep_25.2.5_fmap"
    mdir = f"{c['root']}/derivatives/network_masks/sub-{sub}"
    cen = f"{mdir}/sub-{sub}_space-MNI152NLin6Asym_res-2_cen_mask.nii.gz"
    dmn = f"{mdir}/sub-{sub}_space-MNI152NLin6Asym_res-2_dmn_mask.nii.gz"
    if not os.path.exists(cen):
        print(f"{sub}: no CEN mask"); return
    masker = NiftiMasker(cen, standardize="zscore_sample", detrend=True, t_r=TR)
    dmasker = NiftiMasker(dmn, standardize="zscore_sample", detrend=True, t_r=TR) if os.path.exists(dmn) else None
    rows = []; means = {}
    for img in sorted(glob.glob(f"{fp}/sub-{sub}/{ses}/func/sub-{sub}_{ses}_task-feedback_run-*_space-MNI152NLin6Asym_res-2_desc-preproc_bold.nii.gz")):
        run = int(img.split("run-")[1][:2])
        try:
            confs, samp = load_confounds(img, strategy=("high_pass", "motion", "wm_csf", "global_signal"),
                                         motion="full", wm_csf="basic", global_signal="basic")
            gcol = [c for c in confs.columns if "global_signal" in c]
            g = confs[gcol[0]].to_numpy()
            # keep ALL volumes (no sample_mask) so the target aligns 1:1 with the EEG features
            V = masker.fit_transform(img, confounds=confs.drop(columns=gcol))  # global RETAINED
            T, nvox = V.shape
            g = g[:T]
            # GSR variant: regress the global signal out of every CEN voxel
            G = np.column_stack([np.ones(T), g])
            Vg = V - G @ np.linalg.lstsq(G, V, rcond=None)[0]
            means[f"run{run}"] = V.mean(1); means[f"run{run}_gsr"] = Vg.mean(1)
            if dmasker is not None:  # clean DMN mask-mean (same confound regression)
                means[f"run{run}_dmn"] = dmasker.fit_transform(img, confounds=confs.drop(columns=gcol)).mean(1)
            fb = slice(BASELINE_TR + HRF_DROP, T)
            quick = {}
            for denoise, Vd in [("raw", V), ("gsr", Vg)]:
                for w in SMOOTH:
                    for win, sl in [("fb", fb), ("full", slice(0, T))]:
                        rs = []
                        for _ in range(NREP):
                            perm = RNG.permutation(nvox)
                            a1 = smooth(Vd[:, perm[:nvox // 2]].mean(1), w)
                            b1 = smooth(Vd[:, perm[nvox // 2:]].mean(1), w)
                            rs.append(corr(a1[sl], b1[sl]))
                        r = float(np.nanmean(rs)); rel = 2 * r / (1 + r) if r > -1 else np.nan
                        rows.append(dict(cohort=a.cohort, subject=sub, run=run, nvox=nvox,
                                         denoise=denoise, smooth_tr=w, window=win, r_halfhalf=r,
                                         reliability=rel, ceiling=np.sqrt(rel) if rel > 0 else np.nan))
                        if w == 1 and win == "fb":
                            quick[denoise] = rel
            print(f"  {sub} run{run}: nvox={nvox} fb rel(1TR) raw={quick['raw']:.2f} "
                  f"gsr={quick['gsr']:.2f}", flush=True)
        except Exception as e:
            print(f"  {sub} run{run}: FAIL {type(e).__name__}: {str(e)[:70]}", flush=True)
    if rows:
        pd.DataFrame(rows).to_csv(out / f"cenrel_{a.cohort}_{sub}.csv", index=False)
        np.savez_compressed(out / f"cenmean_{a.cohort}_{sub}.npz",
                            **{k: v for k, v in means.items()})
        print(f"saved cenrel_{a.cohort}_{sub}.csv ({len(rows)} rows)", flush=True)


if __name__ == "__main__":
    main()
