#!/usr/bin/env python3
"""
restfc_extract.py  (runs on cluster)
------------------------------------
Pre/post resting-state connectivity for the mbNF cohorts (Zhang-2023-style DMN measures).
For each rest run, extract timeseries from the fMRIPrep BOLD with confound denoising
(motion-full + WM/CSF + high-pass cosines, low-pass 0.1 Hz, detrend), and compute
Fisher-z-averaged connectivity in three families:

  DiFuMo-64 (atlas, Yeo-7 labels):
    within_dmn_difumo  — mean pairwise r among Default components
    within_cen_difumo  — mean pairwise r among Cont (frontoparietal) components
    dmn_cen_difumo     — mean r between Default and Cont
    mpfc_pcc, acc_pcc, acc_mpfc — core DMN-hub edges
       MPFC=comp38 (dorsomedial PFC); PCC=comps3,13 (post/mid-post cingulate); ACC=comp22
  Personalized DMN/CEN masks (the participant's neurofeedback masks):
    within_dmn_pers / within_cen_pers — mean pairwise r among (subsampled) mask voxels
    dmn_cen_pers                       — r between the mask-mean DMN and CEN timeseries

Rest runs: DMNELF ses-dmnelf task-rest run-01(pre)/02(post);
           rtBPD ses-nf1 & ses-nf2 task-rest run-01,02(pre)/03,04(post).
Usage: python restfc_extract.py --cohort {dmnelf,rtbpd} --subject SUB
"""
import argparse, os, glob, warnings
from pathlib import Path
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
os.environ.setdefault("NILEARN_DATA", "/projects/swglab/software/nilearn_data")
from nilearn.maskers import NiftiMapsMasker, NiftiMasker
from nilearn.datasets import fetch_atlas_difumo
from nilearn.interfaces.fmriprep import load_confounds

ROOT = {"dmnelf": "/projects/swglab/data/DMNELF", "rtbpd": "/projects/swglab/data/rtBPD"}
SES = {"dmnelf": ["ses-dmnelf"], "rtbpd": ["ses-nf1", "ses-nf2"]}
OUT = Path(__file__).resolve().parent.parent / "results"
TR = 1.2
MPFC, PCC, ACC = [38], [3, 13], [22]        # DiFuMo-64 0-based indices
MAXVOX = 300                                 # subsample cap for personalized pairwise r


def fz(r): return np.arctanh(np.clip(r, -0.999, 0.999))


def within(C, idx):
    s = C[np.ix_(idx, idx)]; iu = np.triu_indices(len(idx), 1)
    return float(np.tanh(np.mean(fz(s[iu])))) if len(iu[0]) else np.nan


def between(C, a, b):
    return float(np.tanh(np.mean(fz(C[np.ix_(a, b)]))))


def pairwise_mean(ts):
    if ts.shape[1] > MAXVOX:
        ts = ts[:, np.random.default_rng(0).choice(ts.shape[1], MAXVOX, replace=False)]
    C = np.corrcoef(ts.T); iu = np.triu_indices(ts.shape[1], 1)
    return float(np.tanh(np.mean(fz(C[iu]))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohort", required=True, choices=["dmnelf", "rtbpd"])
    ap.add_argument("--subject", required=True)
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args(); out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    sub = a.subject; root = ROOT[a.cohort]; fp = f"{root}/derivatives/fmriprep_25.2.5_fmap"

    atlas = fetch_atlas_difumo(dimension=64, resolution_mm=2, data_dir=os.environ["NILEARN_DATA"])
    y = pd.DataFrame(atlas.labels)["yeo_networks7"].astype(str).values
    DMN = [i for i, v in enumerate(y) if v.startswith("Default")]
    CEN = [i for i, v in enumerate(y) if v.startswith("Cont")]
    maps = NiftiMapsMasker(atlas.maps, standardize="zscore_sample", detrend=True,
                           low_pass=0.1, t_r=TR, verbose=0)
    mdir = f"{root}/derivatives/network_masks/sub-{sub}"
    dmn_m = f"{mdir}/sub-{sub}_space-MNI152NLin6Asym_res-2_dmn_mask.nii.gz"
    cen_m = f"{mdir}/sub-{sub}_space-MNI152NLin6Asym_res-2_cen_mask.nii.gz"
    has_pers = os.path.exists(dmn_m) and os.path.exists(cen_m)
    if has_pers:
        dmnM = NiftiMasker(dmn_m, standardize="zscore_sample", detrend=True, low_pass=0.1, t_r=TR)
        cenM = NiftiMasker(cen_m, standardize="zscore_sample", detrend=True, low_pass=0.1, t_r=TR)

    rows = []
    for ses in SES[a.cohort]:
        for img in sorted(glob.glob(f"{fp}/sub-{sub}/{ses}/func/sub-{sub}_{ses}_task-rest_run-*_space-MNI152NLin6Asym_res-2_desc-preproc_bold.nii.gz")):
            run = int(img.split("run-")[1][:2])
            try:
                confs, samp = load_confounds(img, strategy=("high_pass", "motion", "wm_csf"),
                                             motion="full", wm_csf="basic")
                ts = maps.fit_transform(img, confounds=confs, sample_mask=samp)
                C = np.corrcoef(ts.T)
                r = dict(cohort=a.cohort, subject=sub, session=ses, run=run, n_tr=ts.shape[0],
                         within_dmn_difumo=within(C, DMN), within_cen_difumo=within(C, CEN),
                         dmn_cen_difumo=between(C, DMN, CEN),
                         mpfc_pcc=between(C, MPFC, PCC), acc_pcc=between(C, ACC, PCC),
                         acc_mpfc=between(C, ACC, MPFC))
                if has_pers:
                    dv = dmnM.fit_transform(img, confounds=confs, sample_mask=samp)
                    cv = cenM.fit_transform(img, confounds=confs, sample_mask=samp)
                    r["within_dmn_pers"] = pairwise_mean(dv)
                    r["within_cen_pers"] = pairwise_mean(cv)
                    r["dmn_cen_pers"] = float(np.corrcoef(dv.mean(1), cv.mean(1))[0, 1])
                rows.append(r)
                print(f"  {sub} {ses} rest-{run:02d}: mpfc-pcc={r['mpfc_pcc']:+.3f} "
                      f"withinDMN(difumo)={r['within_dmn_difumo']:+.3f} "
                      f"pers-DMN-CEN={r.get('dmn_cen_pers', float('nan')):+.3f}")
            except Exception as e:
                print(f"  {sub} {ses} rest-{run}: FAILED {type(e).__name__}: {str(e)[:90]}")
    if rows:
        pd.DataFrame(rows).to_csv(out / f"restfc_{a.cohort}_{sub}.csv", index=False)
        print(f"saved restfc_{a.cohort}_{sub}.csv ({len(rows)} runs, personalized={has_pers})")


if __name__ == "__main__":
    main()
