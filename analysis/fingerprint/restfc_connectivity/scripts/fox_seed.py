#!/usr/bin/env python3
"""
fox_seed.py  (cluster)  —  Fox-2005 MPFC seed-to-voxel resting connectivity
---------------------------------------------------------------------------
For one subject: denoise every resting-state run (motion-full + WM/CSF + high-pass
cosines, low-pass 0.1 Hz, detrend, z-score) exactly as restfc; extract the Fox-2005
MPFC seed mean timeseries; compute per-voxel Pearson r with the seed; Fisher-z;
average across the subject's rest runs -> one subject seed-connectivity z-map.

Also reports the mean z inside the Fox PCC and L/R DLPFC seeds (DMN posterior hub /
task-positive anticorrelation checks). Seeds are 2mm and share the BOLD affine.

Usage: python fox_seed.py --cohort {dmnelf,rtbpd} --subject SUB --out DIR
"""
import argparse, os, glob, warnings
from pathlib import Path
import numpy as np, nibabel as nib, pandas as pd
from scipy.stats import zscore
warnings.filterwarnings("ignore")
os.environ.setdefault("NILEARN_DATA", "/projects/swglab/software/nilearn_data")
from nilearn.maskers import NiftiMasker
from nilearn.interfaces.fmriprep import load_confounds

ROOT = {"dmnelf": "/projects/swglab/data/DMNELF", "rtbpd": "/projects/swglab/data/rtBPD"}
SES = {"dmnelf": ["ses-dmnelf"], "rtbpd": ["ses-nf1", "ses-nf2"]}
SEEDDIR = "/projects/swglab/data/rtBPD/analysis/masks/Fox2005"
MPFC = f"{SEEDDIR}/FOX2005_MPFC.nii"
TARGETS = {"pcc": f"{SEEDDIR}/FOX2005_PCC.nii",
           "lDLPFC": f"{SEEDDIR}/FOX2005_leftDLPFC.nii",
           "rDLPFC": f"{SEEDDIR}/FOX2005_rightDLPFC.nii"}
TR = 1.2
CLEAN = dict(standardize="zscore_sample", detrend=True, low_pass=0.1, t_r=TR)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohort", required=True, choices=["dmnelf", "rtbpd"])
    ap.add_argument("--subject", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(); out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    sub = a.subject; fp = f"{ROOT[a.cohort]}/derivatives/fmriprep_25.2.5_fmap"

    acc = cnt = None; affine = header = None; nrun = 0
    tmask = {k: nib.load(v).get_fdata() > 0 for k, v in TARGETS.items()}
    for ses in SES[a.cohort]:
        for img in sorted(glob.glob(f"{fp}/sub-{sub}/{ses}/func/sub-{sub}_{ses}_task-rest_run-*_space-MNI152NLin6Asym_res-2_desc-preproc_bold.nii.gz")):
            bm = img.replace("desc-preproc_bold", "desc-brain_mask")
            if not os.path.exists(bm):
                print(f"  no brain mask for {os.path.basename(img)}"); continue
            try:
                confs, samp = load_confounds(img, strategy=("high_pass", "motion", "wm_csf"),
                                             motion="full", wm_csf="basic")
                brain = NiftiMasker(bm, **CLEAN)
                X = brain.fit_transform(img, confounds=confs, sample_mask=samp)      # [T, V]
                seed = NiftiMasker(MPFC, **CLEAN).fit_transform(img, confounds=confs, sample_mask=samp).mean(1)
                r = (zscore(X, axis=0, ddof=1) * zscore(seed, ddof=1)[:, None]).mean(0)
                z = np.arctanh(np.clip(r, -0.999999, 0.999999))
                zvol = brain.inverse_transform(z).get_fdata()
                m = nib.load(bm).get_fdata() > 0
                if acc is None:
                    acc = np.zeros(zvol.shape); cnt = np.zeros(zvol.shape)
                    affine = nib.load(bm).affine; header = nib.load(bm).header
                acc[m] += zvol[m]; cnt[m] += 1; nrun += 1
            except Exception as e:
                print(f"  {ses} {os.path.basename(img)}: FAILED {type(e).__name__}: {str(e)[:80]}")
    if nrun == 0:
        print(f"{sub}: no usable rest runs"); return
    subz = np.divide(acc, cnt, out=np.zeros_like(acc), where=cnt > 0)
    nib.Nifti1Image(subz.astype(np.float32), affine, header).to_filename(out / f"foxz_{a.cohort}_{sub}.nii.gz")
    row = dict(cohort=a.cohort, subject=sub, n_rest_runs=nrun)
    for k, tm in tmask.items():
        row[f"z_{k}"] = float(subz[tm].mean())
    pd.DataFrame([row]).to_csv(out / f"foxroi_{a.cohort}_{sub}.csv", index=False)
    print(f"{sub}: {nrun} rest runs | z(PCC)={row['z_pcc']:+.3f} "
          f"z(lDLPFC)={row['z_lDLPFC']:+.3f} z(rDLPFC)={row['z_rDLPFC']:+.3f}")


if __name__ == "__main__":
    main()
