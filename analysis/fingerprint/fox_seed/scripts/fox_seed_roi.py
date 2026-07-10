#!/usr/bin/env python3
"""
fox_seed_roi.py  (cluster, fast)  —  Fox MPFC-seed -> target-ROI connectivity per rest run
------------------------------------------------------------------------------------------
Per rest run, Fisher-z correlation of the Fox-2005 MPFC seed mean-timeseries with the PCC
and L/R DLPFC seeds, denoised identically to restfc/fox_seed. The four seeds are merged into
ONE integer label image so each 4D BOLD is read a SINGLE time (NiftiLabelsMasker), ~4x faster
than one NiftiMasker per seed. Saves per cohort so a killed job never loses the other cohort.

Output: fox_roi_{cohort}.csv  (cohort, subject, session, run, z_pcc, z_ldlpfc, z_rdlpfc)
Usage: python fox_seed_roi.py --cohort {dmnelf,rtbpd} --out DIR
"""
import argparse, os, glob, warnings
from pathlib import Path
import numpy as np, nibabel as nib, pandas as pd
from scipy.stats import zscore
warnings.filterwarnings("ignore")
os.environ.setdefault("NILEARN_DATA", "/projects/swglab/software/nilearn_data")
from nilearn.maskers import NiftiLabelsMasker
from nilearn.interfaces.fmriprep import load_confounds

ROOT = {"dmnelf": "/projects/swglab/data/DMNELF", "rtbpd": "/projects/swglab/data/rtBPD"}
SES = {"dmnelf": ["ses-dmnelf"], "rtbpd": ["ses-nf1", "ses-nf2"]}
SEEDDIR = "/projects/swglab/data/rtBPD/analysis/masks/Fox2005"
# label order: 1=mpfc(seed) 2=pcc 3=ldlpfc 4=rdlpfc
SEED_ORDER = [("mpfc", "FOX2005_MPFC.nii"), ("pcc", "FOX2005_PCC.nii"),
              ("ldlpfc", "FOX2005_leftDLPFC.nii"), ("rdlpfc", "FOX2005_rightDLPFC.nii")]
TR = 1.2


def build_labels():
    ref = nib.load(f"{SEEDDIR}/{SEED_ORDER[0][1]}")
    lab = np.zeros(ref.shape, dtype=np.int16)
    for i, (_, fn) in enumerate(SEED_ORDER, start=1):
        m = nib.load(f"{SEEDDIR}/{fn}").get_fdata() > 0
        lab[m] = i   # seeds are non-overlapping Fox ROIs; last-wins is harmless
    return nib.Nifti1Image(lab, ref.affine, ref.header)


def fisher_r(a, b):
    r = np.corrcoef(zscore(a, ddof=1), zscore(b, ddof=1))[0, 1]
    return float(np.arctanh(np.clip(r, -0.999999, 0.999999)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohort", required=True, choices=["dmnelf", "rtbpd"])
    ap.add_argument("--out", required=True)
    a = ap.parse_args(); out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    masker = NiftiLabelsMasker(build_labels(), standardize="zscore_sample", detrend=True,
                               low_pass=0.1, t_r=TR, verbose=0)
    fp = f"{ROOT[a.cohort]}/derivatives/fmriprep_25.2.5_fmap"
    rows = []
    for subdir in sorted(glob.glob(f"{fp}/sub-*/")):
        sub = os.path.basename(subdir.rstrip("/")).replace("sub-", "")
        for ses in SES[a.cohort]:
            for img in sorted(glob.glob(f"{fp}/sub-{sub}/{ses}/func/sub-{sub}_{ses}_task-rest_run-*_space-MNI152NLin6Asym_res-2_desc-preproc_bold.nii.gz")):
                run = int(img.split("run-")[1][:2])
                try:
                    confs, samp = load_confounds(img, strategy=("high_pass", "motion", "wm_csf"),
                                                 motion="full", wm_csf="basic")
                    ts = masker.fit_transform(img, confounds=confs, sample_mask=samp)   # [T,4]
                    mpfc, pcc, ldl, rdl = ts[:, 0], ts[:, 1], ts[:, 2], ts[:, 3]
                    rows.append(dict(cohort=a.cohort, subject=sub, session=ses, run=run,
                                     z_pcc=fisher_r(mpfc, pcc), z_ldlpfc=fisher_r(mpfc, ldl),
                                     z_rdlpfc=fisher_r(mpfc, rdl)))
                except Exception as e:
                    print(f"  {sub} {ses} run-{run}: FAIL {type(e).__name__}: {str(e)[:70]}", flush=True)
        print(f"{a.cohort} {sub}: done ({sum(r['subject']==sub for r in rows)} runs)", flush=True)
    pd.DataFrame(rows).to_csv(out / f"fox_roi_{a.cohort}.csv", index=False)
    print(f"saved fox_roi_{a.cohort}.csv ({len(rows)} runs)", flush=True)


if __name__ == "__main__":
    main()
