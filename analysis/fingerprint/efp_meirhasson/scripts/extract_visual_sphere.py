#!/usr/bin/env python3
"""
extract_visual_sphere.py
------------------------
Extract the visual-cortex positive-control ROI timeseries — a single 6 mm sphere
at the calcarine (MNI [-1,-86,13] ~ V1), faithful to Meir-Hasson 2014 Fig 5a.

For each subject/run: load the fMRIPrep preproc BOLD (MNI152NLin6Asym res-2),
apply a NiftiSpheresMasker (standardize + detrend, matching how the DMN/CEN mask
timeseries are extracted), and store the (T,) timeseries per run.

Output: results/visual_sphere/{sub}.npz  (keys = run ids -> (T,) float32)

Run on the cluster (needs BOLD access):
    python extract_visual_sphere.py --group
"""
import argparse
from pathlib import Path
import numpy as np
from nilearn.maskers import NiftiSpheresMasker

from efp_features import load_config, run_ids, PROJ_DIR


def bold_path(cfg, sub, run):
    d = cfg["data"]
    # confounds_dir is the fMRIPrep derivatives root (also holds preproc BOLD)
    return (Path(d["confounds_dir"]) / f"sub-{sub}" / d["session"] / "func" /
            f"sub-{sub}_{d['session']}_task-{d['task']}_run-{int(run):02d}"
            f"_space-MNI152NLin6Asym_res-2_desc-preproc_bold.nii.gz")


def extract_subject(cfg, sub, out_dir):
    d = cfg["data"]; tr = d["fmri"]["tr"]
    seed = [tuple(d["fmri"]["vis_sphere_mni"])]
    radius = float(d["fmri"]["vis_sphere_radius_mm"])
    masker = NiftiSpheresMasker(seeds=seed, radius=radius, standardize=True,
                                detrend=True, t_r=tr, allow_overlap=True)
    out = {}
    for run in run_ids(cfg, sub):
        bp = bold_path(cfg, sub, run)
        if not bp.exists():
            print(f"  {sub} run {run}: BOLD missing ({bp.name})")
            continue
        ts = masker.fit_transform(str(bp))[:, 0].astype(np.float32)   # (T,)
        out[str(run)] = ts
        print(f"  {sub} run {run}: T={len(ts)}")
    if out:
        out_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(out_dir / f"{sub}.npz", **out)
        print(f"  saved {out_dir / f'{sub}.npz'}  ({len(out)} runs)")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subjects", nargs="+", default=None)
    ap.add_argument("--group", action="store_true")
    args = ap.parse_args()
    cfg = load_config()
    subs = (cfg["data"]["subjects"]["all"] if args.group
            else (args.subjects or cfg["data"]["subjects"]["pilot"]))
    out_dir = PROJ_DIR / "results" / "visual_sphere"
    for sub in subs:
        print(f"[{sub}]")
        extract_subject(cfg, sub, out_dir)


if __name__ == "__main__":
    main()
