#!/usr/bin/env python3
"""
fox_group.py  (local)  —  group Fox-2005 MPFC seed-to-voxel map
---------------------------------------------------------------
Aggregate the per-subject MPFC seed-connectivity z-maps (foxz_*.nii.gz) into a group
result and answer: does the DMN posterior hub (PCC / precuneus) emerge, and do the
task-positive DLPFC seeds show the Fox anticorrelation?

  - group mean Fisher-z map + one-sample t-map (across subjects)
  - threshold (voxelwise p<.001, |t|) and label surviving clusters via Harvard-Oxford
  - quantify mean group-z in: Fox PCC seed, Harvard-Oxford Precuneous, Fox L/R DLPFC
  - glass-brain + slice figure through the PCC/precuneus

Usage: python fox_group.py [--cohort dmnelf|rtbpd|both]
"""
import argparse, glob, warnings, re
from pathlib import Path
import numpy as np, nibabel as nib, pandas as pd
from scipy import stats
warnings.filterwarnings("ignore")
from nilearn import image, datasets, plotting
from nilearn.maskers import NiftiMasker

PROJ = Path(__file__).resolve().parent.parent
RES = PROJ / "results"
SEEDDIR = Path("/tmp")  # seeds pulled locally alongside maps; overridden below if present
FOX_PCC = RES / "seeds" / "FOX2005_PCC.nii"
FOX_LDLPFC = RES / "seeds" / "FOX2005_leftDLPFC.nii"
FOX_RDLPFC = RES / "seeds" / "FOX2005_rightDLPFC.nii"
FOX_MPFC = RES / "seeds" / "FOX2005_MPFC.nii"


def roi_mean(img, roi_path):
    roi = image.resample_to_img(str(roi_path), img, interpolation="nearest")
    m = roi.get_fdata() > 0
    return float(img.get_fdata()[m].mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohort", default="both", choices=["dmnelf", "rtbpd", "both"])
    a = ap.parse_args()
    pat = "foxz_*.nii.gz" if a.cohort == "both" else f"foxz_{a.cohort}_*.nii.gz"
    QA = re.compile(r"dmnelf(999|1\d\d\d)")   # exclude phantom/QA ids
    files = [f for f in sorted(glob.glob(str(RES / pat))) if not QA.search(f)]
    print(f"{len(files)} subject maps ({a.cohort}, QA excluded)")
    if not files:
        return
    ref = nib.load(files[0])
    data = np.stack([nib.load(f).get_fdata() for f in files], 0)   # [S, X,Y,Z]

    # group mean z + one-sample t across subjects (voxelwise)
    meanz = np.nanmean(data, 0)
    with np.errstate(invalid="ignore", divide="ignore"):
        t, p = stats.ttest_1samp(data, 0, axis=0, nan_policy="omit")
    t = np.nan_to_num(np.asarray(t)); p = np.asarray(p)
    meanz_img = nib.Nifti1Image(meanz.astype(np.float32), ref.affine, ref.header)
    t_img = nib.Nifti1Image(t.astype(np.float32), ref.affine, ref.header)
    tag = a.cohort
    meanz_img.to_filename(RES / f"group_foxz_mean_{tag}.nii.gz")
    t_img.to_filename(RES / f"group_foxz_t_{tag}.nii.gz")

    # thresholded t (voxelwise p<.001 two-sided, df=S-1)
    df = len(files) - 1
    tcrit = stats.t.ppf(1 - 0.0005, df)
    tthr = np.where(np.abs(t) >= tcrit, t, 0)
    nib.Nifti1Image(tthr.astype(np.float32), ref.affine, ref.header).to_filename(
        RES / f"group_foxz_t_p001_{tag}.nii.gz")
    print(f"df={df}  t-crit(p<.001 2-sided)={tcrit:.2f}  "
          f"pos supra-thr voxels={(tthr>0).sum()}  neg={(tthr<0).sum()}")

    # ROI quantification on the group mean-z map
    rows = []
    HO = datasets.fetch_atlas_harvard_oxford("cort-maxprob-thr25-2mm")
    ho_img = HO.maps if isinstance(HO.maps, nib.Nifti1Image) else nib.load(HO.maps)
    ho = image.resample_to_img(ho_img, meanz_img, interpolation="nearest").get_fdata()
    labels = HO.labels
    precu_idx = [i for i, l in enumerate(labels) if "Precuneous" in l]
    for name, path in [("Fox_PCC", FOX_PCC), ("Fox_lDLPFC", FOX_LDLPFC),
                       ("Fox_rDLPFC", FOX_RDLPFC), ("Fox_MPFC(seed)", FOX_MPFC)]:
        if path.exists():
            rows.append(dict(roi=name, mean_z=roi_mean(meanz_img, path)))
    for i in precu_idx:
        m = ho == i
        rows.append(dict(roi=f"HO:{labels[i]}", mean_z=float(meanz[m].mean())))
    # top positive & negative HO cortical regions by group mean-z (excluding seed vicinity)
    reg = []
    for i, l in enumerate(labels):
        if i == 0:
            continue
        m = ho == i
        if m.sum() >= 20:
            reg.append((l, float(meanz[m].mean()), float(t[m].mean()), int(m.sum())))
    reg.sort(key=lambda x: x[1], reverse=True)
    print("\n=== ROI mean group-z (MPFC seed connectivity) ===")
    df_roi = pd.DataFrame(rows)
    print(df_roi.to_string(index=False))
    df_roi.to_csv(RES / f"group_foxroi_{tag}.csv", index=False)
    print("\n=== top +connected Harvard-Oxford cortical regions ===")
    for l, z, tt, n in reg[:8]:
        print(f"  {l:42s} z={z:+.3f} t={tt:+.2f} (n={n})")
    print("=== top -connected (anticorrelated) ===")
    for l, z, tt, n in reg[-6:]:
        print(f"  {l:42s} z={z:+.3f} t={tt:+.2f} (n={n})")

    # figures
    try:
        disp = plotting.plot_glass_brain(t_img, threshold=tcrit, colorbar=True,
                                         plot_abs=False, display_mode="lyrz",
                                         title=f"Fox MPFC seed-to-voxel t (p<.001), {tag}")
        disp.savefig(RES / f"group_foxz_glass_{tag}.png", dpi=150); disp.close()
        d2 = plotting.plot_stat_map(t_img, threshold=tcrit, display_mode="z",
                                    cut_coords=[36, 28, 20, 12], colorbar=True,
                                    title=f"MPFC seed t (p<.001), {tag} — PCC/precuneus")
        d2.savefig(RES / f"group_foxz_slices_{tag}.png", dpi=150); d2.close()
        print(f"\nsaved figures group_foxz_glass_{tag}.png / group_foxz_slices_{tag}.png")
    except Exception as e:
        print("figure failed:", e)


if __name__ == "__main__":
    main()
