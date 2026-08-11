# Resting-state functional connectivity (restfc_connectivity)

Pre/post resting-state fMRI connectivity for both cohorts (DMNELF ses-dmnelf;
rtBPD ses-nf1/ses-nf2), run directly on the HPC cluster
(`explorer.northeastern.edu`, `/projects/swglab/data/{DMNELF,rtBPD}`).

## Scripts

- `restfc_extract.py` (+ `restfc_slurm.sh`, `restfc_dmnelf_subs.txt`,
  `restfc_rtbpd_subs.txt`): DiFuMo-64/Yeo-7 within-DMN, within-CEN, DMN↔CEN,
  and core hub edges (MPFC-PCC, ACC-PCC, ACC-MPFC), plus personalized
  neurofeedback-mask DMN/CEN connectivity. Usage:
  `python restfc_extract.py --cohort {dmnelf,rtbpd} --subject SUB`.
- `fox_seed.py` / `fox_seed_roi.py` (+ `fox_slurm.sh`, `fox_roi_slurm.sh`):
  Fox-2005 MPFC seed-to-voxel connectivity (PCC / L/R DLPFC targets),
  averaged across a subject's rest runs. Per-voxel z-maps (`.nii.gz`) are
  cluster-only (gitignored); only the per-subject/per-ROI CSV summaries are
  tracked here.

## Results

- `results/restfc/restfc_{cohort}_{subject}.csv` — per-run connectivity
  values, one row per rest run.
- `results/foxseed/foxroi_{cohort}_{subject}.csv` +
  `fox_roi_{cohort}.csv` — per-subject mean z within the Fox PCC/DLPFC seeds.

No group-level pre-vs-post stats have been computed yet on this fMRI side
(unlike the EEG PLV analysis in `../rtbpd_replication/connectivity_prepost/`,
which already has paired-stats + FDR correction) — that's the natural next
step once this is reviewed.
