# Fox-2005 MPFC seed connectivity — DMN validation + pre/post change

Two questions with the Fox-2005 seeds (`/projects/swglab/data/rtBPD/analysis/masks/Fox2005`,
2 mm, same affine as the BOLD; used the non-10mm MPFC seed):
1. Does an MPFC seed recover the DMN posterior hub (PCC / precuneus) whole-brain?
2. Does a single mbNF session change the MPFC→PCC coupling (pre vs post rest)?

Denoising identical to restfc (motion-full + WM/CSF-mean + high-pass cosines, low-pass 0.1 Hz,
detrend, z-score; no GSR). Rest-run pre/post: DMNELF run-01=pre/02=post; rtBPD 1,2=pre / 3,4=post.

## 1. Whole-brain MPFC seed-to-voxel → DMN, with PCC/precuneus (n=40, both cohorts)
Per-subject seed-to-voxel Fisher-z maps (`fox_seed.py`, SLURM) → group mean-z + one-sample t
(`fox_group.py`). **The canonical DMN emerges**; 29,804 positive supra-threshold voxels (p<.001)
vs 562 negative. Top positively-coupled Harvard-Oxford regions (group z, t):

| Region | z | t |
|---|---|---|
| Paracingulate / Frontal Medial (anterior DMN) | +0.13 / +0.09 | +4.9 / +3.7 |
| **Cingulate Gyrus, posterior (PCC)** | **+0.083** | **+4.29** |
| **Precuneous Cortex** | **+0.061** | **+2.98** |
| Middle Temporal Gyrus ant/post (lateral DMN) | +0.084 / +0.070 | +4.0 / +3.3 |
| Fox PCC seed ROI | +0.091 | — |

So **yes — PCC and precuneus come up**, along with the full DMN (mPFC, paracingulate, PCC,
precuneus, lateral temporal, angular/frontal pole). Figures: `group_foxz_slices_both.png`,
`group_foxz_glass_both.png`.

**Caveat (honest):** MPFC→DLPFC (task-positive) is ~0 (lDLPFC z=−0.013, rDLPFC +0.007), NOT the
strong Fox anticorrelation. Expected — we did **no global-signal regression** (matching restfc);
anticorrelation magnitude is GSR-dependent. The positive DMN topography is unaffected by that choice.

## 2. Pre → post change in MPFC-seed connectivity (seed-to-seed, `fox_seed_roi.py`/`fox_prepost.py`)
Baseline MPFC–PCC coupling is strong and DMN-like (all-run mean z=+0.48). Paired pre vs post:

| group (n) | MPFC–PCC Δ (dz, p) | MPFC–lDLPFC | MPFC–rDLPFC |
|---|---|---|---|
| DMNELF (16) | −0.024 (−0.11, .67) | +0.041 (.39) | +0.082 (.15) |
| rtBPD nf1 (23) | +0.051 (+0.17, .41) | +0.093 (+0.40, **.068**) | +0.123 (+0.59, **.010**) |
| rtBPD nf2 (13) | −0.011 (−0.05, .87) | +0.009 (.92) | +0.070 (.43) |

- **MPFC–PCC coupling does NOT change pre→post in any group** (all ns), and this null is robust to
  the first-run artifact (run-1 sensitivity: nf1 dz=+0.17 whether pre=runs1,2 or pre=run2 only).
- The only nominally significant edge is rtBPD **nf1 MPFC–rDLPFC becoming less anticorrelated**
  (dz=0.59, p=.010; lDLPFC same direction p=.068). But it is **not replicated in nf2** and would not
  survive correction over the 9 tests → treat as chance, not a neurofeedback effect.
- **Calm link (suggestive):** across sessions, a larger pre→post *reduction* in MPFC–PCC coupling
  goes with a larger CALM increase (r=−0.33, p=.033, n=41) — directionally sensible (less DMN self-
  coupling → calmer) but nominal only (×3 tests → ~.10) and not confirmed by the paired test.

## Bottom line
The MPFC seed cleanly recovers the DMN (PCC + precuneus), validating the network in these data. But
a **single mbNF session does not reliably change the MPFC–PCC (core DMN) coupling** in either clinical
cohort — converging with the DiFuMo restfc null. Resting-state reorganization after one session is not
the story; the replicated signal remains task-state EEG f-SNR / PDA decoding + the calm~PDA anchor.

## Files
- `scripts/fox_seed.py` (whole-brain seed→voxel, SLURM) · `fox_slurm.sh`
- `scripts/fox_group.py` (group mean-z, t-map, HO ROI quantification, figures)
- `scripts/fox_seed_roi.py` (fast seed→ROI per run; merged-label single-read) · `fox_prepost.py`
- `results/group_foxz_{mean,t,t_p001}_both.nii.gz`, `group_foxz_{slices,glass}_both.png`
- `results/group_foxroi_both.csv`, `fox_roi_{dmnelf,rtbpd}.csv`, `fox_prepost_subject.csv`
- `results/seeds/FOX2005_*.nii` (MPFC, PCC, L/R DLPFC)
