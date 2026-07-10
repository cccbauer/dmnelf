# Pre → post resting-state connectivity change (mbNF cohorts)

**Question.** Does a session of DMN–CEN mindfulness neurofeedback (mbNF) shift *resting-state*
functional connectivity, and does any shift track the self-reported CALM change? Tested in both
clinical cohorts: DMNELF (schizophrenia, n=16) and rtBPD (elevated BPD traits; nf1 n=23, nf2 n=13).

## Method
`restfc_extract.py` (cluster, SLURM) → per rest run, fMRIPrep BOLD denoised (motion-full + WM/CSF +
high-pass cosines, low-pass 0.1 Hz, detrend) → three connectivity families:
- **DiFuMo-64 (primary):** within-DMN, within-CEN, DMN–CEN (Yeo-7 labels), plus DMN-hub edges
  MPFC–PCC / ACC–PCC / ACC–MPFC (comps 38 / 3,13 / 22).
- **Personalized NF masks (diagnostic only — see caveat):** within-DMN / within-CEN voxelwise
  coherence and DMN–CEN mask-mean coupling.

Pre/post: DMNELF rest run-01=pre, run-02=post; rtBPD runs 1,2=pre, 3,4=post (mean of each pair).
Paired t-test on post−pre per group (`restfc_prepost.py`).

## Result — no robust neurofeedback-specific change
- **rtBPD (nf1 & nf2): null.** Every DiFuMo network measure is flat pre→post (|d|≤0.32, all p>0.13).
  The strongest was nf1 ACC–MPFC (+0.066, d=0.32, p=0.14) — a non-significant trend, not replicated
  in nf2.
- **DMNELF: within-CEN (+0.107, d=0.96, p=0.002) and DMN–CEN (+0.092, d=0.54, p=0.048) increase**
  from run-01 to run-02. But DMNELF has only **two** rest runs and **no control condition**, so this
  is fully confounded with run order / time-in-session and cannot be attributed to neurofeedback.
- **Connectivity change does NOT track CALM change** (best mpfc_pcc r=0.20, p=0.22; all others ns).

## Discarded — personalized voxelwise "decoherence" is a first-run artifact
The personalized within-network voxelwise pairwise-r measures initially looked striking (within-DMN
and within-CEN both **dropped** pre→post, d≈−1.0 to −1.7, p<0.001 in DMNELF and rtBPD nf1). This is an
artifact, not an effect:
1. It hits **DMN and CEN equally** (not network-specific) → consistent with a shared global-signal term.
2. It is **absent in nf2** (the acclimated second session).
3. Per-run means show **run 1 is an isolated outlier** (within_dmn_pers ≈0.36 in run 1 vs ≈0.12–0.19
   in runs 2–5), i.e. a first-run arousal/motion global-signal inflation, not a pre→post transition.
4. **Sensitivity check (`run1_sensitivity`): dropping run-1 from the pre average reverses the sign**
   (nf1 within-DMN d=−1.32 → +0.54). A genuine effect would not invert.
Voxelwise mean-pairwise-r among raw mask voxels is dominated by shared global signal; these measures
are reported as diagnostic only and are **not** interpreted as neurofeedback effects.

## Bottom line
A single mbNF session produced **no reliable change in resting-state connectivity** in either clinical
cohort, and no connectivity change tracked the calm outcome. This is a clean null that bounds the
manuscript's claims: the positive, replicated signal is the **task-state** EEG f-SNR / PDA decoding and
the calm~PDA clinical anchor — not resting-state reorganization after one session. Any DMNELF within-
CEN / DMN–CEN increase is confounded with run order (2 runs, no control) and should not be over-read.

## Files
- `scripts/restfc_extract.py` — cluster extraction (DiFuMo + personalized, per rest run)
- `scripts/restfc_slurm.sh` — SLURM array driver
- `scripts/restfc_prepost.py` — pre/post stats + run-1 sensitivity + calm link
- `results/restfc_{cohort}_{sub}.csv` — per-subject per-run connectivity (44 subjects)
- `results/restfc_prepost_subject.csv`, `restfc_prepost_stats.csv`, `restfc_calm_link.csv`
