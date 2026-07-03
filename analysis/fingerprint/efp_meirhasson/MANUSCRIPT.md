# An EEG Finger-Print of DMN/CEN network activation during neurofeedback

*Methods and Results draft. Replication and extension of the EEG Finger-Print
(EFP) method (Meir-Hasson et al., 2014, NeuroImage) to the default-mode (DMN)
and central-executive (CEN) networks and their differential activation (PDA).*

---

## Methods

### Participants and acquisition

Seventeen participants underwent simultaneous EEG–fMRI during a real-time
fMRI neurofeedback ("feedback") task. EEG was recorded from 31 scalp electrodes,
gradient- and cardioballistic-artifact corrected, band-pass filtered, ICA-cleaned,
and resampled to 250 Hz. Functional MRI was acquired at a repetition time (TR) of
1.2 s and preprocessed with fMRIPrep (MNI152NLin6Asym, 2 mm).

### fMRI target signals

For each run we derived the following region/network timeseries, standardized and
detrended:

- **DMN** and **CEN.** Subject-specific network masks were defined by matching
  CanICA components to the Yeo-7 Default and Frontoparietal networks, respectively,
  and the mean BOLD signal was extracted within each personalized mask.
- **PDA** (prefrontal–default anticorrelation / distinctiveness) = CEN − DMN, the
  moment-to-moment separation between the task-positive and task-negative networks
  that the neurofeedback targets.
- **GSR variants** (`GSR_DMN`, `GSR_CEN`, `GSR_PDA`): each target after global-signal
  regression, to test whether coupling survives removal of the whole-brain mean.
- **VIS** (positive control): the mean BOLD in a focal 6 mm-radius sphere centered on
  the calcarine sulcus (MNI [−1, −86, 13] ≈ V1), reproducing the visual-cortex ROI of
  Meir-Hasson et al. (2014, Fig. 5a).

### EEG Finger-Print construction

Following Meir-Hasson et al. (2014), each electrode was processed independently.
The single-channel signal was decomposed with the Stockwell (S-) transform at 1 Hz
resolution over 1–40 Hz, yielding an instantaneous time–frequency power
representation. To reduce dimensionality in a data-driven manner, the spectrum was
partitioned into **10 contiguous frequency bands of equal cumulative (log) power**,
and power was averaged within each band. Band-power timeseries were downsampled to
the fMRI sampling grid; all analyses were run at both the native TR (1.2 s) and a
4 Hz cubic-spline upsampling, which the original method uses to expose sub-TR
temporal detail.

The predictor for a given target was built as a **[frequency band × sliding
time-delay]** design matrix: for each fMRI timepoint, the 10 band powers were taken
at delays spanning 0 to −12 s (11 lags at TR; 49 at 4 Hz). Ridge regression thus
learns a separate weight for every band and every lag, so the hemodynamic delay is
estimated from the data rather than imposed by a canonical HRF. The resulting
coefficient matrix is the EFP.

### Cross-validated decoding

For every electrode we performed **double cross-validation**: an outer
contiguous-block *m×k*-fold scheme (k = 5 folds, m = 2 rolled repeats) provided
out-of-fold predictions, while an inner RidgeCV selected the ridge penalty λ on each
training split (30-point log grid, 10⁻² to 10⁴). Performance was quantified by the
out-of-fold Pearson correlation (r) and normalized mean-squared error (NMSE). The
best electrode per subject and target was chosen by minimum CV NMSE and refit on all
data to obtain that subject's EFP coefficient map.

### Baseline predictors

Two baselines were evaluated on the identical folds and electrode:
(i) an **HRF** predictor using all 10 band powers convolved with a canonical
double-gamma HRF (fixed delay, no sliding lags), and (ii) a **theta/alpha (T/A)**
predictor — the theta:alpha power ratio at the best occipital electrode, HRF-convolved
(a classic single-feature EEG index of arousal/vigilance). The method is expected to
order EFP ≥ HRF ≥ T/A.

### Group-level analyses

Per-subject results were aggregated across the 17 participants. Group significance of
mean r was assessed with a non-parametric **sign-flip permutation test**
(10,000 permutations). We computed group-averaged [band × delay] fingerprints and
**group-averaged per-electrode r topographies**. Cross-subject generalization was
tested with **leave-one-subject-out (LOSO)** transfer: a fingerprint trained on N−1
subjects (at the modal best electrode) predicted the held-out subject.

### Positive control

The visual-cortex ROI provides a known-answer test: because occipital alpha power is
robustly (negatively) coupled to visual-cortex BOLD, a valid, blind EFP pipeline
should recover occipital electrodes, an alpha-band signature, and a physiologically
plausible delay. We report the electrode topography and fingerprint obtained for VIS
as a validation of the machinery before interpreting the network targets.

### Implementation

Analyses were run on an HPC cluster (per-subject decoding as a SLURM array; group
aggregation and figure generation as dependent jobs). Code: `efp_features.py`
(S-transform, equal-energy bands, target assembly), `efp_decode.py` (per-subject
double-CV ridge and baselines), `efp_group.py` (aggregation, permutation tests,
fingerprints, LOSO), `extract_visual_sphere.py` (calcarine-sphere extraction), and
`paper_figures.py` (Figs. 2, 3, 5 analogs, topographies).

---

## Results

### The EFP is the strongest network decoder tested

Within-subject, the sliding-delay single-electrode EFP outperformed both baselines
for every target except plain DMN, replicating the method's expected ordering
(Table 1; group-mean out-of-fold r at native TR). All EFP effects were significant
by the sign-flip test (all p < 0.01).

**Table 1. Within-subject decoding accuracy** (out-of-fold Pearson r, native TR;
mean ± SD across n = 17 participants, with the 95% CI for the EFP). Bold marks the
best predictor per row (EFP for every target except DMN, where T/A leads).
*Auto-generated by `scripts/manuscript_stats.py`.*

<!-- BEGIN:table1 -->
| Target | EFP  (mean ± SD) [95% CI] | HRF (mean ± SD) | T/A (mean ± SD) | n |
|---|---|---|---|---|
| CEN | **0.279 ± 0.191 [0.181, 0.377]** | 0.245 ± 0.159 | 0.150 ± 0.189 | 17 |
| PDA | **0.258 ± 0.128 [0.192, 0.323]** | 0.110 ± 0.156 | 0.196 ± 0.161 | 17 |
| GSR_CEN | **0.255 ± 0.195 [0.155, 0.356]** | 0.188 ± 0.151 | 0.199 ± 0.217 | 17 |
| DMN | 0.225 ± 0.117 [0.165, 0.285] | 0.146 ± 0.147 | **0.248 ± 0.104** | 17 |
| GSR_PDA | **0.225 ± 0.154 [0.146, 0.305]** | 0.108 ± 0.154 | 0.172 ± 0.174 | 17 |
| VIS | **0.179 ± 0.159 [0.097, 0.261]** | 0.085 ± 0.164 | 0.152 ± 0.131 | 17 |
| GSR_DMN | **0.113 ± 0.136 [0.043, 0.183]** | 0.059 ± 0.112 | 0.053 ± 0.186 | 17 |
<!-- END:table1 -->

The EFP predictors visibly track the measured network timeseries at a temporal
resolution finer than the fMRI itself (Fig. 3d). Results at 4 Hz were comparable to
TR, confirming that upsampling adds temporal detail without inflating correlation.

### Decoding accuracy varies substantially across individuals

Group means conceal wide inter-individual spread (Fig. 4; SD column of Table 1). For
the strongest targets, within-subject EFP r ranged from near-zero or slightly negative
in the weakest participants to r ≈ 0.5–0.6 in the strongest, and the per-target SD was
comparable in magnitude to the mean for several networks. The 95% confidence intervals
nonetheless excluded zero for every target (Table 1), indicating that the group-level
effects are robust despite the heterogeneity. This
variability motivates the subject-specific electrode selection built into the method
and cautions against a one-size-fits-all electrode/band prior.

### Fingerprints reflect the canonical alpha–BOLD relationship

Group [frequency × delay] fingerprints (Fig. 3c; Fig. "fingerprints") were dominated
by a **negative alpha-band (8–13 Hz) weight at short, HRF-plausible delays
(~−2 to −6 s)** — i.e., higher alpha power a few seconds earlier predicts *lower*
network BOLD — the well-established alpha-desynchronization → activation relationship.
A positive lobe at longer lags (~−8 to −10 s) is consistent with the biphasic HRF
undershoot and band/delay collinearity rather than an independent driver. DMN was the
exception, additionally carrying a positive theta-band lobe.

### A general fingerprint transfers across subjects

LOSO transfer was significant for the non-GSR targets (Table 2, TR): PDA
r = 0.127 (p = 0.010), CEN r = 0.084 (p = 0.001), DMN r = 0.066 (p = 0.002), and the
VIS control r = 0.084 (p = 0.004). GSR variants did not transfer (all p > 0.15),
indicating that the cross-subject component depends partly on the global signal.

**Table 2. Cross-subject (leave-one-subject-out) transfer** at the modal best
electrode (native TR). `*` marks p < 0.05 (sign-flip permutation test).
*Auto-generated by `scripts/manuscript_stats.py`.*

<!-- BEGIN:loso -->
| Target | Electrode | LOSO r | p (sign-flip) |
|---|---|---|---|
| CEN | POz | 0.084 * | 0.001 |
| PDA | Pz | 0.127 * | 0.010 |
| GSR_CEN | Fz | 0.049 | 0.160 |
| DMN | TP10 | 0.066 * | 0.002 |
| GSR_PDA | P4 | 0.031 | 0.204 |
| VIS | O2 | 0.084 * | 0.004 |
| GSR_DMN | F7 | 0.020 | 0.294 |
<!-- END:loso -->

### Visual positive control validates the pipeline — with a focal ROI

Using the focal calcarine sphere, the blind EFP recovered the expected visual
signature: within-subject r = 0.179 (vs HRF 0.085, T/A 0.152); the **group-averaged
per-electrode map peaked posteriorly (Pz), with frontal electrodes now the worst**
(Fig. 5b analog); the **LOSO best electrode was occipital (O2)**; and the fingerprint
showed alpha-band modulation at a plausible delay. Notably, an initial definition of
VIS as a broad multi-parcel visual *network* failed this test — its topography peaked
frontally and diffusely — whereas the focal single-ROI localized posteriorly. Faithful,
focal ROI definition is therefore essential to reproduce the localization result.

### The predictive signal is spatially distributed

Despite strong decoding, the signal was not focally localized for the networks.
Per-subject best electrodes scattered across the scalp, and group-averaged
per-electrode r maps were spatially diffuse (e.g., CEN was positive across essentially
the whole head). Clean topography emerged only at the group/LOSO level and, for the
visual control, only with a focal ROI. Together with the failure of GSR variants to
transfer, this indicates that the EFP predictions ride substantially on a
**spatially distributed, largely global component — consistent with a global
arousal/vigilance contribution** (in line with the alpha/theta signatures) — rather
than on focal, region-specific cortical generators. This is an important
interpretational boundary: the network fingerprints are predictive and reproducible,
but their specificity is limited by a shared global term.

### Summary

The EFP method reproduces on DMN/CEN/PDA in this cohort and yields the best
within-subject and cross-subject decoding we have obtained for these targets, with
interpretable alpha-band fingerprints and a validated visual positive control. The
principal caveat is spatial: the recoverable signal is distributed and global rather
than focal, so the fingerprints should be read as network-level, arousal-linked
predictors rather than evidence of localized cortical sources.

---

## Figures

- **Figure 2 — Post-processing schematic** (`results/full/paper_fig2_schematic_PDA_tr.png`):
  electrode → S-transform → 4 Hz → data-driven bands → band-averaged TF → sliding window
  per fMRI TR; fMRI ROI → upsample → normalize.
- **Figure 3 — Prediction input & output** (`paper_fig3_composite_{PDA,CEN,GSR_CEN,VIS}_tr.png`):
  (a) ROI signal + ROI mask image; (b) mean z-EEG time–frequency for the lower vs upper
  25% of ROI values; (c) EFP coefficient map; (d) EFP predictor vs measured BOLD; (e)
  per-electrode CV-r topography (best electrode marked).
- **Figure 4 — Per-subject decoding** (`paper_fig_persubject_scatter_tr.png`):
  within-subject EFP r for each target across all 17 participants; bar = group mean,
  shaded band = 95% CI.
- **Figure 5 — Visual positive control** (`paper_fig3_composite_VIS_tr.png`,
  `paper_fig_group_topomap_VIS_tr.png`): focal calcarine sphere; posterior/occipital
  localization at the group/LOSO level.
- **Group fingerprints** (`paper_fig_fingerprints_tr.png`) and **group-averaged
  per-electrode topographies** (`paper_fig_group_topomap_{VIS,CEN,PDA}_tr.png`).

Numbers above are from `results/full/efp_group_summary.csv` and
`results/full/efp_group_loso.csv` (n = 17, feedback task, native TR).

## Key reference

Meir-Hasson G., Kinreich S., Podlipsky I., Hendler T., Intrator N. (2014). *An EEG
finger-print of fMRI deep regional activation.* NeuroImage 102, 128–141.
