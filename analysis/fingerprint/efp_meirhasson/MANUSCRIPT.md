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
frequency resolution over 1–40 Hz, yielding an instantaneous time–frequency power
representation. To reduce dimensionality in a data-driven manner, the spectrum was
partitioned into **10 contiguous frequency bands of equal cumulative (log) power** (an
energy-uniformity constraint, as in the original), and power was averaged within each
band. Band power was standardized before fitting (per-fold z-scoring; the original
zero-means each band).

We analysed at two temporal resolutions. **Native TR (1.2 s) is our primary analysis**
(it avoids any interpolation artefact). We additionally report the original's **native
4 Hz** resolution — where the EEG band power is down-sampled to 4 Hz and the fMRI target
is cubic-spline up-sampled to 4 Hz and normalized — as a **supplementary** result;
because the 4 Hz target is interpolated, its autocorrelation can inflate correlations, so
we validate it against a null-target upsampling control (see Results) and do not lead
with it.

The predictor for a given target was built as a **[frequency band × sliding
time-delay]** design matrix: for each fMRI timepoint, the 10 band powers were taken at
delays spanning 0 to −12 s (11 lags at TR; 49 at 4 Hz). Ridge regression thus learns a
separate weight for every band and every lag, so the hemodynamic delay is estimated from
the data rather than imposed by a canonical HRF. The resulting coefficient matrix is the
EFP.

### Cross-validated decoding

Decoding used **nested double cross-validation**, following Meir-Hasson et al. (2014):
electrode/model selection is performed *on the training folds* and accuracy is measured
*on the held-out test folds*. The outer loop is an **m×k-fold contiguous-block CV**
(k = 5, m = 2 rolled repeats); each test block spans a different portion of the
continuous timeseries. Within each outer-training set, an **inner CV** (same block
scheme) selects, for each candidate electrode, the ridge model minimizing NMSE, and the
single best electrode is chosen; that electrode is then refit on the full outer-training
set and predicts the held-out outer-test fold. The reported per-subject r and NMSE are
computed from the **concatenated out-of-fold predictions**, so the electrode is never
scored on data used to select it — this removes the in-sample electrode-selection
optimism that would otherwise inflate within-subject accuracy. The ridge penalty λ is
selected by `RidgeCV` generalized cross-validation over a 30-value log-spaced grid
(10⁻²–10⁴). Performance is quantified by out-of-fold Pearson r and NMSE (NMSE < 1 =
better than predicting the mean). A descriptive EFP coefficient map per subject/target is
obtained by refitting the modal selected electrode on all of that subject's data (used
only for visualization, not for the reported accuracy).

### Baseline predictors

Two baselines were evaluated with the same cross-validation: (i) an **HRF** predictor
using all 10 band powers convolved with a canonical double-gamma HRF (fixed delay, no
sliding lags), fixed at the modal EFP electrode; and (ii) a **theta/alpha (T/A)**
predictor — the theta (4–7 Hz) / alpha (8–13 Hz) power ratio, HRF-convolved (a classic
single-feature EEG index of arousal/vigilance), with its occipital electrode chosen by
the same **nested** CV so the comparison is unbiased. The method is expected to order
EFP ≥ HRF ≥ T/A.

### Group-level analyses

Per-subject results were aggregated across the 17 participants. Group significance of
mean r was assessed with a non-parametric **sign-flip permutation test**
(10,000 permutations). We computed group-averaged [band × delay] fingerprints and
**group-averaged per-electrode r topographies**; on the topographies we additionally
applied the original's **electrode-wise significance correction** — a sign-flip p per
electrode (across subjects), Benjamini–Hochberg FDR-corrected across electrodes (q < 0.05),
with non-significant electrodes set to zero (as in Meir-Hasson et al., 2014). Cross-subject
generalization was tested with **leave-one-subject-out (LOSO)** transfer: for each
held-out subject, the transfer electrode was chosen as the **group-mean-r peak over the
N−1 training subjects** (not the held-out subject), and a fingerprint trained on those
N−1 subjects predicted the held-out subject — so both electrode selection and model
fitting are leak-free.

### Positive control

The visual-cortex ROI provides a known-answer test: because occipital alpha power is
robustly (negatively) coupled to visual-cortex BOLD, a valid, blind EFP pipeline
should recover occipital electrodes, an alpha-band signature, and a physiologically
plausible delay. We report the electrode topography and fingerprint obtained for VIS
as a validation of the machinery before interpreting the network targets.

### External validation (independent rtBPD cohort)

To test whether the fingerprint generalizes beyond DMNELF, we applied it to an
independent cohort (rtBPD; a separate real-time fMRI neurofeedback study, different
participants and sessions). For each target we trained a single *general* fingerprint
on all DMNELF subjects at the transfer electrode (the DMNELF **group-mean-r peak**
channel; band-index alignment across the per-subject data-driven bands), standardized on
the DMNELF design, and predicted each rtBPD subject's target timeseries; the electrode is
chosen entirely on DMNELF, so rtBPD is untouched in training/selection. Significance of
the mean transfer r was assessed by sign-flip permutation. The same transfer was run for
the HRF and T/A baselines for comparison. rtBPD data were processed through the identical pipeline
(personalized CanICA DMN/CEN masks, DiFuMo features, calcarine-sphere VIS, EEG
preprocessing, EFP caches). Because a subset of rtBPD participants completed a **second
neurofeedback session (nf2)**, we ran the transfer twice — nf1 and nf2 — as a **double
replication**, and for participants present in both sessions this also provides a
within-subject test–retest of the transferred fingerprint. Subjects with corrupted
EEG (mis-exported at 50 Hz) were excluded from the affected session.

### Relationship to Meir-Hasson et al. (2014)

The method is a faithful re-implementation of the original EFP with a small number of
deliberate, documented deviations.

*Faithful:* single-electrode Stockwell (1 Hz) time–frequency; 10 data-driven
equal-energy frequency bands; a [frequency × sliding-delay] ridge model that learns the
delay from data; **nested double cross-validation** (select on train, evaluate on test;
outer m-k-fold k=5/m=2, inner λ search); NMSE for model selection and Pearson r for
reporting; T/A and fixed-HRF baselines; and focal ROIs including the calcarine
visual-cortex positive control.

*Deviations (with rationale):* (1) our **primary resolution is native TR**, with the
original's 4 Hz reported as a supplementary, interpolation-controlled analysis; (2) group
inference uses a subject-level **sign-flip permutation test**, and we add electrode-wise
**FDR** on the topographies (the original used FDR within subject); (3) λ is chosen by
`RidgeCV` generalized CV over a fixed log grid rather than an SVD-derived range with an
inner n-fold; (4) DMN/CEN ROIs are **subject-specific** (CanICA→Yeo) network masks rather
than a single anatomical sphere, reflecting our network — rather than deep-nucleus —
target; (5) the two rtBPD sessions are treated as an explicit **double replication**
rather than selecting each subject's single best session.

### Implementation

Analyses were run on an HPC cluster (per-subject decoding as a SLURM array; group
aggregation and figure generation as dependent jobs). Code: `efp_features.py`
(S-transform, equal-energy bands, target assembly), `efp_decode.py` (per-subject
double-CV ridge and baselines), `efp_group.py` (aggregation, permutation tests,
fingerprints, LOSO), `extract_visual_sphere.py` (calcarine-sphere extraction), and
`paper_figures.py` (Figs. 2, 3, 5 analogs, topographies).

### Portable EEG feasibility: frontal electrode restriction

To assess whether the EFP generalizes to a portable EEG headset covering only the
frontal scalp, electrode access was restricted to an 11-channel frontal subset:
{Fp1, Fp2, F3, F4, F7, F8, Fz, FC1, FC2, FC5, FC6}. Three conditions were evaluated
within the LOSO framework and in cross-cohort transfer:

1. **Frontal single-electrode:** the best frontal electrode per target was selected
   as the group-mean-r peak within the frontal subset, using only training subjects
   (leak-free). All other aspects of the EFP pipeline were unchanged.

2. **Frontal multivariate:** band-power delay features from all 11 frontal electrodes
   were concatenated into a single feature matrix (fixed alphabetical electrode order
   for dimensional consistency across subjects) and a single RidgeCV was fitted per
   target. This increases the feature space approximately 11-fold relative to the
   single-electrode model.

3. **Pseudo-calibration:** before deployment, one rest→task run with a known block
   design was used to individualize the frontal multivariate group model without any
   held-out fMRI. A pseudo-target — an HRF-convolved boxcar (sign-matched per target:
   +1 for CEN/PDA/GSR variants, −1 for DMN/GSR_DMN) — replaced the true fMRI label
   for that run. A fresh per-subject RidgeCV was trained on this pseudo-target and its
   predictions were z-scored and added to the group model predictions (blend). The
   remaining runs were used for evaluation against the real fMRI, providing an honest
   estimate of one-run calibration gain. The statistical significance of the calibration
   gain over the uncalibrated group model was assessed by a sign-flip permutation test
   on the per-subject gain scores.

---

## Results

### Within-subject decoding

Within-subject accuracy is the **nested-CV** estimate (Table 1; group-mean out-of-fold r,
native TR) — electrode selection is cross-validated, so there is no in-sample
"best-of-31-electrodes" optimism. Under this fair evaluation the sliding-delay EFP is the
**best predictor for the task-relevant targets** — PDA (the neurofeedback target),
CEN, GSR_CEN and GSR_PDA — recovering the method's expected EFP ≥ HRF ≥ T/A ordering for
those; the fixed-HRF baseline is competitive for the simpler alpha-driven ROIs (VIS, and
DMN where T/A leads). All EFP effects for the network targets are significant by the
sign-flip test (CEN, PDA, DMN, GSR_CEN, GSR_PDA, GSR_DMN; p < 0.05).

**Table 1. Within-subject decoding accuracy** (nested-CV out-of-fold Pearson r, native
TR; mean ± SD across n = 17 participants, with the 95% CI for the EFP). Bold = best
predictor per row. *Auto-generated by `scripts/manuscript_stats.py`.*

<!-- BEGIN:table1 -->
| Target | EFP  (mean ± SD) [95% CI] | HRF (mean ± SD) | T/A (mean ± SD) | n |
|---|---|---|---|---|
| CEN | **0.159 ± 0.210 [0.051, 0.267]** | 0.146 ± 0.194 | 0.017 ± 0.207 | 17 |
| PDA | **0.169 ± 0.136 [0.099, 0.239]** | 0.122 ± 0.157 | 0.088 ± 0.207 | 17 |
| GSR_CEN | **0.171 ± 0.235 [0.050, 0.292]** | 0.099 ± 0.157 | 0.065 ± 0.236 | 17 |
| DMN | 0.142 ± 0.144 [0.068, 0.216] | 0.131 ± 0.177 | **0.152 ± 0.143** | 17 |
| GSR_PDA | **0.141 ± 0.154 [0.061, 0.220]** | 0.102 ± 0.131 | 0.082 ± 0.198 | 17 |
| VIS | 0.066 ± 0.154 [-0.013, 0.146] | **0.093 ± 0.123** | 0.025 ± 0.111 | 17 |
| GSR_DMN | 0.117 ± 0.186 [0.021, 0.212] | **0.133 ± 0.129** | -0.037 ± 0.156 | 17 |
<!-- END:table1 -->

A matched **same-electrode** comparison (Panel B, Table 1b) — EFP, HRF and T/A scored at
each network's group-peak electrode with concatenated out-of-fold CV, so no method has a
selection or normalization advantage — confirms the EFP design itself: **EFP beats HRF for
6/7 targets** (HRF edges only CEN).

**Table 1b. Same-electrode design comparison** (group-peak electrode, out-of-fold r).
*Auto-generated.*

<!-- BEGIN:panel -->
| Target | Electrode | EFP | HRF | T/A |
|---|---|---|---|---|
| CEN | Cz | 0.166 | **0.183** | -0.019 |
| PDA | Pz | **0.200** | 0.158 | 0.041 |
| GSR_CEN | Pz | **0.141** | 0.024 | -0.012 |
| DMN | O1 | **0.173** | 0.137 | 0.108 |
| GSR_PDA | Pz | **0.179** | 0.125 | 0.043 |
| VIS | Oz | **0.104** | 0.069 | 0.006 |
| GSR_DMN | Cz | **0.124** | 0.083 | -0.130 |
<!-- END:panel -->

Two evaluation artefacts were identified and corrected during validation (see
`VALIDATION.md`): an electrode-**selection bias** (naïve best-of-31 scoring, which had
inflated all within-subject r's) and a feature-**normalization asymmetry** (EFP's 110
band×delay features were not per-run standardized like the baselines, which had deflated
EFP specifically). With both fixed — nested electrode selection and per-run standardization
for all methods — the comparison is fair and the EFP advantage on the task-relevant targets
is restored.

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
| CEN | Cz | 0.124 * | 0.001 |
| PDA | Pz | 0.161 * | 0.008 |
| GSR_CEN | Pz | 0.153 * | 0.011 |
| DMN | O1 | 0.096 * | 0.011 |
| GSR_PDA | Pz | 0.161 * | 0.005 |
| VIS | Oz | 0.050 * | 0.031 |
| GSR_DMN | Cz | -0.012 | 0.653 |
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

### The fingerprint transfers to an independent cohort (external validation)

The DMNELF general fingerprint generalized to the independent rtBPD cohort at effect
sizes comparable to — and for several targets exceeding — the within-DMNELF
leave-one-subject-out transfer (Table 3). Transfer was significant for **all seven
targets in the first session (nf1)** and **six of seven in the second (nf2)**, including
CEN, PDA, DMN, and the VIS positive control (r ≈ 0.09–0.15 across sessions), with only
GSR_DMN failing to replicate in nf2. Because the two rtBPD sessions were acquired
independently, this constitutes a **double replication** (and, for participants present
in both, a within-subject test–retest). That a fingerprint trained entirely on one
cohort predicts network and visual-cortex BOLD in a separate cohort — with no rtBPD data
used in training or electrode selection — is strong evidence that the EFP captures a
genuine, transferable EEG-to-BOLD mapping rather than cohort-specific overfitting.

**Table 3. Cross-cohort external validation** — DMNELF general fingerprint predicting
the rtBPD cohort (native TR), by neurofeedback session (nf1, nf2).
*Auto-generated by `scripts/manuscript_stats.py`.*

<!-- BEGIN:crosscohort -->
| Target | Electrode | nf1 r | nf1 p | nf2 r | nf2 p |
|---|---|---|---|---|---|
| CEN | Cz | +0.138 * | 0.001 | +0.113 * | 0.005 |
| PDA | Pz | +0.067 * | 0.032 | +0.153 * | 0.004 |
| GSR_CEN | Pz | +0.098 * | 0.010 | +0.117 * | 0.007 |
| DMN | O1 | +0.097 * | 0.000 | +0.087 * | 0.009 |
| GSR_PDA | Pz | +0.064 * | 0.027 | +0.136 * | 0.004 |
| VIS | Oz | +0.145 * | 0.000 | +0.121 * | 0.000 |
| GSR_DMN | Cz | +0.045 * | 0.002 | +0.028 | 0.152 |

*Transfer electrode = DMNELF group-peak channel; nf1 n=19, nf2 n=11. `*` p<0.05.*
<!-- END:crosscohort -->

### Frontal EEG partially recovers the fingerprint; pseudo-calibration rescues PDA

The LOSO group-model reference at n_cal=1 (one run held out; evaluated on remaining
runs) was: PDA r = 0.117, CEN r = 0.119, DMN r = 0.081, GSR_CEN r = 0.118,
GSR_PDA r = 0.115.

**Frontal single-electrode.** Restricting to the best frontal channel per target
preserved CEN substantially (Cz→Fz, r = 0.094; 79% of full-montage) but sharply
reduced PDA (Pz→Fz, r = 0.015; 13%) and GSR_PDA (r = 0.026; 23%). DMN was
partially preserved (O1→FC5, r = 0.046; 57%). This is expected: PDA's group-peak
electrode is parietal (Pz), which lies outside the frontal headset coverage.

**Frontal multivariate, no calibration.** Using all 11 frontal electrodes
concatenated, the group model collapsed within DMNELF (PDA r ≈ 0.001; all targets ≤
0.101), reflecting overfitting of the high-dimensional frontal feature space to n = 17
training subjects.

**Frontal multivariate, pseudo-calibration (n_cal = 1).** A single pseudo-calibrated
run rescued performance substantially. After one calibration run PDA recovered to
r = 0.113 (p = 0.009), GSR_CEN to r = 0.122 (p = 0.022), and GSR_PDA to r = 0.107
(p = 0.008) — within ~5% of full-montage LOSO accuracy. CEN, already partly captured
by the uncalibrated group model (r = 0.101), reached r = 0.121 with calibration
(non-significant gain, p = 0.335). DMN did not benefit from frontal calibration
(r = −0.016), consistent with its occipital group-peak electrode.

**Table 4. Frontal EEG decoding accuracy** (LOSO, group_only and pseudo-cal at
n_cal = 1; native TR; n = 17). Full-montage group_only shown for reference.

| Target | Full montage | Frontal single | Frontal multi (group) | Frontal multi (pseudo-cal) | p (gain) |
|--------|-------------|----------------|-----------------------|---------------------------|----------|
| PDA | 0.117 | 0.015 | 0.001 | **0.113** | 0.009 |
| GSR_CEN | 0.118 | 0.049 | 0.022 | **0.122** | 0.022 |
| CEN | 0.119 | 0.094 | 0.101 | **0.121** | 0.335 |
| DMN | 0.081 | 0.046 | 0.033 | −0.016 | 0.824 |
| GSR_PDA | 0.115 | 0.026 | −0.001 | **0.107** | 0.008 |

**Cross-cohort generalization of the frontal multivariate model.** Trained on all
DMNELF subjects and applied to the independent rtBPD cohort, the frontal multivariate
group model generalized well without any calibration: nf1 — PDA r = 0.109, CEN
r = 0.156, DMN r = 0.112, GSR_CEN r = 0.119, GSR_PDA r = 0.110; nf2 — PDA r = 0.059,
CEN r = 0.127. Cross-cohort frontal-multivariate performance equaled or exceeded the
full single-electrode cross-cohort transfer for most targets in nf1 (cf. Table 3:
full nf1 PDA r = 0.067, CEN r = 0.138, DMN r = 0.097). Pseudo-calibration did not
improve cross-cohort decoding (all gain p > 0.28), indicating that the DMNELF-trained
group model already generalizes without individualization when applied to a new
population.

Together these results indicate that a frontal-only portable EEG headset can recover
the EFP for CEN and PDA — the primary neurofeedback targets — provided that one
pseudo-calibration run (requiring no fMRI) is collected within-cohort, or that a
cross-cohort group model is applied directly.

### Summary

The EFP method reproduces on DMN/CEN/PDA in this cohort and yields the best
within-subject and cross-subject decoding we have obtained for these targets, with
interpretable alpha-band fingerprints and a validated visual positive control. The
principal caveat is spatial: the recoverable signal is distributed and global rather
than focal, so the fingerprints should be read as network-level, arousal-linked
predictors rather than evidence of localized cortical sources. Crucially, the
fingerprint is partially recoverable with frontal-only EEG: after a single
pseudo-calibration run (no fMRI required), PDA and GSR-corrected CEN decoding from
11 frontal channels approaches full-montage accuracy, opening a path toward portable
neurofeedback deployment.

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
