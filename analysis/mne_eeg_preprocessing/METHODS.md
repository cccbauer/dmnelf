DMNELF Microstate PDA Pipeline — Methods and Usage Guide

Study: DMNELF (simultaneous EEG-fMRI neurofeedback)
Author: Clemens C.C. Bauer, MD PhD
Lab: EPIC Brain Lab, Northeastern University / Gabrieli Lab, MIT
Grant: R21MH130915
Cluster: explorer.northeastern.edu / account: suewhit


PART 2 — METHODS
================

Overview

We develop an offline EEG-to-fMRI decoder that predicts the real-time
neurofeedback signal (Positive Diametric Activity, PDA) from EEG
microstates. The pipeline replicates and extends the TESS method (Custo
et al. 2014, 2017) on simultaneous EEG-fMRI data collected during
mindfulness-based neurofeedback training (Bloom et al. 2023). The
ultimate goal is to replace the real-time fMRI signal with an EEG-derived
proxy, enabling neurofeedback outside the MRI scanner.


Dataset

Study: DMNELF. Design: simultaneous EEG-fMRI, Siemens Prisma 3T.
Fifteen subjects total. Ten have complete simultaneous EEG and fMRI
recordings (sub-dmnelf001, 004, 005, 006, 007, 008, 009, 010, 011, 1001).
Two have matched but fewer runs due to incomplete acquisition
(sub-dmnelf1002, 1003). Three have fMRI only because no R128 scanner
trigger marker was recorded during EEG acquisition (sub-dmnelf002, 003, 999).

Tasks: rest (2 runs, ~350 volumes each, TR=1.2s) used for microstate
template fitting; shortrest (1 run, ~100 volumes) used as near-transfer
test; feedback (4 runs, ~125 volumes each) used for decoder training and
evaluation; experiencesampling (4 runs, EEG only, not used in decoder).

EEG: 32-channel BrainProducts MRI-compatible amplifier (BrainCap MR),
simultaneously recorded at 5000 Hz. fMRI: TR=1.2s, 2mm isotropic,
preprocessed with fMRIPrep 24.1.1, MNI152NLin6Asym space, res-2.


Step 0a — EEG Minimal Preprocessing (BrainVision Analyzer)

Raw EEG was recorded simultaneously with fMRI at 5000 Hz. Minimal
preprocessing was performed in BrainVision Analyzer 2. First, MR gradient
artifacts were corrected using the Average Artifact Subtraction method
(AAS; Allen et al. 2000) with a sliding average window of 21 artifacts,
time-locked to the MR slice acquisition trigger (R128 marker). Second, data
were downsampled from 5000 Hz to 1000 Hz with anti-aliasing filter applied.
Third, each run was segmented from onset to offset trigger markers and
exported as individual EDF files. Output files are stored in BIDS-compatible
format with the description label bvaAC1kHz.

Reference: Allen PJ, Josephs O, Turner R (2000). A method for removing
imaging artifact from continuous EEG recorded during functional MRI.
NeuroImage, 12(2), 230-239. https://doi.org/10.1006/nimg.2000.0599


Step 0b — EEG Full Preprocessing

Automated preprocessing was implemented in eeg_preproc.py and deployed
to the Explorer HPC cluster via eeg_preproc_deploy.py. Two versions are
produced in parallel: one downsampled to 250 Hz and one to 500 Hz, to
allow comparison of microstate decoder performance at different temporal
resolutions.

The pipeline proceeds as follows.

Loading and ECG identification: the BVA-preprocessed EDF is loaded and
the ECG channel identified by name matching.

R-peak detection: NeuroKit2 ecg_clean and ecg_peaks functions detect
R-peak timing from the ECG channel. These R-peaks drive the BCG correction
step and inform cardiac ICA detection.

Bad channel detection: channels are flagged as bad using two criteria.
First, variance z-score: channels with |z| > 3.0 relative to the
distribution of all channel standard deviations are flagged. Second,
high-frequency noise: channels with HF band power (40-50 Hz) z-score > 2.5
are flagged. Both lists are combined.

Edge annotation: scanner gradient ramp-up and ramp-down artifact at run
onset and offset are detected by computing the per-sample RMS across EEG
channels and comparing to 3 times the stable median RMS (computed from
the middle 80% of the recording). Noisy edge periods are annotated as
BAD_edge_start and BAD_edge_end and excluded from all subsequent processing.

Bandpass filtering: a 1-40 Hz zero-phase FIR filter is applied to EEG
channels. The upper cutoff at 40 Hz is chosen to avoid residual gradient
artifact harmonics present in simultaneous EEG-fMRI recordings.

BCG correction: ballistocardiogram (BCG) artifact is corrected using
Average Artifact Subtraction. Raw filtered EEG is epoched around each
NeuroKit2-detected R-peak (-200 to +600 ms window covering the full cardiac
cycle). An average BCG template is computed across all epochs and subtracted
from the continuous data at each heartbeat onset. This approach directly uses
the simultaneously recorded ECG channel for precise artifact timing.

Downsampling: data are resampled to the target frequency (250 or 500 Hz).

Montage: the standard 10-20 montage is applied to enable ICLabel
classification.

ICA decomposition: Picard ICA (ortho=False, extended=True) is fitted on
EEG channels excluding BAD-annotated segments. The number of components is
set to min(29, n_eeg-1). Picard is used instead of FastICA because it
provides faster convergence and more stable decompositions for EEG data
(Ablin et al. 2018).

Artifact component identification proceeds through three parallel methods.
ICLabel (Pion-Tonachini et al. 2019) classifies all components; those
labeled non-brain are candidates for removal, capped at 20% of total
components to prevent over-rejection. Cardiac components are identified
using the CTPS method (ica.find_bads_ecg, threshold=0.9) applied to ECG
epochs created from the ECG channel; if CTPS finds no components, the
highest-correlation component is selected as fallback (threshold r>0.2).
EOG components are identified by Pearson correlation between ICA sources
and the mean of Fp1/Fp2 used as ocular proxy channels (threshold r>0.5).
The final exclusion set is the union of cardiac, EOG, and capped ICLabel
components, with cardiac and EOG components always preserved.

Bad channel interpolation: flagged channels are interpolated using spherical
spline interpolation.

Average reference: data are re-referenced to the average of all EEG channels.

Output files are saved as FIF with the description label preproc250Hz or
preproc500Hz. QC images are saved per run showing channel standard
deviations, whole-run RMS traces, PSD comparison before and after
preprocessing, and per-channel traces before and after.

References:
Ablin P, Cardoso JF, Gramfort A (2018). Faster independent component
analysis by preconditioning with Hessian approximations. IEEE Transactions
on Signal Processing, 66(15), 4040-4049.
https://doi.org/10.1109/TSP.2018.2844203

Gramfort A et al. (2013). MEG and EEG data analysis with MNE-Python.
Frontiers in Neuroscience, 7, 267.
https://doi.org/10.3389/fnins.2013.00267

Pion-Tonachini L, Kreutz-Delgado K, Makeig S (2019). ICLabel: An automated
electroencephalographic independent component classifier, dataset, and
website. NeuroImage, 198, 181-197.
https://doi.org/10.1016/j.neuroimage.2019.05.026


Step 0c — DiFuMo-64 Timeseries Extraction

BOLD signal timeseries were extracted from 64 predefined brain parcels
using the DiFuMo-64 probabilistic atlas (Dadi et al. 2020) and
NiftiMapsMasker from nilearn with standardize=False. Each parcel timeseries
is the weighted average of all voxels within that parcel's probability map,
yielding an (n_volumes, 64) matrix per run. Output TSV files contain
columns: volume, time, ROI_01 through ROI_64. DiFuMo parcels are
data-driven, overlapping, and anatomically meaningful, providing good
coverage of DMN and CEN with minimal spatial leakage between networks.

Reference: Dadi K et al. (2020). Fine-grain atlases of functional modes for
fMRI analysis. NeuroImage, 221, 117126.
https://doi.org/10.1016/j.neuroimage.2020.117126


Step 0d — Personalized DMN/CEN Masks

Subject-specific DMN and CEN binary masks were extracted using CanICA
decomposition of each subject's resting-state fMRI data, and used to
compute subject-specific DiFuMo-64 parcel overlap weights.

The procedure is as follows. The fMRIPrep rest run-01 BOLD
(MNI152NLin6Asym res-2, approximately 350 volumes) is loaded. CanICA
(nilearn, 35 components matching the real-time neurofeedback pipeline,
random_state=42) is run with the fMRIPrep brain mask. The Yeo-7 atlas is
resampled to BOLD space: the Default network (label 7, covering PCC, mPFC,
and angular gyrus) serves as the DMN reference, and the Frontoparietal
network (label 6, covering bilateral IPS and dlPFC) serves as the CEN
reference. Yeo-7 is preferred over Yeo-17 because it provides unambiguous
single labels for each network covering the full bilateral extent, whereas
Yeo-17 splits DMN and CEN into sub-networks that cause incomplete coverage
and PCC/IPS confusion at network boundaries.

Spatial correlations between all ICA components and the Yeo reference masks
are computed in a vectorized fashion inside the brain mask only, adapted
from the pineuro pipeline (Hacker et al. 2013). The top 2 components per
network are selected by absolute spatial correlation. CEN selection excludes
midline posterior components (centroid |x| < 15mm AND y < -45mm) to prevent
PCC/precuneus contamination — a DMN hub that consistently leaks into the
ContA Yeo label. Components are sign-corrected by flipping those
anti-correlated with the Yeo reference, then z-scored before combining to
ensure equal contribution from both components regardless of value range
(adapted from pineuro combine_ica_components). The combined map is
thresholded to the top 2000 voxels to produce a binary mask.

For each DiFuMo-64 parcel p, the overlap weight is computed as
w(p) = sum(parcel_p times mask) / sum(all parcel-mask products),
normalized to sum=1 across all 64 parcels.

References:
Yeo BT et al. (2011). The organization of the human cerebral cortex
estimated by intrinsic functional connectivity. Journal of
Neurophysiology, 106(3), 1125-1165.
https://doi.org/10.1152/jn.00338.2011

Hacker CD et al. (2013). Resting state network estimation in individual
subjects. NeuroImage, 82, 616-633.
https://doi.org/10.1016/j.neuroimage.2013.05.108

Varoquaux G et al. (2010). A group model for stable multi-subject ICA on
fMRI datasets. NeuroImage, 51(1), 288-299.
https://doi.org/10.1016/j.neuroimage.2009.12.091


Step 0e — Add Personalized Parcel Composites

Each DiFuMo-64 timeseries TSV is augmented with two subject-specific
weighted composite columns. DMN_personal(t) is computed as the weighted
sum of all 64 parcel timeseries using the subject's dmn_weights vector.
CEN_personal(t) is computed analogously using cen_weights. The final TSV
contains 68 columns: volume, time, ROI_01 through ROI_64, DMN_personal,
and CEN_personal.

Three design options were considered: (A) two composite columns only,
losing individual parcel information; (B) 128 reweighted parcel columns,
redundant and very wide; (C) 64 raw parcels plus 2 composites. Option C
was chosen because it preserves the full 64-parcel signal for group-level
and exploratory analysis while adding personalized network summaries that
directly enable subject-specific PDA computation.


Step 1 — Microstate Map Fitting

Seven prototype EEG scalp topographies (microstate maps) are learned from
resting-state data pooled across all EEG+fMRI subjects. Preprocessed FIF
files from rest run-01 and run-02 are loaded. GFP peak samples are
extracted — these are moments of maximum spatial coherence where the
topography is most clearly defined (Lehmann et al. 1987). Outlier peaks
with GFP > 3 SD are rejected. Polarity-invariant k-means clustering with
k=7 is run with 20 random restarts and the solution with highest Global
Explained Variance (GEV) is retained. Polarity invariance is applied
because EEG oscillates and the same neural generator produces both
topographic polarities within ~50ms.

The choice of k=7 is motivated by Custo et al. (2017), who showed that
k=7 separates the DMN microstate (C, sources in PCC and precuneus) from
the salience network microstate (E, sources in dACC and insula), which
k=4 conflates due to spatial similarity (r > 0.7). Tarailis et al. (2023)
systematically reviewed 50 microstate studies and confirmed that k=4
causes systematic misattribution of DMN versus salience network dynamics.
For our purpose of predicting PDA = CEN minus DMN, clean separation of
these networks is essential.

Expected GEV is approximately 57%, lower than the typical literature range
of 68-88% for high-density EEG, due to the lower channel count (31 channels
versus 64-256 in most studies) and residual BCG artifact from the
simultaneous fMRI acquisition.

The functional assignments of the 7 microstate maps are as follows.
Microstate A corresponds to auditory and arousal networks with sources
in temporal cortex. Microstate B corresponds to visual and self-referential
processing with sources in occipital cortex and precuneus. Microstate C
corresponds to the DMN with sources in PCC, precuneus, and angular gyrus —
predicted to carry negative weight in the PDA decoder. Microstate D
corresponds to the frontoparietal/CEN network with sources in right IPS
and dlPFC — predicted to carry positive weight. Microstate E corresponds
to the salience network with sources in dACC and insula. Microstate F
corresponds to the anterior DMN with sources in mPFC. Microstate G
corresponds to somatosensory network with sources in right parietal cortex.

References:
Custo A et al. (2017). Electroencephalographic resting-state networks:
source localization of microstates. Brain Connectivity, 7(9), 671-682.
https://doi.org/10.1089/brain.2016.0476

Tarailis P et al. (2023). The functional aspects of resting EEG
microstates: a systematic review. Brain Topography, 37, 181-217.
https://doi.org/10.1007/s10548-023-01006-2

Michel CM, Koenig T (2018). EEG microstates as a tool for studying the
temporal dynamics of whole-brain neuronal networks: a review.
NeuroImage, 180, 577-593.
https://doi.org/10.1016/j.neuroimage.2017.11.062


Step 2 — TESS Features

For every EEG sample in every run, the instantaneous scalp topography is
projected onto each of the 7 microstate templates using matrix
multiplication: T-hat(t) = Templates times v(t), yielding 7 continuous
T-hat coefficients per sample. Unlike binary winner-takes-all microstate
labeling, these preserve signal amplitude and within-TR temporal dynamics.
At 250 Hz with TR=1.2s each volume contains 300 EEG samples; at 500 Hz,
600 samples per volume — information entirely discarded by binary labeling.
Our previous pipeline with binary features achieved r~0.025 with PDA,
insufficient for a decoder.

Each T-hat timecourse is convolved with a canonical double-gamma
hemodynamic response function (Glover 1999) and downsampled to the fMRI TR
by averaging within each TR window, accounting for the approximately 6s
neurovascular delay. Two additional features are computed per TR: Global
Field Power (GFP, spatial standard deviation across channels, reflecting
overall signal strength) and Global Map Dissimilarity (GMD, rate of
topographic change, reflecting microstate transitions). The final feature
matrix is (n_volumes, 9): T-hat_A through T-hat_G, GFP, and GMD.

References:
Custo A et al. (2014). Generalized TESS for EEG source localization.
Brain Topography, 27(1), 95-105.
https://doi.org/10.1007/s10548-013-0319-5

Britz J et al. (2010). BOLD correlates of EEG topography reveal rapid
resting-state network dynamics. NeuroImage, 52(4), 1162-1170.
https://doi.org/10.1016/j.neuroimage.2010.05.052

Glover GH (1999). Deconvolution of impulse response in event-related
BOLD fMRI. NeuroImage, 9(4), 416-429.
https://doi.org/10.1006/nimg.1998.0419


Step 3 — PDA Target Computation

The neurofeedback target signal PDA(t) = mean(CEN_z(t)) minus
mean(DMN_z(t)) is computed for each run in two variants. Positive PDA
indicates CEN dominance (ball moves up on screen during feedback);
negative PDA indicates DMN dominance (ball moves down). To align with
what subjects actually received during neurofeedback, each parcel
timeseries is baseline z-scored relative to the first 25 volumes (30s)
of each run, matching the MURFI real-time computation (Hinds et al. 2011;
Bloom et al. 2023).

PDA_group uses fixed group-level DiFuMo parcel indices: DMN parcels
[3, 6, 22, 29, 35, 38, 58, 60] and CEN parcels [4, 31, 47, 48, 50, 51].
PDA_personal uses the subject-specific weighted composites from step 0e:
PDA_personal(t) = CEN_personal_z(t) minus DMN_personal_z(t).

The group mean DMN-CEN correlation is +0.636 in this dataset without
global signal regression (GSR). This positive correlation is expected:
Murphy et al. (2009) demonstrated that GSR mathematically forces
anticorrelation and should not be applied when the goal is to preserve
naturalistic network dynamics.

References:
Hinds O et al. (2011). Computing moment-to-moment BOLD activation for
real-time neurofeedback. NeuroImage, 54(1), 361-368.
https://doi.org/10.1016/j.neuroimage.2010.07.060

Murphy K et al. (2009). The impact of global signal regression on
resting state correlations. Journal of Neuroscience, 29(38), 13513-13531.
https://doi.org/10.1523/JNEUROSCI.3090-09.2009

Bloom PA et al. (2023). Mindfulness-based real-time fMRI neurofeedback.
BMC Psychiatry, 23, 757.
https://doi.org/10.1186/s12888-023-05241-2


Step 4 — Decoder Training

A regularized linear decoder is trained to predict single-TR PDA from
the 9 EEG microstate features. ElasticNet regression (Zou & Hastie 2005)
combines L1 sparsity and L2 stability, appropriate for correlated features.
Leave-one-run-out cross-validation is applied across the 4 feedback runs
per subject. Features and target are z-scored before fitting. One decoder
is trained per subject per PDA variant. Hyperparameters (alpha and L1 ratio)
are selected by inner CV.

Based on the microstate functional assignments from Custo et al. (2017) and
Tarailis et al. (2023), the predicted weight pattern is: T-hat_C negative
(DMN sources drive ball down), T-hat_D positive (CEN/FPN sources drive ball
up), T-hat_E and T-hat_F secondary negative (salience and anterior DMN),
and sensory microstates (A, B, G) near zero.

References:
Zou H, Hastie T (2005). Regularization and variable selection via the
elastic net. Journal of the Royal Statistical Society Series B, 67(2),
301-320. https://doi.org/10.1111/j.1467-9868.2005.00503.x

Meir-Hasson Y et al. (2016). An EEG finger-print of fMRI deep regional
activation. NeuroImage, 131, 120-128.
https://doi.org/10.1016/j.neuroimage.2015.11.053


Step 5 — Evaluation

Decoder performance is evaluated on two held-out test sets. Near transfer
uses the shortrest run from the same session — same subject, same day, no
active neurofeedback, tests generalization from task to passive rest.
Far transfer uses the left-out feedback runs from LORO cross-validation,
testing within-task generalization. The primary metric is Pearson r between
predicted and actual PDA. Secondary metrics include R-squared, per-run
breakdown, and group mean plus standard deviation across subjects. The
group decoder and personal decoder are compared directly.

Target performance based on Meir-Hasson et al. (2016) with the EFP
framework is r approximately 0.3. With TESS features on 12 subjects we
target r > 0.25 as a minimum threshold for the decoder to be clinically
meaningful.


Step 6 — Microstate Epoch Statistics

To identify which microstates best differentiate high-PDA from low-PDA brain
states, and which combination of microstates carries the most discriminative
information, we apply a three-tier statistical analysis implemented in
06_stats_microstate_pda.py.

Primary analysis — Two-stage t-test. For each subject, all TRs are split
into PDA-positive epochs (pda_direct >= 0, CEN dominant) and PDA-negative
epochs (pda_direct < 0, DMN dominant) using the personalized-mask PDA signal
(pda_direct) as the ground truth, matching the real-time MURFI computation.
For each of the 7 microstate T-hat coefficients, a Welch two-sample t-test
(unequal variance) is applied between PDA-positive and PDA-negative T-hat
distributions. Effect size is quantified as pooled-variance Cohen's d.
Within each subject, p-values are corrected across the 7 microstates using
the Benjamini-Hochberg false discovery rate (FDR, alpha = 0.05). At the
group level, the per-subject mean differences (PDA+ mean minus PDA- mean)
for each microstate are submitted to a one-sample t-test against zero,
implementing the standard second-level random-effects summary-statistics
approach used in neuroimaging group analyses (equivalent to FSL/SPM
random-effects inference). Group-level p-values are again FDR-corrected
across the 7 microstates. The analysis is run separately for all tasks pooled
(maximizing statistical power) and for the feedback task only (most relevant
to the neurofeedback application).

Confirmatory analysis — Linear Mixed Model (LMM). For each microstate,
a LMM is fitted using the statsmodels MixedLM implementation. The fixed-effects
structure is T-hat ~ pda_sign + C(task) (all-tasks model) or T-hat ~ pda_sign
(feedback-only model), where pda_sign is binary (1 = PDA-positive, 0 =
PDA-negative). The random-effects structure includes a per-subject random
intercept and random slope for pda_sign, estimated by restricted maximum
likelihood (REML). The random slope quantifies between-subject heterogeneity
in the microstate-PDA relationship. If the random-slope model fails to
converge, a random-intercept-only fallback is used. An additional interaction
model (T-hat ~ pda_sign * C(task)) tests whether the microstate-PDA
relationship differs across tasks (rest, shortrest, feedback), probing whether
neurofeedback training modulates microstate selectivity. The key output is the
fixed-effect coefficient for pda_sign (beta, in T-hat units) with 95%
confidence intervals, reported alongside FDR-corrected p-values across the 7
microstates.

The LMM provides advantages over the two-stage t-test by: (1) modelling
within-subject random slopes to capture individual differences in the
microstate-PDA relationship; (2) including task as a fixed covariate to
control for task-dependent mean shifts; and (3) enabling a formal interaction
test for task-modulated microstate selectivity. The two-stage approach is
retained as the primary analysis because it is transparent, computationally
robust, and directly comparable to standard neuroimaging random-effects models.

Combination analysis — Logistic regression. To assess whether a linear
combination of all 7 T-hat features predicts PDA sign more accurately than
any individual microstate, a logistic regression classifier (L2 penalty,
C = 1, class-weight balanced to handle PDA+/PDA- imbalance) is trained using
leave-one-run-out cross-validation within the feedback task runs. Features are
z-scored within each training fold. Classification performance is quantified
by the area under the receiver operating characteristic curve (AUC), and
tested against chance (AUC = 0.5) at the group level using a one-sample
t-test. The mean classifier weights across LORO folds indicate which
microstates contribute most to the discriminant boundary, complementing the
univariate t-test and LMM results.

Outputs: individual_stats_250Hz.csv (per-subject t-statistics, p-values,
Cohen's d, FDR flags), group_stats_250Hz.csv (group t-test and LMM fixed
effects with 95% CI, interaction p-values), combination_logistic_250Hz.csv
(per-subject AUC, accuracy, discriminant weights), and group_stats_250Hz.png
(Panel A: two-stage t-test bar chart; Panel B: LMM forest plot).

References:
Benjamini Y, Hochberg Y (1995). Controlling the false discovery rate: a
practical and powerful approach to multiple testing. Journal of the Royal
Statistical Society Series B, 57(1), 289-300.
https://doi.org/10.1111/j.2517-6161.1995.tb02031.x

Bates D et al. (2015). Fitting linear mixed-effects models using lme4.
Journal of Statistical Software, 67(1), 1-48.
https://doi.org/10.18637/jss.v067.i01

Friston KJ et al. (1999). How many subjects constitute a study?
NeuroImage, 10(1), 1-5. https://doi.org/10.1006/nimg.1999.0439


Full Reference List

Allen PJ, Josephs O, Turner R (2000). A method for removing imaging
artifact from continuous EEG recorded during functional MRI. NeuroImage,
12(2), 230-239. https://doi.org/10.1006/nimg.2000.0599

Ablin P, Cardoso JF, Gramfort A (2018). Faster independent component
analysis by preconditioning with Hessian approximations. IEEE Transactions
on Signal Processing, 66(15), 4040-4049.
https://doi.org/10.1109/TSP.2018.2844203

Bloom PA et al. (2023). Mindfulness-based real-time fMRI neurofeedback:
a randomized controlled trial to optimize dosing for depressed adolescents.
BMC Psychiatry, 23, 757. https://doi.org/10.1186/s12888-023-05241-2

Britz J et al. (2010). BOLD correlates of EEG topography reveal rapid
resting-state network dynamics. NeuroImage, 52(4), 1162-1170.
https://doi.org/10.1016/j.neuroimage.2010.05.052

Custo A et al. (2014). Generalized TESS for EEG source localization.
Brain Topography, 27(1), 95-105.
https://doi.org/10.1007/s10548-013-0319-5

Custo A et al. (2017). Electroencephalographic resting-state networks:
source localization of microstates. Brain Connectivity, 7(9), 671-682.
https://doi.org/10.1089/brain.2016.0476

Dadi K et al. (2020). Fine-grain atlases of functional modes for fMRI
analysis. NeuroImage, 221, 117126.
https://doi.org/10.1016/j.neuroimage.2020.117126

Glover GH (1999). Deconvolution of impulse response in event-related
BOLD fMRI. NeuroImage, 9(4), 416-429.
https://doi.org/10.1006/nimg.1998.0419

Gramfort A et al. (2013). MEG and EEG data analysis with MNE-Python.
Frontiers in Neuroscience, 7, 267.
https://doi.org/10.3389/fnins.2013.00267

Hacker CD et al. (2013). Resting state network estimation in individual
subjects. NeuroImage, 82, 616-633.
https://doi.org/10.1016/j.neuroimage.2013.05.108

Hinds O et al. (2011). Computing moment-to-moment BOLD activation for
real-time neurofeedback. NeuroImage, 54(1), 361-368.
https://doi.org/10.1016/j.neuroimage.2010.07.060

Meir-Hasson Y et al. (2016). An EEG finger-print of fMRI deep regional
activation. NeuroImage, 131, 120-128.
https://doi.org/10.1016/j.neuroimage.2015.11.053

Michel CM, Koenig T (2018). EEG microstates as a tool for studying the
temporal dynamics of whole-brain neuronal networks: a review.
NeuroImage, 180, 577-593.
https://doi.org/10.1016/j.neuroimage.2017.11.062

Murphy K et al. (2009). The impact of global signal regression on resting
state correlations. Journal of Neuroscience, 29(38), 13513-13531.
https://doi.org/10.1523/JNEUROSCI.3090-09.2009

Pion-Tonachini L, Kreutz-Delgado K, Makeig S (2019). ICLabel: An automated
electroencephalographic independent component classifier, dataset, and
website. NeuroImage, 198, 181-197.
https://doi.org/10.1016/j.neuroimage.2019.05.026

Tarailis P et al. (2023). The functional aspects of resting EEG microstates:
a systematic review. Brain Topography, 37, 181-217.
https://doi.org/10.1007/s10548-023-01006-2

Varoquaux G et al. (2010). A group model for stable multi-subject ICA on
fMRI datasets. NeuroImage, 51(1), 288-299.
https://doi.org/10.1016/j.neuroimage.2009.12.091

Yeo BT et al. (2011). The organization of the human cerebral cortex
estimated by intrinsic functional connectivity. Journal of Neurophysiology,
106(3), 1125-1165. https://doi.org/10.1152/jn.00338.2011

Zhang Y et al. (2023). Reducing default mode network connectivity with
mindfulness-based fMRI neurofeedback: a pilot study. Molecular Psychiatry,
28, 3184-3195. https://doi.org/10.1038/s41380-023-02193-9

Zou H, Hastie T (2005). Regularization and variable selection via the
elastic net. Journal of the Royal Statistical Society Series B, 67(2),
301-320. https://doi.org/10.1111/j.1467-9868.2005.00503.x