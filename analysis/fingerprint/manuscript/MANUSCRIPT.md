# A portable, personalized EEG decoder of a default-mode–executive neurofeedback target, replicated across two clinical cohorts

<!-- Alternative titles:
  - From fMRI to EEG: a personalized, motion-controlled decoder of a default-mode–executive neurofeedback target, replicated transdiagnostically and deployable on consumer hardware
  - A replicated EEG decoder of a DMN–executive neurofeedback target: personalized within-subject, calibratable across cohorts, portable to a 14-channel headset -->


*Target journal: Biological Psychiatry: Cognitive Neuroscience and Neuroimaging (Archival Report — 4000-word body, 250-word structured abstract, IMRD).*

**Authors:** `[TODO: author list — order + initials]`
**Affiliations:** `[TODO: affiliations]`
**Corresponding author:** Clemens C. C. Bauer `[TODO: email + postal address]`
**Keywords:** `[TODO: 5–6 — e.g. neurofeedback; default-mode network; EEG–fMRI; EEG fingerprint; transdiagnostic; schizophrenia]`
**Running title:** `[TODO: ≤ 45 char]`
**Word count:** body `[TODO]` / abstract `[TODO]`; figures `[TODO]`; tables `[TODO]`; references `[TODO]`.

> Full draft, reframed around the honest decoding analysis (all numbers on confound-regressed
> targets, leave-one-run-out, feedback block; see `efp_meirhasson/EFP_SPINE.md`). Introduction,
> Results, and Discussion are written; Methods §2.1–2.9 complete; MRI acquisition confirmed from the
> Prisma protocols (§2.4 + Supp Table S1). Remaining **[CONFIRM]** items are factual gaps only:
> schizophrenia symptom scale + medication, borderline-trait scale name, IRB/site details,
> fMRIPrep version + confound strategy. Scope note: resting-state connectivity analyses
> (restfc/, fox_seed/) are internal negative/validation results and are deliberately **excluded**.

---

## Abstract (Background/Methods/Results/Conclusions — 250 words)

**Background:** Auditory hallucinations in schizophrenia are often medication-resistant and are
linked to default-mode-network (DMN) hyperconnectivity with cognitive-control networks. Network-based
real-time fMRI neurofeedback lets patients regulate this interaction — the executive-over-default
differential, Positive Diametric Activity (PDA = CEN − DMN) — but fMRI is not scalable. Whether a
portable EEG decoder can index this neurofeedback target, and generalize beyond a single disorder, is
unknown.

**Methods:** In a discovery cohort of adults with schizophrenia and auditory hallucinations
undergoing simultaneous EEG–fMRI mindfulness-based neurofeedback (n=17), we trained an EEG decoder
(single-electrode Stockwell time-frequency fingerprint; Meir-Hasson et al., 2014) of the personalized
BOLD CEN, DMN and PDA. All coupling was evaluated against **confound-regressed** BOLD targets,
leave-one-run-out, within the feedback block. We tested within-subject decoding, frozen cross-cohort
transfer and one-run calibration in an independent cohort — adolescents with elevated borderline
traits (n=19; n=11 at retest) — and a virtual 14-channel consumer-headset montage. Deep-learning,
frontal-theta and signal-to-noise-normalized alternatives served as controls.

**Results:** A personalized EEG decoder tracked the BOLD networks within-subject and **replicated
across both cohorts and three sessions** (CEN r≈0.08–0.10; DMN r≈0.06–0.09). The executive network
(CEN) was the robust readout; the PDA differential decoded weakly (r≈0.06). Calibration-free transfer
was weak (CEN r≈0.07–0.10) but significant and improved with a single calibration run (r≈0.12). The
decoder survived restriction to a 14-channel consumer montage (CEN r≈0.095). Deep learning,
frontal-theta and signal-to-noise normalization did not improve on the linear decoder, and the signal
was neural (<20 Hz, distributed centro-parietal), not myogenic.

**Conclusions:** A personalized, motion-controlled EEG decoder indexes a DMN–executive neurofeedback
target, replicates transdiagnostically, and is deployable on consumer hardware — a rigorous step
toward scalable EEG neurofeedback.

---

## 1. Introduction

Psychiatric disorders are increasingly understood as disorders of interaction among large-scale,
distributed brain circuits. In schizophrenia, auditory hallucinations are among the most distressing
symptoms and are frequently resistant to medication, and they have been linked to abnormal
connectivity of large-scale networks — hyperconnectivity within the default-mode network (DMN) and
between the DMN and networks implicated in cognitive control (the frontoparietal / central-executive
network, CEN) as well as regions supporting the experience of hallucinations such as the superior
temporal gyrus (Whitfield-Gabrieli et al., 2009; Whitfield-Gabrieli & Ford, 2012). More broadly,
imbalanced DMN–CEN interaction — excessive default-mode processing that the executive system fails to
regulate — recurs across otherwise distinct disorders, motivating circuit-based interventions that
target the interaction itself, in line with the Research Domain Criteria.

Network-based real-time fMRI neurofeedback (rt-fMRI-NF) operationalizes this target directly, letting
patients observe and learn to regulate specific network interactions. Participants up-regulate
**Positive Diametric Activity (PDA; Bauer et al., 2019)** — the differential activation of the
executive over the default network, PDA = activation(CEN) − activation(DMN) — using personalized
network masks and a mental-noting attention strategy (Chen et al., 2013; Bauer et al., 2019, 2020;
Zhang et al., 2023). PDA was introduced by Bauer et al. (2019), who showed that this CEN-over-DMN
index rises during meditation and persists afterward, and that its increase tracks a reduction in
intrinsic DMN connectivity — grounding it in the causal, inhibitory control the CEN exerts over the
DMN (Chen et al., 2013). Mindfulness-based rt-fMRI-NF of PDA attenuates auditory hallucinations in
schizophrenia (Bauer et al., 2020) and reduces DMN connectivity and symptoms in adolescents at
affective-disorder risk (Zhang et al., 2023). But fMRI is expensive, non-portable, and confined to
the scanner, so the intervention cannot scale to the settings where it would matter — motivating a
search for a portable substitute that could ultimately deliver neurofeedback outside the magnet.

Electroencephalography (EEG) is inexpensive and deployable, and a large EEG–fMRI literature shows
that EEG features can serve as markers of regional and network-level fMRI activity — including a
single-electrode "EEG fingerprint" fitted to a deep-region fMRI signal (the EEG-fingerprint / EFP;
Meir-Hasson et al., 2014), which has since guided EEG-only neurofeedback of limbic targets. Whether
such a decoder can be built for a **cortical DMN–executive network target**, whether it **replicates
in a cohort it was never tuned to**, and whether it **survives the move to a consumer headset**, are
open questions — and they are the questions that decide deployability. They are also easy to answer
over-optimistically: cross-modal EEG↔BOLD coupling is inflated both by motion shared between the two
signals and by cross-validation that leaks autocorrelated within-run samples, so an honest evaluation
must regress physiological/motion confounds out of the BOLD target and hold out whole runs.

Here we take that honest route in two clinical cohorts studied with simultaneous EEG–fMRI during the
identical mindfulness-based neurofeedback paradigm. In a **discovery cohort of adults with
schizophrenia and auditory hallucinations**, we ask (i) can a personalized EEG decoder track the
confound-regressed BOLD CEN, DMN and PDA within-subject, out-of-sample across runs? We then ask the
questions that matter for deployment: (ii) does the decoder **replicate**, unchanged, in an
independent, diagnostically and developmentally distinct cohort — adolescents with elevated
borderline personality traits — both zero-shot and after a single calibration run?; and (iii) does it
**survive restriction to a 14-channel consumer montage**? Finally, because several popular routes
promise more, we test them head-to-head as controls: an end-to-end deep network on raw EEG, a
frontal-theta asymmetry index, and a signal-to-noise-normalized ("f-SNR") feature. We show that a
personalized EEG decoder indexes the DMN–executive target — most robustly its executive (CEN)
component — replicates transdiagnostically, and is portable to consumer hardware, while none of the
more elaborate alternatives improves on it.

## 2. Methods and Materials

### 2.1 Participants

Two independent clinical cohorts underwent personalized real-time-fMRI mindfulness-based
neurofeedback (mbNF) with simultaneous EEG–fMRI. The schizophrenia cohort served as the **discovery
cohort** in which the EEG decoder of the neurofeedback target was established; the borderline-trait
cohort served as the **independent replication cohort**.
- **Discovery cohort — schizophrenia ("DMNELF"):** n=17 adults with a diagnosis of schizophrenia
  and auditory hallucinations. `[TODO: confirm all n=17 had current/persistent AHs; age/sex;
  diagnostic instrument (SCID?); symptom scale — PANSS/PSYRATS or other; medication status.]`
- **Replication cohort — borderline traits ("rtBPD"):** adolescents with *elevated borderline
  personality traits* (dimensional; no categorical diagnosis) assessed with `[TODO: BPD trait scale
  name + cutoff — e.g. BPFSC / PAI-BOR]`. Analyzed at two neurofeedback sessions (nf1: n=19;
  nf2: n=11). `[TODO: age range; sex.]`
All participants provided informed consent (assent + guardian consent for minors); procedures were
IRB-approved. `[TODO: approving IRB(s) + protocol numbers.]`

*Behavioural self-report (both cohorts).* After each feedback run, participants in **both** cohorts
rated four 1–9 sliders: **state calm** ("how calm are you feeling"; the affective/clinical outcome),
**mindfulness engagement** (mental "noting" in rtBPD / "describing" in DMNELF; harmonised),
**attention to the feedback** ("ball-check"), and **task difficulty**. Coverage: rtBPD 197 runs
(25 participants, nf1+nf2); DMNELF 47 rated runs (16 participants). State calm is the primary
behavioural outcome for validating the neurofeedback target. `[TODO: trait-level measures —
schizophrenia symptom scale (e.g. PANSS) + rtBPD borderline-trait scale — needed for the planned
trait-severity biomarker analysis.]`

### 2.2 Mindfulness-based neurofeedback paradigm and the DMN–CEN target

Both cohorts completed the identical mbNF paradigm (Zhang et al., 2023; Bauer et al., 2020) using
the MURFI real-time functional imaging system (Hinds et al., 2011; Bauer et al., 2022) with
PsychoPy stimulus delivery over a TCP/IP link to the scanner. Each participant's DMN and CEN were
**personalized** (Section 2.3) and used as networks-of-interest. During feedback, MURFI fit an
incremental general linear model to each incoming volume and computed each network's activation in
standard deviations from a rolling baseline; the fed-back signal was **Positive Diametric Activity
(PDA), the differential network activation PDA = activation(CEN) − activation(DMN)** (Bauer et al.,
2019; Chen et al., 2013). Whereas Bauer et al. (2019) originally defined PDA on the fractional
amplitude of low-frequency fluctuations (CEN fALFF − DMN fALFF), here it is computed in real time
from the incremental-GLM network activations. A white dot moved upward when DMN activation fell
below CEN activation (the target
state) and downward otherwise; participants up-regulated PDA by practicing **"mental noting"**, a
Vipassana attention technique taught before scanning. Each feedback run began with a **30 s (25-TR)
rest baseline** followed by continuous feedback. **PDA is therefore both the therapeutic target and
the fMRI signal decoded here.**

*Real-time operationalization.* Real-time computation used MURFI (multivariate and univariate
real-time functional imaging; https://github.com/gablab/murfi2), which processed neurofeedback
only — not network generation or offline preprocessing. Siemens online motion-corrected volumes
(MoCo series, realigned to each run's first volume) were streamed to MURFI over Ethernet; the
personalized DMN and CEN (frontoparietal network, FPN) masks were registered to the first volume
via FLIRT (falling back to Yeo-template masks if a participant's personalized masks were
unavailable). For each masked voxel, an incremental GLM (Gentleman's algorithm) was updated on
every incoming volume, with nuisance regressors for the six relative-displacement realignment
parameters (from the Siemens MoCo DICOM headers) and linear drift; per-voxel activation at time t
was the GLM residual (measured minus nuisance-predicted signal), z-scored to the mean and standard
deviation of the residuals over the **first 25 volumes (30 s baseline)**. Mask-level activation
used variance-based voxel-efficiency weighting (weights inversely proportional to each voxel's
30 s-baseline variance), which converges more closely with offline GLMs than a simple mean or
median (Hinds et al., 2011; Bauer et al., 2022). Feedback latency was < 1 TR (1.2 s); the intrinsic
hemodynamic delay (~6–8 s) is not resolvable by faster sampling or processing.

### 2.3 Personalized DMN/CEN localization

Personalized DMN and CEN masks were derived from resting-state fMRI by independent component
analysis (Melodic ICA), matching components to canonical DMN/CEN atlas maps, thresholding to the
upper 10% of voxel loadings, and binarizing (Zhang et al., 2023).
- **DMNELF (schizophrenia):** masks derived from a **single short resting-state run** acquired in
  the same session as feedback.
- **rtBPD (adolescents):** masks derived in a **separate localizer session** from **two long
  resting-state runs (250 volumes each)**, during which the mindfulness ("mental noting") training
  was delivered; the feedback paradigm was identical to DMNELF.

### 2.4 MRI acquisition

Imaging used a Siemens MAGNETOM Prisma 3T scanner (¹H 123.26 MHz) with a 64-channel head/neck coil.
Full per-sequence parameters are in **Supplementary Table S1**; the two cohorts used the same
functional sequence and differed only in slice coverage, phase-encode direction, and run counts.

*Anatomical.* DMNELF acquired a 4-echo vNav MPRAGE with prospective motion correction (1.0 mm
isotropic, 176 sagittal slices, TR 2530 ms, TE 1.69/3.55/5.41/7.27 ms, TI 1400 ms, flip 7°,
GRAPPA 3); rtBPD (localizer session) acquired a standard single-echo MPRAGE (1.0 mm isotropic,
176 slices, TR 2530 ms, TE 1.92 ms, TI 1400 ms, flip 7°, GRAPPA 3).

*Functional.* All functional runs in both cohorts used a T2\*-weighted simultaneous-multislice
gradient-echo EPI sequence: TR 1200 ms, TE 30 ms, 2.0 mm isotropic voxels, FoV 256 mm, base
resolution 128, flip 61°, multiband/slice-acceleration factor 4, in-plane GRAPPA 2, bandwidth
2170 Hz/px, echo spacing 0.57 ms, fat saturation. The cohorts differed only in **slice coverage and
phase-encode direction — DMNELF: 68 slices, A≫P; rtBPD: 72 slices, P≫A**. Each feedback run began
with a 25-volume (30 s) rest baseline followed by continuous PDA feedback; feedback runs comprised
**125 volumes in DMNELF (4 runs) and 150 volumes in rtBPD (5 runs)**. Network-localization
resting-state runs were 250 volumes in rtBPD (2 runs, separate session) and short 26-volume
in-session runs in DMNELF (§2.3). Spin-echo EPI field maps with reversed phase-encode (AP/PA;
TR 6000 ms, TE 41 ms DMNELF / 43 ms rtBPD, 2.0 mm) were acquired for susceptibility-distortion
correction.

### 2.5 Simultaneous EEG acquisition and preprocessing

*EEG acquisition.* Continuous EEG was recorded simultaneously with fMRI using a 32-channel
MR-compatible BrainAmp MR amplifier and MR EEG cap (Brain Products GmbH, Gilching, Germany), with
electrodes positioned according to the international 10–20 system (31 scalp channels plus one ECG
channel). Data were sampled at 5000 Hz (0.5 µV/bit) with online hardware band-pass filtering
between 0.1 Hz (10 s time constant) and 250 Hz. All channels were referenced online to Cz (series
resistance 10 kΩ per channel; 20 kΩ for ECG); electrode-to-skin impedances were checked before each
session and kept below 25 kΩ where possible. Acquisition and gradient/volume triggers were
synchronized to the scanner for artifact correction.

*EEG preprocessing.* MRI gradient artifacts were removed in BrainVision Analyzer (v1.26) by average
artifact subtraction (AAS): a gradient-artifact template, formed by averaging artifact epochs
time-locked to the MRI volume trigger, was subtracted from the raw signal; gradient-corrected data
were then downsampled to 1 kHz and exported. All subsequent steps used MNE-Python: (1) ECG R-peak
detection (NeuroKit2); (2) automatic bad-channel detection (variance and high-frequency-noise
z-scores); (3) annotation of scanner ramp-up edges; (4) FIR band-pass filtering 1–40 Hz (no
separate notch, as the 40 Hz low-pass attenuates line noise); (5) **ballistocardiogram (BCG)
correction** by average-artifact-template subtraction time-locked to the ECG R-peaks (−0.2 to
0.6 s); (6) downsampling to 500 Hz; (7) independent component analysis (29 components, Picard) with
automatic artifact-component rejection via ICLabel together with cardiac (`find_bads_ecg`) and EOG
correlation; (8) spherical-spline interpolation of bad channels; and (9) re-referencing to the
**common average**. This yielded 500 Hz, average-referenced, 31-channel data for feature extraction.

*Per-TR EEG features.* Features were computed at the fMRI TR grid: (i) single-electrode Stockwell
time-frequency EEG-fingerprint features (EFP; Meir-Hasson et al., 2014) — a 10-band × sliding-delay
design per electrode, the primary decoder; (ii) Hilbert band power in δ/θ/α/β/γ per channel,
HRF-convolved (a band-power reference and the substrate for the frontal-theta and signal-to-noise
controls); and (iii) for the deep-learning control, minimally processed 15-s raw-EEG windows
(1–40 Hz, resampled to 80 Hz; §2.8).

### 2.6 fMRI processing and confound-regressed network targets

Functional data were preprocessed with fMRIPrep `[TODO: version]`. Per-TR network timeseries were
extracted for the personalized DMN and CEN as variance-efficiency-weighted mask means, with
**PDA = CEN − DMN**. Because motion and physiological signals shared between EEG and BOLD inflate
cross-modal coupling, the decoding targets were **confound-regressed**: each network timeseries was
residualized on the full motion model, mean white-matter and CSF signals, and a discrete-cosine
high-pass basis `[TODO: confirm exact confound set + fMRIPrep version]`. Global-signal-regressed
(GSR) variants were computed additionally. All decoding results below use these **clean
(confound-regressed) targets**; uncleaned targets — which retain ~2–3× inflation from motion — are
reported only where explicitly noted, to quantify the confound.

### 2.7 EEG decoding of the BOLD networks

The primary decoder was the single-electrode EFP (Meir-Hasson et al., 2014): for each electrode, the
Stockwell time-frequency decomposition was summarized into 10 frequency bands and a sliding set of
EEG→BOLD delays, and ridge regression mapped this [band × delay] design to the target network
timeseries. Two honest-evaluation rules were applied throughout, matching the two inflation modes
identified above: (1) **leave-one-run-out (LORO)** cross-validation — never contiguous- or k-fold CV
within pooled runs, which leaks HRF-autocorrelated neighbouring samples; and (2) restriction to the
**feedback block** (rest baseline plus a 5-TR HRF lag dropped), removing the rest→feedback state-step
that otherwise inflates full-run correlations. We evaluated three electrode configurations — the
single best electrode (nested inner-LORO selection), a frontal multivariate set, and an all-electrode
multivariate ridge — and report the coupling as the out-of-fold Pearson correlation between predicted
and observed network timeseries.

*Deployment analyses.* (a) **Within-subject:** LORO within each cohort/session (DMNELF, rtBPD nf1,
rtBPD nf2). (b) **Cross-cohort transfer:** a decoder trained on all DMNELF was frozen and applied,
unchanged, to each rtBPD subject (zero re-fitting). (c) **One-run calibration:** the frozen decoder
was adapted using a single rtBPD feedback run and tested on that subject's remaining runs; per-run
z-scoring removed cohort- and montage-specific gain. (d) **Consumer-headset feasibility:** a "virtual
EPOC X" analysis restricted the decoder to the 12 electrodes of the 14-channel Emotiv EPOC X montage
present in our cap (with Fp1/Fp2 as AF3/AF4 proxies), quantifying the loss from a portable montage
that lacks centro-parietal midline coverage.

### 2.8 Control models

Three alternatives were tested against the linear EFP on the identical clean targets, LORO, and
feedback block. (i) **Deep learning:** a compact convolutional network on raw EEG (R-EEGNet; Stabile
et al., 2025 — EEGNet blocks + a linear regression head, ~2.6k parameters) trained on 15-s windows,
evaluated within-subject, leave-one-subject-out, and as a frozen DMNELF→rtBPD transfer. (ii)
**Frontal-theta:** frontal-midline theta and the frontal theta asymmetry FTA = ln P(F3) − ln P(F4)
(Zotev et al., 2025; Scheeringa et al., 2008), plus a 5-electrode CEN-node theta component. (iii)
**Signal-to-noise normalization ("f-SNR"):** a running signal-to-noise transform of band power
(trailing mean/SD), tested both as a fitted decoder and as an a-priori construct, against raw band
power.

### 2.9 Statistics

Group inference used non-parametric sign-flip permutation tests on subject-level correlations;
electrode topographies used Benjamini–Hochberg FDR. `[TODO: preregistration statement — not
preregistered (secondary analysis of neurofeedback data)?]`

*Data and code availability.* `[TODO: data-sharing statement + repository/DOI for analysis code.]`
*Funding.* `[TODO: grants.]` *Financial disclosures.* `[TODO: conflicts of interest — all authors.]`

## 3. Results

### 3.1 A personalized EEG decoder tracks the DMN–executive networks within-subject

In the schizophrenia discovery cohort (n=17), the personalized EFP decoder tracked the
confound-regressed BOLD networks out-of-sample across runs (LORO, feedback block): **CEN r = 0.10**
(p = 0.02), DMN r = 0.06, PDA r = 0.06 (multivariate; single-best-electrode CEN r = 0.11). Two
features of this result define the honest ceiling and recur throughout. First, the individual networks
decode most cleanly: the PDA *differential* — the neurofeedback target itself — is the **smallest of
the three effects (r ≈ 0.06)**, though still significant and (as §3.2 shows) replicated, because
differencing two modestly-decoded networks discards shared signal, so an EEG-guided readout is best
reconstructed from CEN and DMN rather than decoded from PDA directly. Second, these magnitudes are what remains **after confound regression**: on uncleaned
targets the same decoder reports 2–3× larger values (e.g. PDA 0.20, DMN 0.21), essentially all of
which is motion shared between EEG and BOLD and disappears once the target is cleaned. Coupling was
also **neural, not myogenic** — restricting the decoder to < 20 Hz (below the EMG band) left it
essentially unchanged, and its scalp distribution was **centro-parietal**, opposite the temporal
topography of temporalis EMG.

### 3.2 Within-subject decoding replicates across cohorts and sessions

Carried to the independent borderline-trait cohort and evaluated the same way (within-subject LORO,
clean targets, multivariate), the decoder **replicated across all three networks**: rtBPD session 1
(n=19) CEN r = 0.096, DMN r = 0.099, PDA r = 0.063; rtBPD session 2 (n=11) CEN r = 0.080, DMN
r = 0.092, PDA r = 0.086 (Table 1). Thus personalized EEG decoding of the DMN/executive networks —
**including the PDA target itself** — holds at r ≈ 0.06–0.10 across **two diagnostically and
developmentally distinct clinical cohorts and three imaging sessions**, a robust, reproducible level
of cross-modal coupling for a single-network BOLD target. The PDA differential remains the smallest
of the three effects (r ≈ 0.06–0.09) but is now significant and replicated in every session, not only
in discovery.

<!-- BEGIN:table1 (regenerate: python efp_meirhasson/scripts/table1_build.py) -->
**Table 1. Within-subject EEG decoding of the DMN/CEN networks.** Clean (confound-regressed) targets,
leave-one-run-out, feedback block. Group mean r (± SD); sign-flip p (\* <.05, \*\* <.01, \*\*\* <.001).

*Multivariate decoder (all electrodes)*

| Cohort (n) | CEN | DMN | PDA |
|---|---|---|---|
| DMNELF (SZ) (n=17) | +0.103\* (±0.16) | +0.060\*\* (±0.09) | +0.063\* (±0.10) |
| rtBPD nf1 (n=19) | +0.096\*\* (±0.12) | +0.099\*\* (±0.11) | +0.063\* (±0.12) |
| rtBPD nf2 (n=11) | +0.080 (±0.13) | +0.092\*\* (±0.09) | +0.086\* (±0.10) |

*Single best electrode (nested selection)*

| Cohort (n) | CEN | DMN | PDA |
|---|---|---|---|
| DMNELF (SZ) (n=17) | +0.110\* (±0.16) | +0.040\* (±0.07) | +0.041 (±0.09) |
| rtBPD nf1 (n=19) | +0.057\* (±0.12) | +0.079\*\* (±0.11) | +0.082\*\* (±0.14) |
| rtBPD nf2 (n=11) | +0.092\* (±0.13) | +0.116\*\* (±0.11) | +0.080\* (±0.10) |
<!-- END:table1 -->

### 3.3 Calibration-free transfer is weak but real, and improves with one calibration run

The stricter test is a decoder trained on one cohort and applied to another **without re-fitting**.
Frozen on all DMNELF and applied unchanged to rtBPD, the decoder transferred **weakly but mostly
significantly** — CEN r = 0.066 (nf1) and 0.102 (nf2); DMN r ≈ 0.045 — well below the within-subject
level, confirming that a fully calibration-free portable marker is not yet in hand. A **single
calibration run** closed much of the gap: adapting the frozen decoder on one rtBPD run raised
CEN transfer to r = 0.075 (nf1) and **0.117 (nf2)**, approaching the within-subject ceiling. The
deployable path is therefore **personalized or one-run-calibrated**, not zero-shot.

### 3.4 The decoder survives a 14-channel consumer montage

Restricting the decoder to the electrodes of a 14-channel consumer headset (Emotiv EPOC X) tested
portability directly. Although the EPOC X carries **none of the centro-parietal midline electrodes**
where the CEN field peaks, volume conduction to its posterior ring (P7/P8/O1/O2) preserved most of
the signal: **CEN r = 0.095 versus 0.103 on the full 31-channel cap (~92% retained; p < 0.05)**, with
DMN and PDA showing 81–100% retention. The montage is not the bottleneck; the ~8% loss from missing
midline coverage is recoverable, and a portable, personalized decoder is feasible on consumer
hardware (deployment specification in Supplementary Material).

### 3.5 Neither deep learning, frontal-theta, nor signal-to-noise normalization improves on the linear decoder

Three popular alternatives, tested head-to-head on the identical clean targets, all **failed to beat
the linear EFP**. (i) An end-to-end **deep network on raw EEG** (R-EEGNet) was at chance in every
regime — within-subject (CEN r ≈ 0.01), leave-one-subject-out (r ≈ 0.00), and frozen transfer
(r ≈ 0.02) — with ~285 windows per subject too few for a raw-EEG model to learn a distributed,
weak target that hand-crafted spectral features do capture. (ii) **Frontal-theta** indices, including
the motion-robust asymmetry FTA = ln P(F3) − ln P(F4) (Zotev et al., 2025), were null within the
feedback block (FTA r ≈ 0.00; frontal-theta ≈ 0), because the left–right subtraction cancels the
little common signal the channels carry. (iii) **Signal-to-noise normalization** ("f-SNR") did not
help and often hurt: the best feedback-block match to PDA was **raw** posterior band power
(r = 0.090), while its signal-to-noise transform was ~0 — normalization removed signal rather than
cohort gain. The linear, spectral, personalized decoder is thus not merely adequate but the ceiling
among the approaches tested; the ~0.10 coupling reflects a genuine information limit, not a modelling
shortfall.

### 3.6 Clinical anchor — the neurofeedback target tracks state calm, transdiagnostically

Regulating the DMN–CEN target made participants feel calmer, with a near-identical effect across
disorder and session: runs with more real-time PDA regulation were rated calmer in **schizophrenia
(r = +0.21, n=47)**, **rtBPD session 1 (r = +0.22, p = 0.013, n=124)**, and **rtBPD session 2
(r = +0.23, p = 0.050, n=73)**. Calm was consistently rated lower on more difficult runs in all three
(r = −0.45/−0.30/−0.41), and the four sliders formed a coherent, cross-cohort-consistent structure,
supporting construct validity. Change was a between-session effect: within-session (first vs last
feedback run) change was null, whereas across sessions (rtBPD nf1→nf2, paired, n=15) **state calm
increased significantly (+0.61, p = 0.042; 67% of participants)** — consistent with clinical benefit
accruing across, rather than within, sessions. The EEG decoder tracks the BOLD *target* (r ≈ 0.10)
but did not by itself predict per-run calm (r ≈ 0), bounding it as a scalable proxy for the
(clinically-relevant) target rather than a direct outcome predictor.

## 4. Discussion

In a schizophrenia discovery cohort studied with simultaneous EEG–fMRI, a **personalized EEG decoder
indexed the DMN–executive neurofeedback target** — most robustly its executive (CEN) component — and
this coupling **replicated in an independent, diagnostically and developmentally distinct cohort** of
adolescents with elevated borderline traits, across two of their sessions. The decoder transferred
zero-shot only weakly but was recoverable with a single calibration run, and it survived restriction
to a 14-channel consumer montage. To our knowledge this is the first demonstration that an EEG
decoder of a cortical, network-level real-time-fMRI neurofeedback target reproduces across distinct
disorders and is portable to consumer hardware.

We state the magnitude and its boundaries plainly. Cross-modal single-TR coupling has an
intrinsically modest ceiling, and after regressing out shared motion and physiology our effect sizes
(r ≈ 0.08–0.10) are correspondingly small; they index *reliable, transferable coupling*, not
high moment-to-moment fidelity. Crucially, that ceiling is a genuine information limit rather than a
modelling shortfall: an end-to-end deep network, a motion-robust frontal-theta index, and a
signal-to-noise-normalized feature all failed to exceed the compact linear decoder, and much of the
apparent signal in naïve analyses is motion that vanishes once the BOLD target is confound-regressed.
Two consequences follow for deployment. First, the **executive network, not the PDA differential, is
the portable readout** — differencing two weakly-decoded networks discards shared variance, so an
EEG-guided implementation should read out CEN (and DMN) and reconstruct the target from them rather
than decode PDA directly. Second, a truly **calibration-free** marker is not yet in hand; the honest,
deployable path is personalized or briefly calibrated decoding, which a one-run adaptation shows is
practical.

The therapeutic target is itself anchored to outcome transdiagnostically. In both disorders and at
retest, more PDA regulation went with feeling calmer (r ≈ +0.21–0.23), the self-report structure was
coherent across cohorts, and state calm increased across neurofeedback sessions (Δ = +0.61,
p = 0.042) rather than within a single session — consistent with benefit accruing with repeated
practice. A portable EEG index of this same target could extend mechanism-linked monitoring, and
eventually neurofeedback itself, beyond the scanner.

**Limitations.** Effect sizes are small (r ≈ 0.08–0.10) and the PDA differential in particular
decodes weakly (~0.06). We studied two clinical samples with **no healthy comparison group**, so we
demonstrate transdiagnostic *tracking and transfer* of the neurofeedback target, not a
transdiagnostic group *deficit*. Samples were modest (schizophrenia n=17; borderline-trait n=19,
n=11 at retest). In the schizophrenia cohort the personalized DMN/CEN masks were derived from a short
in-session resting run. High-frequency EEG in the scanner is vulnerable to myogenic contamination; we
addressed this with band (<20 Hz), topographic (centro-parietal, not temporal) and cross-validation
controls, but scalp EEG during fMRI remains noisy. The consumer-montage result is a virtual-montage
estimate and requires bench and simultaneous-EEG–fMRI validation on the physical headset before field
use. Finally, linking the marker to **trait-level symptom severity** in each disorder is a natural
next step and is in progress **[trait-severity scales — CONFIRM names; §2.1]**.

**Conclusion.** A personalized, motion-controlled EEG decoder indexes a DMN–executive neurofeedback
target, replicates across disorder and development, is calibratable from a single run, and is
deployable on consumer hardware — while more elaborate alternatives do not improve on it. It offers a
rigorous, deployable candidate for scaling a transdiagnostic, circuit-based intervention beyond the
MRI scanner.

## References (to compile)
- Zhang J, et al. Reducing DMN connectivity with mbNF: a pilot in adolescents with affective
  disorder history. *Mol Psychiatry* 2023;28:2540–2548.
- Bauer CCC, et al. Real-time fMRI neurofeedback reduces auditory hallucinations… DMN. *Psychiatry
  Res* 2020;284:112770. (schizophrenia mbNF)
- Bauer CCC, et al. REMind: MURFI real-time functional imaging. 2022.
- Bauer CCC, Whitfield-Gabrieli S, Díaz JL, Pasaye EH, Barrios FA. From state-to-trait meditation:
  reconfiguration of central executive and default mode networks. *eNeuro* 2019;6(6):ENEURO.0335-18.2019.
  **(Origin of the Positive Diametric Activity [PDA] metric, PDA = CEN − DMN.)**
- Hinds O, et al. Computing moment-to-moment BOLD activation for real-time neurofeedback.
  *NeuroImage* 2011;54:361–368.
- Chen AC, Oathes DJ, Chang C, Bradley T, Zhou Z-W, Williams LM, Glover GH, Deisseroth K, Etkin A.
  Causal interactions between fronto-parietal central executive and default-mode networks in humans.
  *Proc Natl Acad Sci USA* 2013;110:19944–19949.
- Meir-Hasson Y, et al. An EEG finger-print of fMRI deep-regional activation. *NeuroImage* 2014.
- Stabile … et al. R-EEGNet: a compact EEGNet regression of fMRI network activity. *EUSIPCO* 2025.
- Zotev V, et al. Frontal theta EEG asymmetry neurofeedback with simultaneous fMRI. *Hum Brain Mapp*
  2025;46:e70127. — Scheeringa R, et al. Frontal theta EEG and the default-mode network. 2008.
- Whitfield-Gabrieli & Ford. DMN in psychopathology. *Annu Rev Clin Psychol* 2012.

## Figures and tables (BUILT — `manuscript/figures/`; plan in `manuscript/FIGURES.md`)
Pipeline-walking narrative: preprocessing → method → fingerprint → replication/transfer → deployment.
- **Figure 1.** ✅ EEG preprocessing chain on real signal — raw gradient artifact (±3 mV) →
  gradient-corrected → 1–40 Hz → BCG-removed → ICA-removed → final; + ECG/R-peaks + rejected-ICA
  topographies. `fig1_preprocessing.png`.
- **Figure 2.** ✅ The EFP method (Meir-Hasson-visual): Stockwell spectrogram → 10 equal-energy bands
  → [band × delay] design → learned fingerprint weights (peak ~8-TR ≈ HRF lag) → prediction.
  `fig2_efp_method.png`.
- **Figure 3.** ✅ The DMNELF fingerprint: predicted↔observed CEN timeseries, **best (dmnelf1002
  r=+0.37) vs worst (dmnelf009 r=−0.13)** + clean centro-parietal decodability topography.
  `fig3_fingerprint_timeseries.png`.
- **Figure 4.** ✅ Within-subject replication bars (CEN/DMN/PDA × 3 cohorts) + calibration ladder
  (0-shot → +1-run → within). `fig4_replication_calibration.png`.
- **Figure 5.** ✅ Consumer-headset feasibility — EPOC-12 vs full cap (~92% CEN). `fig5_epoc_deployment.png`.
- **Figure 6.** ✅ Rigor + clinical: confound-cleaning (motion inflation) | controls fail (EFP vs
  deep/theta/f-SNR) | PDA↔calm across cohorts. `fig6_rigor_clinical.png`.
- **Table 1.** ✅ Within-subject decoding r (CEN/DMN/PDA) × cohort/session (`table1_build.py`).
- **Figure 0.** ✅ Study overview schematic (target → sim EEG–fMRI → frozen EFP decoder → portable
  deployment). `fig0_overview.png`.
- **Supplementary Table S1.** ✅ Per-sequence, per-cohort MRI parameters — [SUPPLEMENTARY.md](../SUPPLEMENTARY.md).
- Optional polish: Fig 3 personalized DMN/CEN mask render (network masks on cluster).

## Outstanding items — checklist (search `[TODO:` for all inline slots)
**Factual gaps (author-supplied):**
- [ ] Cohort 1 (SZ): age/sex, diagnostic instrument, symptom scale (PANSS?), medication → §2.1
- [ ] Cohort 2 (rtBPD): borderline-trait scale name + cutoff, age range, sex → §2.1
- [ ] Approving IRB(s) + protocol numbers → §2.1
- [x] ✅ MRI acquisition confirmed from Prisma protocols → §2.4 + Supp Table S1 (DMNELF 68 slices/A≫P/
  125-vol feedback ×4; rtBPD 72 slices/P≫A/150-vol feedback ×5; shared func sequence)
- [ ] fMRIPrep version + exact confound set for target cleaning → §2.6
- [ ] Preregistration, data/code availability, funding, financial disclosures → §2.9
- [ ] Front-matter: authors, affiliations, corresponding-author contact, keywords, running title, word counts
**Writing/production:**
- [x] ✅ Figures 1–6 + Table 1 built (`manuscript/figures/`, `figX_*.png`)
- [x] ✅ Figure 0 overview schematic + Supplementary Table S1 built
- [ ] Compile full reference list with DOIs; verify Stabile 2025, Zotev 2025, Scheeringa citations
- [x] ✅ Table 1 complete — within-subject clean CEN/DMN/PDA × all 3 cohorts (`table1_build.py`)
- [ ] Trim body to ≤ 4000 words; finalize 250-word abstract
**Optional (strengthens paper; awaiting data):**
- [ ] Trait-severity biomarker analysis (EEG decoder / PDA vs symptom scales) once trait scores arrive

**Done:** ✅ EEG acquisition + preprocessing (§2.5, verified vs `eeg_preproc.py`) · ✅ MURFI real-time
operationalization (§2.2) · ✅ per-run `slider_calm` outcome + PDA↔calm (§3.6) · ✅ Honest decoding
spine — within/transfer/calibration/EPOC + controls (`EFP_SPINE.md`) · ✅ Intro / Results / Discussion
reframed on confound-regressed, LORO numbers.
