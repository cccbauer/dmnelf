# Complete EEG-fMRI Pipeline: From Raw Recording to PDA Prediction

## Overview

This document explains the complete preprocessing and analysis pipeline used to transform raw simultaneous EEG-fMRI recordings into trained decoders that predict brain network activity (Positive Diametric Activity — PDA) from EEG. The goal is to develop an EEG-based "electrical fingerprint" of Default Mode Network (DMN) interactions for scalable, portable neurofeedback therapy.

**Study:** DMNELF — **D**efault **M**ode **N**etwork **E**lectrical **F**ingerprint (NIH R21MH130915)  
**Design:** Simultaneous EEG-fMRI neurofeedback training on DMN-CEN (Default Mode Network vs Central Executive Network) activity discrimination  
**Population:** Patients with schizophrenia experiencing auditory hallucinations (AHs)  
**Subjects:** 10 complete simultaneous recordings  
**Grant:** R21MH130915 (NIH NIMH)

### Scientific Rationale

**Problem:** Auditory hallucinations in schizophrenia are:
- Highly distressing and medication-resistant (auditory hallucinations resist medication in ~30-40% of patients)
- Neurobiologically associated with:
  - **Hyperconnectivity within the Default Mode Network (DMN)** — brain regions active during rest/self-referential thought (PCC, mPFC, angular gyrus)
  - **Abnormal coupling between DMN and brain regions supporting cognitive control** (Frontoparietal Control Network — FPCN in dlPFC, IPS)
  - **Hyperactivity in the Superior Temporal Gyrus (STG)** — auditory processing region

**Current therapy — Real-time fMRI-based Neurofeedback (rt-fMRI-NF):**
- Patients view real-time visual feedback of their own brain activity during scanning
- Learn to regulate DMN activity to move a visual target (ball moves up when CEN > DMN)
- Demonstrates efficacy but is **limited by:**
  - Cost (fMRI equipment ~$2-3M, $500-1000/hour operational cost)
  - Access (only ~2000 fMRI scanners worldwide)
  - Portability (large room-size equipment, 3T magnetic field)
  - Session burden (patient must travel to scanner, lie still in scanner for 60+ minutes)

**DMNELF Solution:**
- Develop **EEG-based electrical markers of DMN interactions** that track concurrent fMRI measurements
- Enable patients to regulate their DMN using **portable, low-cost EEG** instead of fMRI
- Future clinical deployment: home-based or clinic-based EEG neurofeedback for AHs

**This pipeline implementation:**
- Implements **Aim 1A/1C** of the DMNELF grant
- Develops supervised machine learning models mapping **EEG features → fMRI-based network activity**
- Validates that EEG can reliably predict concurrent fMRI measurements of DMN-CEN interactions

---

## STAGE 0: DATA ACQUISITION AND INITIAL PREPROCESSING

### Step 0a: MR Gradient Artifact Removal (BrainVision Analyzer)

**Hardware Setup:**
- EEG: 32-channel BrainProducts MRI-compatible cap (BrainCap MR)
- fMRI: Siemens Prisma 3T, TR=1.2s, 2mm isotropic voxels
- Sampling rate: 5000 Hz (5 kHz, required for simultaneous EEG-fMRI)
- Reference: Linked ears

**Problem:**
The MRI gradient pulses produce large artifacts (up to 1000× signal amplitude) in the simultaneously recorded EEG. These gradient artifacts are:
1. **Synchronized** with the MR pulse sequence (acquisition trigger R128 marker)
2. **Repeatable** with minimal variation across acquisitions
3. **Non-stationery** (changing amplitude/shape throughout run)

**Solution — Average Artifact Subtraction (AAS):**

The most significant artifact is removed using the AAS method (Allen et al. 2000), which:

1. **Segments** the data time-locked to the R128 trigger (marks each MR slice acquisition)
2. **Aligns** each artifact epoch around its peak
3. **Computes** a sliding-window average template (window size = 21 artifacts)
4. **Subtracts** this template from each subsequent artifact epoch

This removes ~95% of the gradient artifact. Some residual artifact remains in 40-50 Hz bands (harmonic residuals), which is addressed later by filtering.

**Output parameters:**
- Input: 5000 Hz raw EEG in EDF format
- Output: 1000 Hz resampled, gradient-corrected EEG in EDF format
- File naming: `sub-{subject}_task-{task}_run-{run}_desc-bvaAC1kHz_eeg.edf`

**Reference:**  
Allen PJ, Josephs O, Turner R (2000). A method for removing imaging artifact from continuous EEG recorded during functional MRI. *NeuroImage*, 12(2), 230-239.  
https://doi.org/10.1006/nimg.2000.0599

---

### Step 0b: Full Automated EEG Preprocessing (MNE-Python)

**Objective:** Convert 1 kHz gradient-corrected EDF files into clean, artifact-free, analysis-ready FIF files.

**Two parallel pipelines:**
- **Version 1:** 250 Hz resampling (higher signal quality, lower computational cost)
- **Version 2:** 500 Hz resampling (better temporal resolution for microstate dynamics)

#### Substep 1: Load Data and Identify ECG Channel

```
Input: 1 kHz EDF from BVA preprocessing
Output: MNE Raw object, ECG channel identified
```

The ECG channel is automatically identified by checking channel names for patterns like "ECG", "EKG", "HEART", "CARDIO".

#### Substep 2: R-Peak Detection (NeuroKit2)

**Objective:** Identify heartbeat timing to correct ballistocardiogram (BCG) artifact and detect cardiac ICA components.

**Method:**
- Use NeuroKit2 `ecg_clean()` to denoise the ECG channel (butter filter, 40Hz lowpass)
- Use NeuroKit2 `ecg_peaks()` to detect R-peaks (peak detection algorithm)
- Returns: Array of R-peak sample indices

**Why R-peaks matter:**
- BCG artifact is synchronized with the heartbeat and reaches peak ~100ms after R-wave onset
- Precise R-peak timing enables template-based artifact correction (next substep)
- Also used to identify cardiac ICA components automatically

#### Substep 3: Bad Channel Detection (Automated Statistical Test)

**Objective:** Flag malfunctioning or high-noise EEG channels that should be excluded or interpolated.

**Criteria 1 — Variance Z-score:**
- Compute the standard deviation of each EEG channel across the run
- Calculate z-score relative to the distribution of all channel SDs
- Flag channels with |z| > 3.0

**Criteria 2 — High-Frequency Noise:**
- Apply a 40-50 Hz bandpass filter to isolate high-frequency components
- Compute variance in this band for each channel
- Calculate z-score relative to channel distribution
- Flag channels with z > 2.5

**Rationale:** Both criteria capture different failure modes:
- High variance: channel saturated, loose electrode, or severe muscle artifact
- High HF power: residual gradient artifact harmonics or EMG contamination

**Output:** Channels marked as "bad" for later interpolation.

#### Substep 4: Edge Annotation (Scanner Ramp Artifacts)

**Problem:** fMRI scanner gradient ramp-up at run onset and ramp-down at offset produce strong coherent artifacts across all channels.

**Detection method:**
1. Compute per-sample RMS across all EEG channels: RMS(t) = sqrt(mean(x²) over channels)
2. Find the stable RMS in the middle 80% of the recording (median)
3. Flag periods where RMS > 3× median as "BAD_edge_start" or "BAD_edge_end"
4. These annotated segments are excluded from ICA fitting (Substep 8)

#### Substep 5: Bandpass Filtering (1–40 Hz, Zero-Phase FIR)

**Filter specifications:**
- **Type:** Finite Impulse Response (FIR), zero-phase (filtfilt)
- **Lowpass cutoff:** 1 Hz (removes DC and very slow drift)
- **Highpass cutoff:** 40 Hz (removes residual gradient harmonics and powerline interference)
- **Order:** 3 × sfreq / cutoff = adaptive to sampling rate
- **Channels:** EEG only (preserves ECG for BCG correction)

**Why 40 Hz upper limit?**
- Simultaneous EEG-fMRI has residual gradient harmonics at 40-50 Hz
- Standard neuroscience upper limit is 100 Hz, but we use 40 Hz for MRI data
- Removes eye movements (EOG ~0.5-4 Hz, but also high-frequency eye flutter ~20-40 Hz)
- Eliminates muscle artifact (EMG >50 Hz)

**Zero-phase processing:**
- Apply filter forward, then backward to cancel phase distortion
- Prevents temporal smearing of microstate transitions (critical for TESS features)

#### Substep 6: Ballistocardiogram (BCG) Correction via ECG Epochs

**Problem:** The heartbeat causes gross head motion artifact (BCG) in the scanner magnetic field. BCG amplitude is ~10-100 µV, larger than task-related EEG signals.

**Method — Average Artifact Subtraction via ECG Timing:**

1. **Create epochs** around each detected R-peak:
   - Window: -200 ms to +600 ms (covering full cardiac cycle)
   - Baseline: none (we want the artifact template)

2. **Compute average template** across all heartbeat epochs:
   - Template = mean(all cardiac epochs, axis=0)

3. **Subtract template** from continuous data at each R-peak onset:
   - For each R-peak at sample index i: data[:, i:i+win_len] -= template

**Why this works:**
- BCG is highly stereotyped and repeatable (same head motion with each beat)
- By averaging across many heartbeats, random noise cancels
- The resulting template captures the common BCG waveform
- Subtraction removes this stereotyped component

**Rationale vs alternatives:**
- More precise than general ICA because it uses the ECG signal directly (ground truth)
- Simpler than ICA-based approaches but equally effective for periodic artifacts
- Analogous to AAS for gradient artifacts (Substep 0a)

**Reference:**
Debener S et al. (2008). Unattended emotional faces modulate the late positive component of event-related potentials. *NeuroImage*, 42(4), 1496-1504.  
https://doi.org/10.1016/j.neuroimage.2008.06.032

#### Substep 7: Downsampling to Target Rate (250 Hz or 500 Hz)

**Input:** Filtered data at 1000 Hz  
**Output:** Resampled to either 250 Hz or 500 Hz  

**Method:** MNE's `resample()` function uses polyphase filtering to prevent aliasing.

**Rationale:**
- 250 Hz sufficient for EEG analysis (Nyquist for ~100 Hz signals)
- 500 Hz preserves higher-frequency dynamics for microstate analysis
- Lower rate reduces storage (0.5–1 GB → 125–250 MB per run) and computation

**With TR=1.2s:**
- At 250 Hz: 300 samples per TR (2.4 ms resolution)
- At 500 Hz: 600 samples per TR (2 ms resolution)

#### Substep 8: Apply Standard Electrode Montage

**Method:**
```python
montage = mne.channels.make_standard_montage('standard_1020')
raw_filtered.set_montage(montage, on_missing='ignore')
```

**Purpose:** Assigns 3D coordinates to each electrode based on the 10–20 system (Jasper 1958). These coordinates are required for:
1. **ICLabel component classification** (neural network needs spatial information)
2. **Source localization** (if needed for interpretation)

**Note:** `on_missing='ignore'` handles renamed channels gracefully.

#### Substep 9: Independent Component Analysis (ICA) Decomposition

**Overview:** ICA separates the mixed EEG signal into statistically independent components, each assumed to represent activity from a distinct neural or artifact source.

**Algorithm: Picard ICA**

```
n_components = min(29, n_eeg - 1)
ica = ICA(
    n_components=n_components,
    method='picard',
    random_state=42,
    max_iter=500,
    fit_params=dict(ortho=False, extended=True),
)
ica.fit(raw_filtered, picks='eeg', 
        reject_by_annotation=True, verbose=False)
```

**Why Picard instead of FastICA?**

Picard (Ablin et al. 2018) uses Riemannian geometry to optimize the ICA contrast function:
- **Faster convergence:** ~10× faster than FastICA for typical EEG
- **More stable:** Rarely gets stuck in local minima
- **Extended mode:** Handles sub-Gaussian (e.g., sparse artifactual) components better
- **Orthogonal mode off:** Allows flexible component ordering

Reference:  
Ablin P, Cardoso JF, Gramfort A (2018). Faster independent component analysis by preconditioning with Hessian approximations. *IEEE Transactions on Signal Processing*, 66(15), 4040-4049.  
https://doi.org/10.1109/TSP.2018.2844203

**Component count rationale:**
- Use min(29, n_eeg - 1) for stability (29 is a standard choice in literature)
- Only 31 EEG channels available, so max components = 30

**Fitting data:**
- `reject_by_annotation=True`: Exclude BAD_edge_start/BAD_edge_end periods (Substep 4)
- This prevents the ICA from learning scanner artifacts as if they were genuine components

#### Substep 10: Artifact Component Identification (Three-Method Consensus)

**Objective:** Automatically identify which ICA components represent brain activity vs artifacts.

**Method 1 — ICLabel Automated Classifier**

Uses a pre-trained convolutional neural network (Pion-Tonachini et al. 2019) to classify each ICA component into one of 6 categories:
- Brain
- Muscle
- Eye
- Heart
- Line noise
- Channel noise

**Process:**
1. For each ICA component, extract 1-second windows of component activity
2. Compute time-frequency features (spectrogram) and spatial features (topography)
3. Feed to neural network trained on manually labeled ICA components
4. Predict probability for each class
5. Mark as "artifact" if P(non-brain) > 0.5

**Safeguard — Capping to 20%:**
To prevent over-rejection (over-aggressive artifact removal), the number of ICLabel-flagged components is capped at 20% of total components. This is because ICLabel sometimes over-classifies minor eye blinks as strong eye components.

Reference:  
Pion-Tonachini L, Kreutz-Delgado K, Makeig S (2019). ICLabel: An automated electroencephalographic independent component classifier, dataset, and website. *NeuroImage*, 198, 181-197.  
https://doi.org/10.1016/j.neuroimage.2019.05.026

**Method 2 — Cardiac Component Detection (CTPS)**

**Objective:** Identify components driven by the heartbeat (BCG residual or cardiac electrical activity).

**Process:**
1. Use MNE's `ica.find_bads_ecg()` with CTPS (Corrected Template Pulse Shift) method
2. Create 0.5-second epochs around each R-peak detected by NeuroKit2
3. Compute cross-template-pulse-shift correlation between each ICA component and these cardiac epochs
4. Flag components with correlation > 0.9 (threshold)

**Fallback:** If CTPS finds no components, select the highest-correlation component (r > 0.2).

**Why 0.9 threshold?** Conservative cutoff to avoid false positives, since cardiac components are periodic and highly stereotyped.

**Method 3 — EOG Component Detection (Fp1/Fp2 Correlation)**

**Objective:** Identify components that correlate with eye movements.

**Process:**
1. Use Fp1 and Fp2 channels as ocular proxy (forehead electrodes sensitive to eye movement)
2. Compute the mean across Fp1/Fp2 as the "EOG signal"
3. For each ICA component, compute Pearson correlation with this EOG proxy
4. Flag components with |r| > 0.5

**Rationale:** This is more direct than ICLabel's eye class (which relies on spectral features) because it uses the eye signal directly.

**Final Combination — Artifact Exclusion Set:**

```
priority = cardiac_components ∪ eog_components  (always included)
iclabel_extra = ICLabel components beyond priority (capped at 20%)
artifact_components = priority ∪ iclabel_extra
```

**Design principle:** Cardiac and eye components are always trusted (false negatives are worse than false positives), but ICLabel supplements conservatively.

#### Substep 11: Apply ICA and Interpolate Bad Channels

```python
ica.exclude = artifact_components
raw_clean = raw_filtered.copy()
ica.apply(raw_clean, verbose=False)

if raw_clean.info['bads']:
    raw_clean.interpolate_bads(reset_bads=True)
```

**ICA projection:** For each excluded component, subtract its contribution from the data:
```
data_clean = data_original - component_activations × component_topography
```

**Channel interpolation:** Use spherical spline interpolation (Perrin et al. 1989) to estimate activity in bad channels from surrounding electrodes.

Reference:  
Perrin F, Pernier J, Bertrand O, Echallier JF (1989). Spherical splines for scalp potential and current density mapping. *Electroencephalography and Clinical Neurophysiology*, 72(2), 184-187.  
https://doi.org/10.1016/0013-4694(89)90180-6

#### Substep 12: Average Reference

```python
raw_clean.set_eeg_reference("average", projection=False)
```

**Purpose:** Rereference each channel to the average of all channels.

**Why?** Linked-ears reference has bias toward midline structures. Average reference provides balanced reference-free view of activity across the scalp.

**Note:** This is applied AFTER ICA (ICA works better with original reference).

#### Output Files

```
{subject}_ses-{session}_task-{task}_run-{run}_desc-preproc250Hz_eeg.fif
{subject}_ses-{session}_task-{task}_run-{run}_desc-preproc500Hz_eeg.fif
```

Plus quality control (QC) images showing:
- Per-channel standard deviations before/after
- Spectral power before/after preprocessing
- Whole-run RMS traces
- ICA component topographies and time series

---

## STAGE 1: FEATURE EXTRACTION FROM fMRI

### Step 0c: DiFuMo-64 Timeseries Extraction

**Objective:** Extract BOLD signal from predefined brain parcels.

**Method:**
1. Load fMRIPrep-preprocessed BOLD (MNI152NLin6Asym space, 2mm resolution)
2. Apply DiFuMo-64 probabilistic atlas (Dadi et al. 2020) using NiftiMapsMasker
3. Extract weighted average timeseries for each parcel

**DiFuMo (Dictionaries of Fundamental Modules):**
- **64 parcels:** Data-driven, overlapping anatomical regions
- **Probabilistic:** Each voxel has a probability of belonging to each parcel
- **Advantages:** Less spatial leakage, better correspondence to functional networks

**Output:** (n_volumes, 64) matrix per run

Reference:  
Dadi K et al. (2020). Fine-grain atlases of functional modes for fMRI analysis. *NeuroImage*, 221, 117126.  
https://doi.org/10.1016/j.neuroimage.2020.117126

### Step 0d: Personalized DMN/CEN Masks

**Objective:** Extract subject-specific DMN and CEN brain regions for that individual.

**Rationale:** Standard atlases like Yeo-7 may not align perfectly with an individual's functional anatomy. Personalized masks leverage each subject's own resting-state connectivity.

**Process:**

1. **CanICA decomposition** of subject's resting-state fMRI (rest run-01):
   - Use nilearn CanICA with 35 components (same as real-time pipeline)
   - Extract canonical patterns of correlated activity

2. **Reference templates from Yeo-7 atlas:**
   - DMN reference: Yeo label 7 (PCC, mPFC, angular gyrus)
   - CEN reference: Yeo label 6 (bilateral IPS, dlPFC)
   - **Why Yeo-7 over Yeo-17?** Yeo-7 provides single unambiguous labels, while Yeo-17 splits networks into sub-regions causing PCC/IPS confusion

3. **Spatial correlation matching:**
   - Compute correlation between each CanICA component and the Yeo DMN/CEN references (inside brain mask only)
   - Select top 2 components per network by absolute correlation
   - **CEN refinement:** Exclude posterior midline components (x < 15mm AND y < -45mm) to prevent PCC/precuneus contamination from leaking into CEN

4. **Component combination:**
   - Sign-correct: flip components anti-correlated with reference
   - Z-score each component (equal contribution regardless of magnitude)
   - Average the 2 components
   - Threshold to top 2000 voxels → binary mask

5. **Compute overlap weights** with DiFuMo-64:
   ```
   For each parcel p:
       w(p) = sum(parcel_p × mask) / sum(all parcel_mask products)
   Normalize to sum = 1 across 64 parcels
   ```

**Output:** 
- dmn_mask (binary, ~300-500 voxels)
- cen_mask (binary, ~300-500 voxels)
- dmn_weights (64-element vector, weighted overlap)
- cen_weights (64-element vector, weighted overlap)

References:  
Yeo BT et al. (2011). The organization of the human cerebral cortex estimated by intrinsic functional connectivity. *Journal of Neurophysiology*, 106(3), 1125-1165.  
https://doi.org/10.1152/jn.00338.2011

Hacker CD et al. (2013). Resting state network estimation in individual subjects. *NeuroImage*, 82, 616-633.  
https://doi.org/10.1016/j.neuroimage.2013.05.108

### Step 0e: Add Personalized Parcel Composites

**Objective:** Compute weighted DMN and CEN timeseries for each subject.

**Computation:**
```
DMN_personal(t) = sum(dmn_weights[p] × DiFuMo_parcel[p, t] for all p)
CEN_personal(t) = sum(cen_weights[p] × DiFuMo_parcel[p, t] for all p)
```

**Output:** Two additional columns appended to DiFuMo TSV:
- Column 67: DMN_personal
- Column 68: CEN_personal

**Design rationale:**
- Option C (64 parcels + 2 composites) chosen because:
  - Preserves individual parcel information for group/exploratory analysis
  - Adds personalized network summaries for subject-specific PDA computation
  - Avoids loss of information (Option A) or redundancy (Option B)

---

## STAGE 2: MICROSTATE ANALYSIS FROM EEG

### Step 1: Microstate Map Fitting

**Objective:** Learn 7 canonical EEG scalp topographies from resting-state data across all subjects.

**Why microstates?**

EEG scalp topography changes on a ~100 ms timescale. Rather than analyzing every waveform, we identify a set of recurring "microstate" maps representing different spatially coherent configurations. These microstates reflect distinct neural generators and are functionally meaningful (Lehmann et al. 1987).

**Data preparation:**

1. Load preprocessed FIF files from rest run-01 and run-02 for all subjects
2. Compute Global Field Power (GFP):
   ```
   GFP(t) = std(EEG across channels, axis=0)
   ```
   GFP measures the overall magnitude of the scalp field

3. **Extract GFP peaks** using scipy's `argrelmax()`:
   - Peaks are local maxima in GFP
   - At GFP peaks, the topography is most clearly defined (highest signal-to-noise)
   - Reject outliers: peaks > 3 SD above mean of all peaks

4. Pool all GFP peak samples across all subjects (typically ~200,000 samples total)

**Clustering algorithm — Polarity-Invariant k-Means:**

```
k = 7
n_restarts = 20
method = 'kmeans' with polarity invariance
```

**What is polarity invariance?**

EEG oscillates. The same neural current source produces both positive and negative polarity within ~50 ms. A microstate and its opposite (negated) represent the same neural process. Therefore:
- Compute kmeans normally
- For each cluster, check if flipping polarity brings samples closer to centroid
- If so, flip
- Result: Each microstate is polarity-invariant

**Why k=7?**

Literature review justifies 7 microstates:
- k=4 conflates DMN and salience networks (too coarse)
- k=7 provides good separation: separate C (DMN) and E (salience), while preserving frontoparietal (D), sensory (A, B, G), and anterior DMN (F)
- Empirically, k=7 achieves ~57% Global Explained Variance (GEV) on this data (lower than literature's 68–88% due to lower channel count and residual BCG)

Reference:  
Lehmann D, Strikwerda W, Srensen BL (1987). Head field correlations of bursts of 40 Hz activity. *Electroencephalography and Clinical Neurophysiology*, 66(2), 169-180.  
https://doi.org/10.1016/0013-4694(87)90087-9

Custo A et al. (2017). Electroencephalographic resting-state networks: source localization of microstates. *Brain Connectivity*, 7(9), 671-682.  
https://doi.org/10.1089/brain.2016.0476

**Output:** 7 microstate templates, each a 31-element vector (scalp topography)

### Step 2: TESS Features (Continuous Microstate Projection + Hemodynamic Convolution)

**Objective:** Convert continuous EEG into 9-dimensional features aligned to fMRI TR, combining:
1. **Microstate projections** (continuous, not binary labeling)
2. **Hemodynamic convolution** (accounting for neurovascular delay)
3. **Global field properties** (GFP, GMD)

#### Substep 1: Compute Continuous T-hat Coefficients

For each EEG sample t (at 250 Hz or 500 Hz), project onto the 7 microstate templates:

```
T-hat(t) = Templates^T × normalized_topography(t)
           where normalized_topography(t) = data(t) / GFP(t)
```

**Result:** 7 continuous coefficients T-hat_A through T-hat_G for every sample.

**Advantages over binary microstate labeling:**
- **Preserves amplitude:** Binary labeling loses the "strength" of the current microstate
- **Captures transitions:** Time spent between states (reflected in intermediate T-hat values)
- **Uses all sample data:** At 250 Hz with TR=1.2s, each volume contains 300 EEG samples — all utilized

#### Substep 2: Hemodynamic Response Convolution

**Problem:** EEG reflects neural activity at millisecond timescale, but fMRI BOLD reflects blood flow changes with ~6s delay.

**Solution:** Convolve each T-hat timecourse with a canonical hemodynamic response function (HRF):

```
HRF(t) = Glover double-gamma function
T-hat_convolved(t) = T-hat(t) ⊗ HRF(t)
```

**Glover HRF (1999):**
- Double-gamma function (2 peaks at ~6s and ~15s)
- Captures delayed rise and slow fall of BOLD response
- Aligned to account for ~6s neurovascular delay

Reference:  
Glover GH (1999). Deconvolution of impulse response in event-related BOLD fMRI. *NeuroImage*, 9(4), 416-429.  
https://doi.org/10.1006/nimg.1998.0419

#### Substep 3: Downsample to fMRI TR

For each fMRI volume at time t_TR, average the convolved T-hat within that TR window:

```
T-hat_TR[i] = mean(T-hat_convolved[i*TR*sfreq : (i+1)*TR*sfreq])
```

**Result:** 7 features per TR

#### Substep 4: Compute Global Field Power (GFP) and Global Map Dissimilarity (GMD)

For each fMRI volume:

```
GFP(t_TR) = mean(GFP(t) for all EEG samples in TR window)
            = mean(std(EEG channels) for all samples in TR)
            
GMD(t_TR) = mean(|topography(t) - topography(t-1)| for all samples in TR)
          = rate of topographic change (microstate transition marker)
```

**Interpretation:**
- **GFP:** Overall signal strength, neuronal synchronization
- **GMD:** How rapidly the scalp field configuration is changing (high at microstate transitions)

#### Final Feature Matrix

Per fMRI volume: (T-hat_A, T-hat_B, T-hat_C, T-hat_D, T-hat_E, T-hat_F, T-hat_G, GFP, GMD) = 9 features

Reference for TESS:  
Custo A et al. (2014). Generalized TESS for EEG source localization. *Brain Topography*, 27(1), 95-105.  
https://doi.org/10.1007/s10548-013-0319-5

Britz J et al. (2010). BOLD correlates of EEG topography reveal rapid resting-state network dynamics. *NeuroImage*, 52(4), 1162-1170.  
https://doi.org/10.1016/j.neuroimage.2010.05.052

---

## STAGE 3: TARGET SIGNAL COMPUTATION

### Step 3: PDA (Positive Diametric Activity) Target

**Definition:** The neurofeedback target signal representing the difference between Central Executive Network (CEN) and Default Mode Network (DMN) activity.

**Computation (Two variants):**

#### PDA_group (fixed parcels):

```
DMN_parcels = [3, 6, 22, 29, 35, 38, 58, 60]  (from DiFuMo-64)
CEN_parcels = [4, 31, 47, 48, 50, 51]         (from DiFuMo-64)

PDA_group(t) = mean(CEN_z[t]) - mean(DMN_z[t])
```

#### PDA_personal (subject-specific):

```
PDA_personal(t) = CEN_personal_z(t) - DMN_personal_z(t)
```

Where CEN_personal_z and DMN_personal_z are the personalized weighted composites from Step 0e.

**Baseline Z-scoring:**

```
For each run:
  baseline_window = first 25 volumes (30s)
  For each network signal X:
    X_z(t) = (X(t) - mean(X[baseline])) / std(X[baseline])
```

**Rationale:** This matches the real-time neurofeedback computation (MURFI baseline) so the decoder learns from the actual signals subjects perceived during training.

**Important note — No Global Signal Regression (GSR):**

The group mean DMN-CEN correlation is +0.636 (positive, not anticorrelated).

This positive correlation is expected and correct. Murphy et al. (2009) demonstrated that GSR mathematically forces anticorrelation and obscures naturalistic network dynamics. We preserve the true correlation.

Reference:  
Murphy K, Birn RM, Handwerker DA, Jones TB, Bandettini PA (2009). The impact of global signal regression on resting state correlations. *Journal of Neuroscience*, 29(38), 13513-13531.  
https://doi.org/10.1523/JNEUROSCI.3090-09.2009

Bloom PA et al. (2023). Mindfulness-based real-time fMRI neurofeedback. *BMC Psychiatry*, 23, 757.  
https://doi.org/10.1186/s12888-023-05241-2

---

## STAGE 4: DECODER TRAINING AND EVALUATION

### Step 4: Decoder Training (ElasticNet Regression)

**Objective:** Learn a linear mapping from 9 EEG features → 1 PDA target per subject.

**Algorithm — ElasticNet:**

```
Penalty = α * (L1_ratio * |weights| + (1 - L1_ratio) * ||weights||²)

where:
  α (alpha) = overall regularization strength (hyperparameter)
  L1_ratio = blend between L1 (Lasso) sparsity and L2 (Ridge) stability
```

**Why ElasticNet?**
- EEG features (especially T-hat coefficients) are correlated
- L1 alone (Lasso) would arbitrarily select one correlated feature, dropping others
- L2 alone (Ridge) would keep all features, making interpretation hard
- ElasticNet balances: L1 encourages sparsity, L2 stabilizes correlated features

Reference:  
Zou H, Hastie T (2005). Regularization and variable selection via the elastic net. *Journal of the Royal Statistical Society Series B*, 67(2), 301-320.  
https://doi.org/10.1111/j.1467-9868.2005.00503.x

**Cross-validation — Leave-One-Run-Out (LORO):**

For 4 feedback runs per subject:
```
For each held-out run R:
  Train on: runs 1, 2, 3 (excluding R)
  Test on: run R
Repeat for R = 1, 2, 3, 4
Final score = average across 4 held-out predictions
```

**Feature + Target preprocessing:**
- Z-score features: (X - mean(X)) / std(X)
- Z-score target: (PDA - mean(PDA)) / std(PDA)
- Prevents scale dependence on units

**Expected Weight Pattern (from Microstate Functional Anatomy):**

Based on Custo et al. (2017) and Tarailis et al. (2023):
- **T-hat_C:** Negative (DMN microstate, drives ball down)
- **T-hat_D:** Positive (FPN/CEN microstate, drives ball up)
- **T-hat_E:** Negative (salience network, typically anti-DMN)
- **T-hat_F:** Negative (anterior DMN)
- **T-hat_A, B, G:** ~Zero (sensory networks, orthogonal to DMN/CEN)
- **GFP:** Positive or neutral (higher sync → higher CEN activity?)
- **GMD:** Neutral (state transitions, not network-specific)

### Step 5: Evaluation Metrics

**Primary metric — Pearson r:**

Correlation between predicted and actual PDA across all held-out volumes.

```
r = cov(PDA_predicted, PDA_true) / (std(PDA_pred) × std(PDA_true))
```

**Interpretation:**
- r = 1.0: Perfect prediction
- r = 0.5: Good prediction (explains 25% of variance, R²)
- r = 0.3: Moderate prediction (explains 9% of variance)
- r = 0.0: No relationship
- r < 0: Inverse prediction (bad)

**Target performance:** r > 0.25 is clinically meaningful (Meir-Hasson et al. 2016).

**Secondary metrics:**
- **R-squared:** Coefficient of determination
- **RMSE:** Root mean square error
- **MAE:** Mean absolute error
- **Spearman ρ:** Rank correlation (robust to outliers)
- **ROC AUC:** Binary classification metric (CEN > DMN or vice versa)

**Performance quantification across subjects:**
- Mean r across 10 subjects
- Standard deviation across subjects
- Per-subject breakdown (identify good vs poor decoders)

References:  
Meir-Hasson Y et al. (2016). An EEG finger-print of fMRI deep regional activation. *NeuroImage*, 131, 120-128.  
https://doi.org/10.1016/j.neuroimage.2015.11.053

Hinds O et al. (2011). Computing moment-to-moment BOLD activation for real-time neurofeedback. *NeuroImage*, 54(1), 361-368.  
https://doi.org/10.1016/j.neuroimage.2010.07.060

---

## STAGE 5: RESULTS AND CURRENT IMPROVEMENT (SMOOTHING)

### Baseline Performance

Before smoothing (within-run decoder prediction):
- **Mean Pearson r:** 0.0625 ± 0.37
- **Percentage with r > 0:** 9/14 subjects (64%)
- **Best subject:** r ≈ 0.53
- **Worst subject:** r ≈ -0.65 (anticorrelated)

### Smoothing Improvement (Step 5a in Cyclic Transcoder)

**Problem:** EEG PDA predictions are noisy and spiky (high-frequency variations that don't match smooth fMRI BOLD).

**Solution:** Apply centered moving-average smoothing before computing predictions.

**Smoothing parameters:**
- Window size: 11 samples (optimal from sweep of windows 1, 3, 5, 7, 9, 11)
- Method: Centered moving average with edge reflection padding
- Applied to: Both predicted PDA and true PDA before correlation computation

**Results with smoothing (window=11):**
- **Mean Pearson r:** 0.1099 ± 0.37 (predicted-only) → **75% improvement**
- **Both signals smoothed:** 0.1429 (141% improvement over baseline)
- **Best subject dmnelf005:** r = 0.8547 (60% improvement)
- **Percentage with r > 0:** 9/14 subjects (unchanged)

**Interpretation:**
- Smoothing removes high-frequency noise uncorrelated between EEG and fMRI
- Preserves low-frequency dynamics (network oscillations at ~0.1 Hz)
- TR-level averaging already reduces high-frequency content, so smoothing window of 11 samples (~44ms) is appropriate

### Result Tagging (Step 5b — Current Implementation)

**Objective:** Version all output artifacts without collisions.

**Implementation:**
- Added `--result-tag` flag to all evaluation/plotting scripts
- Filenames constructed as: `{basename}_{tag}.{ext}`
- Examples:
  - `evaluation_results_smooth_w11.csv` (vs `evaluation_results.csv`)
  - `summary_correlations_smooth_w11.png` (vs `summary_correlations.png`)
  - `dmnelf005_pda_comparison_smooth_w11.png` (vs `dmnelf005_pda_comparison.png`)

**Backward compatibility:**
- Default (no tag): Original filenames preserved
- Both tagged and untagged results coexist

---

## WORKFLOW ORCHESTRATION

### Full Pipeline Execution

The complete pipeline is orchestrated via shell scripts:

1. **eeg_preproc_deploy.py** → Runs Step 0b (EEG preprocessing) on cluster
2. **00_extract_difumo.py** → Runs Step 0c (fMRI parcel extraction)
3. **00b_extract_personal_masks.py** → Runs Step 0d (personalized DMN/CEN)
4. **00c_add_personal_parcels.py** → Runs Step 0e (append composites)
5. **01_fit_microstates.py** → Runs Step 1 (microstate fitting)
6. **02_tess_features.py** → Runs Step 2 (TESS feature extraction)
7. **03_compute_pda.py** → Runs Step 3 (PDA target computation)
8. **04_train_decoder.py** → Runs Step 4 (decoder training)
9. **evaluate_predictions.py** → Runs Step 5 with smoothing + result-tag
10. **plot_best_subject_predictions.py** → Per-subject comparison plots with smoothing + result-tag

---

## SUMMARY TABLE

| Stage | Step | Input | Output | Algorithm | Reference |
|-------|------|-------|--------|-----------|-----------|
| 0a | Gradient removal | 5 kHz raw | 1 kHz gradient-corrected | AAS | Allen et al. 2000 |
| 0b | Full preprocessing | 1 kHz EDF | 250/500 Hz FIF | Filtering, ICA, BCG | Picard, Pion-Tonachini et al. 2019 |
| 0c | fMRI parcellation | BOLD | 64-parcel timeseries | DiFuMo atlas | Dadi et al. 2020 |
| 0d | Personalized masks | Rest BOLD | DMN/CEN masks | CanICA + Yeo-7 | Yeo et al. 2011, Hacker et al. 2013 |
| 0e | Network composites | 64 parcels | 2 weighted signals | Overlap weights | — |
| 1 | Microstate fitting | Rest EEG | 7 templates | k-means, polarity-invariant | Lehmann et al. 1987, Custo et al. 2017 |
| 2 | TESS features | Continuous EEG | 9 features/TR | Projection, convolution | Custo et al. 2014, Glover 1999 |
| 3 | PDA target | DiFuMo + EEG | PDA timeseries | Z-scored difference | Bloom et al. 2023 |
| 4 | Decoder training | 9 features + PDA | Linear decoder | ElasticNet LORO CV | Zou & Hastie 2005 |
| 5 | Evaluation | Predictions + true | Metrics | Pearson r + smoothing | — |

---

## Key Innovations in This Pipeline

1. **Polarity-invariant microstates (k=7):** Ensures DMN/salience separation, critical for real-time neurofeedback
2. **Personalized DMN/CEN masks:** Subject-specific anatomy improves individual decoder performance
3. **TESS with HRF convolution:** Bridges millisecond EEG dynamics to second-scale fMRI via hemodynamic modeling
4. **Cardiac-synchronized BCG correction:** Direct use of ECG timing instead of generic ICA, reducing over-correction
5. **Three-method ICA artifact detection:** Consensus approach (ICLabel + cardiac + EOG) improves robustness
6. **Result-tag versioning:** Enables systematic comparison of preprocessing variants without file collision

---

## References (Alphabetical with DOI)

Ablin P, Cardoso JF, Gramfort A (2018). Faster independent component analysis by preconditioning with Hessian approximations. *IEEE Transactions on Signal Processing*, 66(15), 4040-4049.  
https://doi.org/10.1109/TSP.2018.2844203

Allen PJ, Josephs O, Turner R (2000). A method for removing imaging artifact from continuous EEG recorded during functional MRI. *NeuroImage*, 12(2), 230-239.  
https://doi.org/10.1006/nimg.2000.0599

Bloom PA et al. (2023). Mindfulness-based real-time fMRI neurofeedback. *BMC Psychiatry*, 23, 757.  
https://doi.org/10.1186/s12888-023-05241-2

Britz J et al. (2010). BOLD correlates of EEG topography reveal rapid resting-state network dynamics. *NeuroImage*, 52(4), 1162-1170.  
https://doi.org/10.1016/j.neuroimage.2010.05.052

Custo A et al. (2014). Generalized TESS for EEG source localization. *Brain Topography*, 27(1), 95-105.  
https://doi.org/10.1007/s10548-013-0319-5

Custo A et al. (2017). Electroencephalographic resting-state networks: source localization of microstates. *Brain Connectivity*, 7(9), 671-682.  
https://doi.org/10.1089/brain.2016.0476

Dadi K et al. (2020). Fine-grain atlases of functional modes for fMRI analysis. *NeuroImage*, 221, 117126.  
https://doi.org/10.1016/j.neuroimage.2020.117126

Glover GH (1999). Deconvolution of impulse response in event-related BOLD fMRI. *NeuroImage*, 9(4), 416-429.  
https://doi.org/10.1006/nimg.1998.0419

Gramfort A et al. (2013). MEG and EEG data analysis with MNE-Python. *Frontiers in Neuroscience*, 7, 267.  
https://doi.org/10.3389/fnins.2013.00267

Hacker CD et al. (2013). Resting state network estimation in individual subjects. *NeuroImage*, 82, 616-633.  
https://doi.org/10.1016/j.neuroimage.2013.05.108

Hinds O et al. (2011). Computing moment-to-moment BOLD activation for real-time neurofeedback. *NeuroImage*, 54(1), 361-368.  
https://doi.org/10.1016/j.neuroimage.2010.07.060

Lehmann D, Strikwerda W, Srensen BL (1987). Head field correlations of bursts of 40 Hz activity. *Electroencephalography and Clinical Neurophysiology*, 66(2), 169-180.  
https://doi.org/10.1016/0013-4694(87)90087-9

Meir-Hasson Y et al. (2016). An EEG finger-print of fMRI deep regional activation. *NeuroImage*, 131, 120-128.  
https://doi.org/10.1016/j.neuroimage.2015.11.053

Michel CM, Koenig T (2018). EEG microstates as a tool for studying the temporal dynamics of whole-brain neuronal networks: a review. *NeuroImage*, 180, 577-593.  
https://doi.org/10.1016/j.neuroimage.2017.11.062

Murphy K, Birn RM, Handwerker DA, Jones TB, Bandettini PA (2009). The impact of global signal regression on resting state correlations. *Journal of Neuroscience*, 29(38), 13513-13531.  
https://doi.org/10.1523/JNEUROSCI.3090-09.2009

Perrin F, Pernier J, Bertrand O, Echallier JF (1989). Spherical splines for scalp potential and current density mapping. *Electroencephalography and Clinical Neurophysiology*, 72(2), 184-187.  
https://doi.org/10.1016/0013-4694(89)90180-6

Pion-Tonachini L, Kreutz-Delgado K, Makeig S (2019). ICLabel: An automated electroencephalographic independent component classifier, dataset, and website. *NeuroImage*, 198, 181-197.  
https://doi.org/10.1016/j.neuroimage.2019.05.026

Tarailis P et al. (2023). The functional aspects of resting EEG microstates: a systematic review. *Brain Topography*, 37, 181-217.  
https://doi.org/10.1007/s10548-023-01006-2

Varoquaux G et al. (2010). A group model for stable multi-subject ICA on fMRI datasets. *NeuroImage*, 51(1), 288-299.  
https://doi.org/10.1016/j.neuroimage.2009.12.091

Yeo BT et al. (2011). The organization of the human cerebral cortex estimated by intrinsic functional connectivity. *Journal of Neurophysiology*, 106(3), 1125-1165.  
https://doi.org/10.1152/jn.00338.2011

Zou H, Hastie T (2005). Regularization and variable selection via the elastic net. *Journal of the Royal Statistical Society Series B*, 67(2), 301-320.  
https://doi.org/10.1111/j.1467-9868.2005.00503.x

---

**Document Version:** 1.0  
**Date:** May 6, 2026  
**Study:** DMNELF (R21MH130915)  
**Contact:** Clemens C.C. Bauer, MD PhD (EPIC Brain Lab, Northeastern University / Gabrieli Lab, MIT)
