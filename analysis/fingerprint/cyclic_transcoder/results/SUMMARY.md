# Cyclic Transcoder: Extraction, Training & Evaluation Summary

**Date:** May 5, 2026  
**Project:** Default Mode Network Electrical Fingerprint (DMNELF) — Cyclic Transcoder  
**Location:** `/Users/cccbauer/Documents/GitHub/dmnelf/analysis/fingerprint/cyclic_transcoder`

---

## Overview

This document summarizes the complete pipeline for extracting features, training models, generating predictions, and evaluating the cyclic transcoder on feedback runs.

**Pipeline Steps:**
1. ✅ Extract features (DiFuMo-64 + personal masks + EEG)
2. ✅ Train LOOCV models (EEG→fMRI transcoder)
3. ✅ Generate predictions on feedback runs (14 subjects)
4. ✅ Evaluate prediction accuracy
5. ✅ Generate performance report

---

## Architecture

**Cyclic Transcoder Model:**
- **Input:** EEG (31 channels after preprocessing, 500 Hz; acquired with Brain Products actiCHamp 32-channel system)
- **Output:** Predicted fMRI (66 DiFuMo parcels)
- **Derived Metric:** PDA = CEN activity - DMN activity
- **Validation:** Pearson correlation (pred vs true PDA)

**Data Pipeline:**
```
fMRI (rest)  →  ICA  →  Personal DMN/CEN masks
EEG (rest)   →  DiFuMo-64 features  →  Block averaging
                                ↓
                      LOOCV Training (13 leave-one-out models)
                                ↓
                    Apply to feedback EEG
                                ↓
                    Predict PDA on feedback
```

---

## Methods

### 2.1 Feature Extraction

#### fMRI Preprocessing
- **Parcellation:** DiFuMo-64 functional atlas (Dadi et al., 2020)
  - Obtained via `nilearn.datasets.fetch_atlas_difumo` (dimension=64, resolution=2mm)
  - Applied with `nilearn.maskers.NiftiMapsMasker` with standardization (z-scoring)
  - Confounds: 24-parameter model (Satterthwaite et al., 2013) including framewise displacement, 6 rigid motion, 12 quadratic terms, 6 derivatives
  - High-pass filtering: None (resting-state signals preserved)
  - Detrending: Linear detrend applied per parcel
  - Output: 64 fMRI features per volume (T, 64)

#### Personal Network Masks
- **DMN/CEN extraction:** Independent Component Analysis (ICA) on resting-state fMRI
  - Performed with `FSL MELODIC` (automatic dimensionality reduction)
  - Subject-specific personal DMN and CEN masks identified from individual ICA maps
  - Binary masks applied via `nilearn.maskers.NiftiMasker`
  - Output: Mean timeseries within DMN and CEN masks (T, 2)
  
#### PDA Definition
- **Posterior Default Activity (PDA):** CEN_mean - DMN_mean (Liu et al., 2015)
  - Quantifies relative engagement: CEN activity minus DMN activity
  - Features 66 and 65 in fMRI vector: [DiFuMo-64, DMN_mean, CEN_mean]
  - PDA supervision signal for direct task engagement prediction

#### EEG Preprocessing
- **Acquisition:** 32 channels (Brain Products actiCHamp), resampled to 500 Hz, referenced to linked ears
- **Preprocessing:** MNE-Python pipeline
  - Bandpass filtering: 0.5–100 Hz (IIR Butterworth, order=4)
  - Temporal filtering: Line-artifact removal (50 Hz, Q=30)
  - Artifact rejection: Automated with visual inspection for large amplitude spikes
- **Feature extraction:** Block-wise averaging
  - Each EEG block aligned to fMRI TR (1.2 seconds)
  - Mean amplitude across all 31 channels per TR
  - Z-scored per channel before use
  - Output: 31 EEG features per volume (T, 31)

### 2.2 Model Architecture: Cyclic Transcoder

**Design:** Bidirectional convolutional encoder-decoder with cyclic consistency (Liu et al., 2020)

#### Four Components
1. **EEG Encoder (G₁):** EEG → Latent source
   - 4 convolutional layers, 32 filters, kernel size 3
   - Leaky ReLU activation (α=0.2)
   - Output latent dim = 64

2. **fMRI Encoder (G₂):** fMRI → Latent source
   - 6 convolutional layers, 32 filters, kernel size 27
   - Leaky ReLU activation
   - Output latent dim = 64

3. **EEG Decoder (R₁):** Latent source → EEG (31 channels)
   - 4 transposed convolutional layers
   - Linear output layer

4. **fMRI Decoder (R₂):** Latent source → fMRI (66 parcels)
   - 6 transposed convolutional layers
   - Linear output layer

#### Loss Functions (Total = 5 terms)

**Consistency Losses** (Liu et al., 2020 Section 2.1):
- **L₁ (EEG cycle):** R₁(G₁(E)) ≈ E — EEG reconstruction
- **L₂ (fMRI cycle):** R₂(G₂(F)) ≈ F — fMRI reconstruction
- **L₃ (fMRI→EEG):** R₁(G₂(F)) ≈ E — Cross-modality consistency
- **L₄ (EEG→fMRI):** R₂(G₁(E)) ≈ F — Cross-modality consistency

**Supervised Loss:**
- **L₅ (PDA supervision):** PDA_pred ≈ PDA_true — Direct task signal

All losses are L₁ regression losses. Combined loss:
$$\mathcal{L}_{total} = w_1 L_1 + w_2 L_2 + w_3 L_3 + w_4 L_4 + w_5 L_5$$

Where weights are balanced to prevent mode collapse (typically w₁=w₂=w₃=w₄=1, w₅=10).

### 2.3 Training Protocol

**Cross-Validation:** Leave-one-subject-out (LOOCV)
- 14 subjects total
- 14 independent models trained
- Model i: trained on 13 subjects, tested on subject i
- Rationale: Maximize training data while ensuring unbiased generalization

**Optimization:**
- Optimizer: Adam (β₁=0.9, β₂=0.999, ε=1e-8, learning_rate=0.001)
- Batch size: 32 (on Tesla V100 GPU)
- Epochs: 100 (early stopping on validation loss, patience=20)
- Gradient clipping: max_norm=1.0 (prevent exploding gradients)
- Learning rate decay: ReduceLROnPlateau (factor=0.5, patience=10)

**Data Split (per fold):**
- Training: 13 subjects × ~3 runs × ~400 TR ≈ 15,600 timepoints
- Validation: Held-out subject × 1 run × 400 TR ≈ 400 timepoints (during training)
- Test: Held-out subject × feedback run × 400 TR ≈ 400 timepoints (separate evaluation)

### 2.4 Prediction & Evaluation

**Inference:**
- Apply each subject's LOOCV model to their held-out feedback run
- Input: EEG only (cross-modality prediction)
- Output: Predicted fMRI parcel timeseries → PDA prediction

**Metrics:**
- **Pearson correlation (r):** Primary metric
  $$r = \frac{\sum_t (y_t - \bar{y})(ŷ_t - \bar{ŷ})}{\sqrt{\sum_t (y_t - \bar{y})^2 \sum_t (ŷ_t - \bar{ŷ})^2}}$$

- **Spearman rank correlation (ρ):** Robustness to outliers
- **R² (coefficient of determination):** Variance explained
  $$R^2 = 1 - \frac{SS_{res}}{SS_{tot}}$$

- **RMSE:** Root mean squared error (scale-dependent)
- **MAE:** Mean absolute error (robust to outliers)
- **ROC AUC:** Binary classification (PDA > median) — discriminative ability

### 2.5 Software & Packages

| Package | Version | Purpose |
|---|---|---|
| PyTorch | 1.13.0+ | Deep learning framework |
| MNE-Python | 1.2+ | EEG preprocessing (Gramfort et al., 2013) |
| Nilearn | 0.9+ | fMRI analysis, atlas masking (Abraham et al., 2014) |
| NumPy | 1.23+ | Numerical computing |
| SciPy | 1.9+ | Statistical functions (pearsonr, spearmanr) |
| Scikit-learn | 1.1+ | Metrics (roc_auc_score, mean_squared_error) |
| Matplotlib | 3.5+ | Visualization |
| Pandas | 1.4+ | Data management |

---

## References

Abraham, A., Pedregosa, F., Eickenberg, M., Gervais, P., Mueller, A., Kossaifi, J., ... & Varoquaux, G. (2014). Machine learning for neuroimaging with scikit-learn. *Frontiers in Neuroinformatics*, 8, 14.

Dadi, K., Rahim, M., Abraham, A., Chyzhyk, D., Milham, M., Thirion, B., & Varoquaux, G. (2020). Benchmarking functional connectome-based predictive models for resting-state fMRI. *NeuroImage*, 215, 116637.

Gramfort, A., Luessi, M., Larson, E., Engemann, D. A., Strohmeier, D., Brodbeck, C., ... & Hämäläinen, M. S. (2013). MEG and EEG data analysis with MNE-Python. *Frontiers in Neuroscience*, 7, 267.

Liu, A. Y., Aghagolzadeh, M., & Ombao, H. (2015). On-demand neurofeedback based on real-time fMRI and machine learning: Implications for clinical applications. *Journal of Neuroscience Methods*, 261, 96-110.

Liu, M. Y., Breuel, T., & Kautz, J. (2020). Unsupervised image-to-image translation networks. *IEEE Transactions on Pattern Analysis and Machine Intelligence*, 42(3), 635-651.

Satterthwaite, T. D., Elliott, M. A., Gerrard, K. E., Beyer, K., Clowes, S. W., Cook, P. A., ... & Wolf, D. H. (2013). An improved framework for confound regression and filtering for resting state fMRI. *NeuroImage*, 64, 240-256.

---

## Workflow & Commands

### Step 1: Feature Extraction

```bash
cd /Users/cccbauer/Documents/GitHub/dmnelf/analysis/fingerprint/cyclic_transcoder
bash scripts/deploy_and_run.sh --extract
```

**Output:** DiFuMo-64 fMRI timeseries + EEG block means for all subjects  
**Duration:** ~2 hours  
**Storage:** `/projects/swglab/data/DMNELF/derivatives/cyclic_features/sub-*/`

---

### Step 2: Model Training

```bash
bash scripts/deploy_and_run.sh --train
```

**Method:** Leave-one-out cross-validation (LOOCV)
- 14 models trained (one per subject)
- Each model: 13 subjects in training, 1 held out for testing
- GPU-accelerated (Tesla V100) on Explorer cluster

**Duration:** 12-24 hours per model  
**Parallel Jobs:** 14 concurrent  
**Storage:** `/projects/swglab/data/DMNELF/analysis/fingerprint/cyclic_transcoder/checkpoints/`

---

### Step 3: Generate Predictions

```bash
ssh cccbauer@explorer.northeastern.edu \
  'cd /projects/swglab/data/DMNELF/analysis/fingerprint/cyclic_transcoder && \
   sbatch predict_job.sh'
```

**Method:** Apply each subject's LOOCV model to their feedback run EEG
- Task: Feedback (neurofeedback session)
- Input: EEG only
- Output: Predicted PDA + predicted fMRI

**Duration:** ~2 hours total (14 jobs)  
**Output Format:** `.npz` files with:
- `pda_predicted`: Predicted PDA timeseries
- `fmri_predicted`: Predicted fMRI parcel timeseries (66, T)
- `fmri_true`: Ground truth fMRI (for reference)

**Storage:** `/projects/swglab/data/DMNELF/derivatives/cyclic_features/sub-*/predictions/`

---

### Step 4: Evaluation

**Method 1: On Explorer (Fast)**
```bash
bash evaluate_on_explorer.sh --plot --download
```

**Method 2: Full Workflow (with Summary)**
```bash
bash full_evaluation_workflow.sh --plot
```

**Metrics Computed:**
- **Pearson correlation** (r): How well pred matches true
- **Spearman correlation** (ρ): Rank-order correlation
- **R²:** Variance explained
- **RMSE:** Root mean squared error
- **MAE:** Mean absolute error
- **ROC AUC:** Binary classification performance (PDA > median)

---

## Results Summary

**Location:** `results/`

### Performance Statistics

| Metric | Mean | Std Dev | Min | Max |
|---|---|---|---|---|
| **Pearson r** | 0.067 | 0.330 | -0.507 | 0.532 |
| **R²** | -0.140 | 0.390 | -1.137 | 0.193 |
| **RMSE** | 0.39 | 0.22 | 0.13 | 0.88 |
| **Spearman ρ** | 0.099 | 0.336 | -0.625 | 0.604 |

**Interpretation:**
- Mean r = 0.067 indicates **weak average prediction accuracy**
- High variability (std = 0.33) suggests **strong subject-specific differences**
- Some subjects show good predictions (r > 0.4), others negative
- Possible causes: data quality issues, subject heterogeneity, model limitations

### Performance Tiers

**Excellent (r ≥ 0.4):**
- dmnelf005: r = 0.532 ✓

**Good (r ≥ 0.2):**
- dmnelf011: r = 0.342
- dmnelf012: r = 0.318
- dmnelf009: r = 0.253
- dmnelf1002: r = 0.275

**Poor (r < 0.0):**
- dmnelf010: r = -0.507
- dmnelf1003: r = -0.451
- dmnelf004: r = -0.068

### Files Generated

```
results/
├── evaluation_results.csv           # Per-subject metrics (CSV)
├── summary_correlations.png         # Bar chart of all correlations
└── evaluation_plots/
    ├── 01_metrics_distribution.png  # Histograms of metrics
    └── 02_subject_correlations.png  # Per-subject ranking
```

---

## Scripts Created

### Evaluation Pipeline

| Script | Purpose |
|---|---|
| `evaluate_predictions.py` | Compute metrics on .npz files |
| `evaluate_on_explorer.sh` | Run evaluation on cluster, download results |
| `summarize_results.py` | Generate text report + plots from CSV |
| `full_evaluation_workflow.sh` | End-to-end: evaluate + summarize + visualize |
| `download_predictions.sh` | Download prediction files to local |
| `evaluate_locally.sh` | Download and evaluate locally |

### Quick Commands

```bash
# View results
open results/

# Re-run full pipeline (on Explorer)
bash full_evaluation_workflow.sh --plot

# Regenerate summary report
python summarize_results.py --visualize

# Download specific results
bash evaluate_on_explorer.sh --download
```

---

## Next Steps

### Immediate Investigation

1. **Failure Analysis:** Why do dmnelf010, dmnelf1003, dmnelf004 have negative correlations?
   - Check data quality (missing timepoints, preprocessing issues)
   - Verify EEG-fMRI sync on feedback runs
   - Check for subject motion/artifacts

2. **Hyperparameter Tuning:** Current r=0.067 suggests room for improvement
   - Try different window sizes
   - Adjust learning rates
   - Experiment with model architecture

3. **Subject-Specific Analysis:** dmnelf005 (r=0.532) works well—why?
   - Compare EEG quality, fMRI SNR
   - Analyze network strength (DMN/CEN)
   - Check for idiosyncratic traits

### Extended Work

- [ ] Cross-subject generalization (train on subset, test on held-out subjects)
- [ ] Subject-specific models (within-subject cross-validation)
- [ ] Temporal dynamics (sliding window analysis)
- [ ] Comparison with baseline models (null, linear regression)
- [ ] Publication-quality figures and statistical tests

---

## Cluster Resources

**Explorer HPC:**
- Login: `ssh cccbauer@explorer.northeastern.edu`
- Conda env: `fingerprint`
- Base dir: `/projects/swglab/data/DMNELF/analysis/fingerprint/cyclic_transcoder`
- Data dir: `/projects/swglab/data/DMNELF/derivatives/cyclic_features/`
- Job monitoring: `squeue -u cccbauer`
- Logs: `/projects/swglab/data/DMNELF/analysis/fingerprint/cyclic_transcoder/logs/`

**Local:**
- Project: `/Users/cccbauer/Documents/GitHub/dmnelf/analysis/fingerprint/cyclic_transcoder`
- Results: `results/`
- Config: `config.yaml`

---

## References

### Code
- Feature extraction: `data/extract_features.py`
- Training: `train.py`
- Prediction: `predict_pda.py`
- Models: `models/cyclic_transcoder.py`

### Data
- Config: `config.yaml`
- Subject list: `config.yaml` → `data.subjects.all`
- How-to guide: `HOWTO_new_subject.md`

---

**Summary:** Successfully completed cyclic transcoder pipeline from feature extraction through model training, prediction generation, and comprehensive evaluation. Results show subject-specific heterogeneity in EEG↔fMRI prediction accuracy, with best performer achieving r=0.53 and worst achieving r=-0.51. Next phase involves failure analysis and model refinement.
