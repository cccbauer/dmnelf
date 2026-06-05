# config.py  — DMNELF microstate PDA pipeline
from pathlib import Path

# ── SSH / Cluster ──────────────────────────────────────────
CLUSTER_USER   = "cccbauer"
CLUSTER_HOST   = "explorer.northeastern.edu"
CLUSTER_SSH    = CLUSTER_USER + "@" + CLUSTER_HOST
SLURM_ACCOUNT  = "suewhit"
PYTHON         = "/home/cccbauer/.conda/envs/eeg_preproc/bin/python"
# ── Cluster paths ──────────────────────────────────────────
CLUSTER_BASE   = "/projects/swglab/data/DMNELF/analysis/fmri_preprocessing"
EEG_ROOT       = "/projects/swglab/data/DMNELF/derivatives/eeg_preprocessed"
DIFUMO_ROOT    = "/projects/swglab/data/DMNELF/analysis/MNE/jupyter/neurobolt/difumo_timeseries"
CONFOUND_ROOT  = "/projects/swglab/data/DMNELF/derivatives/fmriprep_25.2.5_fmap"
EEG_PREP_ROOT = Path("/projects/swglab/data/DMNELF/derivatives/eeg_preprocessed")
FMRIPREP_ROOT = "/projects/swglab/data/DMNELF/derivatives/fmriprep_25.2.5_fmap"
# ── Local paths (machine-agnostic) ─────────────────────────
# On cluster: use cluster paths only
# On local: use local paths if they exist
import os
import socket
_hostname = socket.gethostname()
_is_cluster = "explorer" in _hostname or "login" in _hostname

if _is_cluster:
    # On cluster - use cluster paths
    LOCAL_BASE = Path("/projects/swglab/data/DMNELF/analysis/fmri_preprocessing")
    SCRIPTS_DIR = LOCAL_BASE / "deploy_scripts"
else:
    # On local machine
    _user = os.environ.get("USER", "")
    if _user == "anitya":
        _dropbox = Path("/Users/anitya/MIT Dropbox/Clemen Bauer/00_2022/ResearchScientist"
                        "/01_grants/01_R21-2021/Analysis/MNE")
    elif _user == "cccbauer":
        _dropbox = Path("/Users/cccbauer/MIT Dropbox/Clemen Bauer/00_2022/ResearchScientist"
                        "/01_grants/01_R21-2021/Analysis/MNE")
    else:
        _dropbox = Path.home() / "microstate_pda_v3"

    LOCAL_BASE  = _dropbox / "microstate_pda_v3"
    SCRIPTS_DIR = LOCAL_BASE / "scripts"
FIGURES_DIR = LOCAL_BASE / "figures"
LOGS_DIR    = LOCAL_BASE / "logs"
MODELS_DIR  = LOCAL_BASE / "models"
RESULTS_DIR = LOCAL_BASE / "results"

# ── Subjects ───────────────────────────────────────────────
SUBJECTS = [
    "sub-dmnelf012",
    "sub-dmnelf013",
    "sub-dmnelf014",
    "sub-dmnelf015",
]

# Runs with no raw data (never acquired) — excluded from pipeline
MISSING_RUNS = {
    "sub-dmnelf1002": [("rest", "02")],
    "sub-dmnelf1003": [("feedback", "04")],
}

# ── Tasks / runs ───────────────────────────────────────────
RUN_MAP = {
    "rest":      ["01", "02"],
    "shortrest": ["01"],
    "feedback":  ["01", "02", "03", "04"],
}

# Subjects with simultaneous EEG + fMRI (complete acquisition)
SUBJECTS_EEG_FMRI = [
    "sub-dmnelf001", "sub-dmnelf004", "sub-dmnelf005", "sub-dmnelf006",
    "sub-dmnelf007", "sub-dmnelf008", "sub-dmnelf009", "sub-dmnelf010",
    "sub-dmnelf011", "sub-dmnelf1001",
]

# Subjects with EEG + fMRI but fewer runs (EEG and fMRI matched,
# missing runs never acquired — complete in themselves)
SUBJECTS_EEG_FMRI_PARTIAL = [
    "sub-dmnelf1002",  # missing rest-02 EEG+fMRI
    "sub-dmnelf1003",  # missing feedback-04 EEG+fMRI
]

# Subjects with fMRI only (no EEG acquired — R128 marker absent)
SUBJECTS_FMRI_ONLY = [
    "sub-dmnelf002", "sub-dmnelf003", "sub-dmnelf999",
]

# All EEG+fMRI subjects (complete + partial)
SUBJECTS_EEG_FMRI_ALL = SUBJECTS_EEG_FMRI + SUBJECTS_EEG_FMRI_PARTIAL

# ── EEG ────────────────────────────────────────────────────
SFREQ          = 250        # Hz after resampling
N_CHANNELS     = 31
TR_SAMPLES     = 240        # EEG samples per TR at 200 Hz
ALIGNMENT_TOL  = 480        # max tolerated misalignment (~2 TR)

# ── Microstate — Custo 2017 ────────────────────────────────
N_MICROSTATES       = 7     # NOT 4 — validated against Tarailis 2023
N_KMEANS_RESTARTS   = 20
KMEANS_MAX_ITER     = 1000
GFP_OUTLIER_SD      = 3.0   # reject GFP peaks above this for fitting
POLARITY_INVARIANT  = True  # standard for microstate analysis

# ── fMRI / DiFuMo-64 ───────────────────────────────────────
N_PARCELS      = 64
# DiFuMo-64 parcel indices (0-based) for DMN and CEN
# Verified against labels_64_dictionary.csv (Dadi et al. 2020)
# DMN: PCC(3), STS/angular(6), sup rostral gyrus(29), angular sup(35),
#      dmPFC(38), angular inf(58), MTG(60), sup frontal gyrus(61)
# Note: ACC (idx 22) removed — Yeo-17 assigns it to SalVentAttnB
DMN_IDX = [3, 6, 29, 35, 38, 58, 60, 61]

# CEN: parieto-occipital(4), IPS(31), IPS-RH(47), IFsulcus(48),
#      MFG(50), IFG(51)
CEN_IDX = [4, 31, 47, 48, 50, 51]

# ── MURFI-aligned PDA ──────────────────────────────────────
BASELINE_VOLS  = 25         # first 25 vols = 30s baseline per run
PSA_WINDOW     = 25         # rolling window for PSA multiplier

# ── HRF convolution ────────────────────────────────────────
TR             = 1.2        # seconds
HRF_PEAK_S     = 5.0        # canonical HRF peak delay (seconds)

# ── Decoder ────────────────────────────────────────────────
# Features: 7 T-hat (TESS) + GFP + GMD = 9 per TR
N_FEATURES     = 9
DECODER_MODEL  = "elasticnet"
ALPHAS         = [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
CV_STRATEGY    = "leave_one_run_out"
TRAIN_TASK     = "feedback"
TEST_TASKS     = ["shortrest", "feedback"]
Z_SCORE_FEATS  = True
Z_SCORE_TARGET = True

# EEG filename suffix for minimally preprocessed files
EEG_RAW_SUFFIX = "_desc-bvaAC1kHz_eeg.edf"