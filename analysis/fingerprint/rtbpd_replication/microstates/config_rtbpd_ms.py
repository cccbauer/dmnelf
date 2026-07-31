# config_rtbpd_ms.py — rtBPD microstate pipeline (rest pre vs post nf1)
#
# Sibling pipeline to microstate_pda/config.py, but for the rtBPD nf1
# neurofeedback cohort. Contrasts pre-nf1 (runs 01/02) vs post-nf1
# (runs 03/04) resting-state EEG microstate temporal parameters, rather
# than decoding a PDA target — this pipeline has no fMRI/PDA component.
from pathlib import Path

# ── SSH / Cluster ────────────────────────────────────────────
CLUSTER_USER  = "cccbauer"
CLUSTER_HOST  = "explorer.northeastern.edu"
CLUSTER_SSH   = CLUSTER_USER + "@" + CLUSTER_HOST
SLURM_ACCOUNT = "suewhit"
PYTHON        = "/home/cccbauer/.conda/envs/eeg_preproc/bin/python"

# ── Cluster paths ────────────────────────────────────────────
CLUSTER_BASE = "/projects/swglab/data/rtBPD/analysis/fingerprint/rtbpd_replication/microstates"
EEG_ROOT     = "/projects/swglab/data/rtBPD/derivatives/eeg_preprocessed"

# ── Local paths (repo-local; no Dropbox mirror needed) ────────
LOCAL_BASE  = Path(__file__).parent
SCRIPTS_DIR = LOCAL_BASE / "scripts"
LOGS_DIR    = LOCAL_BASE / "logs"
RESULTS_DIR = LOCAL_BASE / "results"

# ── Subjects ─────────────────────────────────────────────────
# The 15 rtBPD nf1 subjects confirmed (Phase 0 preflight, 2026-07-15 ssh ls
# check) to have a COMPLETE 4-run task-rest EDF set. Excluded per explicit
# user decision (no nf2 fallback, no partial-run substitution): rtbpd004
# (only runs 01/02), rtbpd022 (missing run-03), rtbpd026 (missing run-01),
# rtbpd027 (no nf1 rest at all — full set exists only under ses-nf2),
# rtbpd028 (same as rtbpd027), rtbpd034 (only runs 01/02).
SUBJECTS = [
    "sub-rtbpd002", "sub-rtbpd003",
    "sub-rtbpd009", "sub-rtbpd010", "sub-rtbpd011", "sub-rtbpd012",
    "sub-rtbpd013", "sub-rtbpd015", "sub-rtbpd018", "sub-rtbpd020",
    "sub-rtbpd021", "sub-rtbpd024", "sub-rtbpd030", "sub-rtbpd038",
    "sub-rtbpd040",
]

# ── Session / runs ───────────────────────────────────────────
# Phase 1 preprocessing writes ALL 15 subjects' rest derivatives to ses-nf
# (pilots rtbpd002/003 raw+out session = ses-nf; the other 13 subjects raw
# session = ses-nf1, out session = ses-nf), unifying the session label the
# same way task-feedback derivatives already are.
SESSION   = "ses-nf"
PRE_RUNS  = ["01", "02"]
POST_RUNS = ["03", "04"]
ALL_RUNS  = PRE_RUNS + POST_RUNS

# ── EEG ──────────────────────────────────────────────────────
SFREQ    = 500
SFREQ_TAG = str(SFREQ) + "Hz"
EEG_DESC = "preproc500Hz"   # matches Phase 1 output filenames:
                            # sub-rtbpdNNN_ses-nf_task-rest_run-NN_desc-preproc500Hz_eeg.fif

# ── Microstate — Custo 2017 (identical constants to microstate_pda/config.py) ──
N_MICROSTATES      = 7      # NOT 4 — validated against Tarailis 2023
N_KMEANS_RESTARTS  = 20
KMEANS_MAX_ITER    = 1000
GFP_OUTLIER_SD      = 3.0   # reject GFP peaks above this for fitting
POLARITY_INVARIANT  = True  # standard for microstate analysis

# ── Channel exclusion ────────────────────────────────────────
# TP9/TP10 are EEG-fMRI-artifact-prone (confirmed present in the rtBPD
# montage in Phase 0), excluded exactly as in the DMNELF pipeline.
#
# NOTE (Phase 0 finding, verified against the real DMNELF code path before
# writing this): the rtBPD preprocessed FIFs also carry a literal "ECG"
# channel that DMNELF's montage does not expose under this name. DMNELF's
# 01_fit_microstates.py never lists "ECG" in EXCLUDE_CHS either — it drops
# ECG/EKG/EMG/EOG/STIM/STATUS via a separate substring match inside
# load_eeg():
#     drop = [ch for ch in raw.ch_names
#             if any(x in ch.upper() for x in
#                    ("ECG","EKG","EMG","EOG","STIM","STATUS"))
#             or ch in EXCLUDE_CHS]
# 01_fit_microstates_rtbpd.py mirrors this exact drop-by-substring logic, so
# EXCLUDE_CHS here stays TP9/TP10 only — ECG is still guaranteed to be
# dropped before k-means/back-fitting ever sees it (29 retained channels:
# 32 total - TP9 - TP10 - ECG).
EXCLUDE_CHS = ["TP9", "TP10"]
