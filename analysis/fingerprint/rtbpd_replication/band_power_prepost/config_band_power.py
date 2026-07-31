# config_band_power.py — rtBPD nf1 rest band-power pre-vs-post
#
# Simple companion analysis to the microstate pipelines (no clustering):
# Welch PSD -> 5 standard band powers, averaged across channels, pre vs
# post nf1. READ-ONLY reuse of the Phase 1 preprocessed rest FIFs -- no
# new cluster preprocessing, and this does not touch the microstate
# pipelines in ../microstates/ or ../microstates/band_k4/.
from pathlib import Path

# ── SSH / Cluster (identical to the microstate pipelines) ─────
CLUSTER_USER  = "cccbauer"
CLUSTER_HOST  = "explorer.northeastern.edu"
CLUSTER_SSH   = CLUSTER_USER + "@" + CLUSTER_HOST
SLURM_ACCOUNT = "suewhit"
PYTHON        = "/home/cccbauer/.conda/envs/eeg_preproc/bin/python"

# ── Cluster paths ────────────────────────────────────────────
CLUSTER_BASE = "/projects/swglab/data/rtBPD/analysis/fingerprint/rtbpd_replication/band_power_prepost"
EEG_ROOT     = "/projects/swglab/data/rtBPD/derivatives/eeg_preprocessed"

# ── Local paths (repo-local; no Dropbox mirror needed) ────────
LOCAL_BASE  = Path(__file__).parent
SCRIPTS_DIR = LOCAL_BASE / "scripts"
LOGS_DIR    = LOCAL_BASE / "logs"
RESULTS_DIR = LOCAL_BASE / "results"

# ── Subjects / session / runs (identical to the microstate pipelines) ──
SUBJECTS = [
    "sub-rtbpd002", "sub-rtbpd003",
    "sub-rtbpd009", "sub-rtbpd010", "sub-rtbpd011", "sub-rtbpd012",
    "sub-rtbpd013", "sub-rtbpd015", "sub-rtbpd018", "sub-rtbpd020",
    "sub-rtbpd021", "sub-rtbpd024", "sub-rtbpd030", "sub-rtbpd038",
    "sub-rtbpd040",
]
SESSION   = "ses-nf"
PRE_RUNS  = ["01", "02"]
POST_RUNS = ["03", "04"]
ALL_RUNS  = PRE_RUNS + POST_RUNS

# ── EEG ──────────────────────────────────────────────────────
SFREQ     = 500
SFREQ_TAG = str(SFREQ) + "Hz"
EEG_DESC  = "preproc500Hz"

# TP9/TP10 excluded by name; ECG/EKG/EMG/EOG/STIM/STATUS dropped via
# substring match (identical approach to the microstate pipelines).
EXCLUDE_CHS = ["TP9", "TP10"]

# ── Welch PSD params ───────────────────────────────────────────
WELCH_WIN_SEC     = 4.0   # 4 s segments
WELCH_OVERLAP_PCT = 0.5   # 50% overlap

# ── Bands (identical to rtbpd_replication/config.yaml's `bands:`) ────
BANDS = {
    "delta": (1.0, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta":  (13.0, 30.0),
    "gamma": (30.0, 40.0),
}
