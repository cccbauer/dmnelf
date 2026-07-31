# config_connectivity.py — rtBPD nf1 rest PLV connectivity pre-vs-post
#
# Simple companion analysis to band_power_prepost/ (no clustering): PLV
# (phase-locking value) between a curated set of DMN-relevant channel
# pairs, averaged across pairs into one scalar per band per run -- same
# structure as band_power_prepost so it's directly comparable/mergeable.
# READ-ONLY reuse of the Phase 1 preprocessed rest FIFs -- no new cluster
# preprocessing, and this does not touch the microstate or band-power
# pipelines.
#
# PLV computation ported (whole-run, no TR-blocking/HRF -- irrelevant for
# a pure rest EEG-EEG measure) from
# wavelet_coupling/scripts/connectivity_features.py's compute_plv_run()
# and get_dmn_relevant_pairs() (DMNELF feedback-task code, never run on
# rtBPD nor used for a plain EEG-EEG connectivity result before this).
from pathlib import Path

# ── SSH / Cluster (identical to the other rtBPD pipelines) ─────
CLUSTER_USER  = "cccbauer"
CLUSTER_HOST  = "explorer.northeastern.edu"
CLUSTER_SSH   = CLUSTER_USER + "@" + CLUSTER_HOST
SLURM_ACCOUNT = "suewhit"
PYTHON        = "/home/cccbauer/.conda/envs/eeg_preproc/bin/python"

# ── Cluster paths ────────────────────────────────────────────
CLUSTER_BASE = "/projects/swglab/data/rtBPD/analysis/fingerprint/rtbpd_replication/connectivity_prepost"
EEG_ROOT     = "/projects/swglab/data/rtBPD/derivatives/eeg_preprocessed"

# ── Local paths (repo-local; no Dropbox mirror needed) ────────
LOCAL_BASE  = Path(__file__).parent
SCRIPTS_DIR = LOCAL_BASE / "scripts"
LOGS_DIR    = LOCAL_BASE / "logs"
RESULTS_DIR = LOCAL_BASE / "results"

# ── Subjects / session / runs (identical to the other pipelines) ──
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
# substring match (identical approach to the other pipelines).
EXCLUDE_CHS = ["TP9", "TP10"]

# ── Bands (identical to band_power_prepost / rtbpd_replication/config.yaml) ──
BANDS = {
    "delta": (1.0, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta":  (13.0, 30.0),
    "gamma": (30.0, 40.0),
}
