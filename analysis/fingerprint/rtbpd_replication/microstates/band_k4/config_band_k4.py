# config_band_k4.py — rtBPD classic-taxonomy (k=4) band-resolved reanalysis
#
# Follow-up to the k=7 DMN/CEN/... pipeline in the parent microstates/
# directory, motivated by Meynaghizadeh Zargar et al. (2025, Frontiers in
# Psychiatry), whose microstate pre/post intervention effect showed up
# specifically in the classic 4-microstate solution (A/B/C/D, Koenig et
# al. 2002) and in theta-band / full-spectrum analyses. Scoped to exactly
# those two conditions (fullspectrum, theta) per explicit user decision —
# not the paper's full 5-band decomposition.
#
# IMPORTANT: this is a READ-ONLY reuse of the Phase 1 preprocessed rest
# FIFs (EEG_ROOT below) — no new cluster preprocessing. This file is
# intentionally self-contained (does not import from the parent
# config_rtbpd_ms.py) to keep the two pipelines fully decoupled per the
# "don't touch the existing k=7 pipeline" instruction; the subject list /
# session / run / channel-exclusion constants below are kept identical to
# config_rtbpd_ms.py by hand — if those ever change there, mirror the
# change here too.
from pathlib import Path

# ── SSH / Cluster (identical to config_rtbpd_ms.py) ───────────
CLUSTER_USER  = "cccbauer"
CLUSTER_HOST  = "explorer.northeastern.edu"
CLUSTER_SSH   = CLUSTER_USER + "@" + CLUSTER_HOST
SLURM_ACCOUNT = "suewhit"
PYTHON        = "/home/cccbauer/.conda/envs/eeg_preproc/bin/python"

# ── Cluster paths ────────────────────────────────────────────
# Own subtree under the existing microstates cluster base -- fully
# separate from the k=7 pipeline's {scripts,microstates,results,logs}.
CLUSTER_BASE = "/projects/swglab/data/rtBPD/analysis/fingerprint/rtbpd_replication/microstates/band_k4"
EEG_ROOT     = "/projects/swglab/data/rtBPD/derivatives/eeg_preprocessed"

# ── Local paths (repo-local; no Dropbox mirror needed) ────────
LOCAL_BASE  = Path(__file__).parent
SCRIPTS_DIR = LOCAL_BASE / "scripts"
LOGS_DIR    = LOCAL_BASE / "logs"
RESULTS_DIR = LOCAL_BASE / "results"

# ── Subjects / session / runs (identical to config_rtbpd_ms.py) ──
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
EEG_DESC  = "preproc500Hz"   # Phase 1 output filenames (broadband 1-40Hz preprocessed)

# ── Band conditions (scoped: exactly these two, not the full 5-band set) ──
BANDS = ["fullspectrum", "theta"]
THETA_BAND = (4.0, 8.0)   # Hz, applied as a zero-phase FIR bandpass on top
                          # of the already-preprocessed broadband signal

# ── Microstate — classic k=4 (Koenig et al. 2002 / Michel & Koenig 2018) ──
# k=4 (NOT 7) — this reanalysis specifically tests the classic A/B/C/D
# taxonomy, unlike the parent pipeline's Custo-2017 k=7 network taxonomy.
N_MICROSTATES      = 4
N_KMEANS_RESTARTS  = 20     # unchanged from config_rtbpd_ms.py
KMEANS_MAX_ITER    = 1000   # unchanged from config_rtbpd_ms.py
GFP_OUTLIER_SD     = 3.0    # unchanged from config_rtbpd_ms.py
POLARITY_INVARIANT = True

# ── Channel exclusion (identical to config_rtbpd_ms.py) ───────
# TP9/TP10 excluded by name; ECG/EKG/EMG/EOG/STIM/STATUS dropped via
# substring match in load_eeg() (see 01_fit_microstates_band_k4.py) --
# same approach as the k=7 pipeline, verified against the real rtBPD
# montage in the original Phase 0 preflight.
EXCLUDE_CHS = ["TP9", "TP10"]
