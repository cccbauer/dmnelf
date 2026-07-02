#!/usr/bin/env python3
"""
fit_hmm.py
----------
Fit a Time-Delay Embedded HMM (TIDE-HMM) on resting-state EEG, replicating
Cooray et al. 2024. Real-time variant: causal embedding (current + previous
timepoints only), K=12 states.

Usage:
  python fit_hmm.py --subjects dmnelf008 dmnelf012   # pilot
  python fit_hmm.py --group                           # all subjects pooled
"""
import argparse, warnings
from pathlib import Path
import numpy as np, yaml, mne

warnings.filterwarnings("ignore")
mne.set_log_level("ERROR")

SCRIPT_DIR = Path(__file__).resolve().parent
PROJ_DIR = SCRIPT_DIR.parent
CONFIG_PATH = PROJ_DIR / "config.yaml"


def load_config(p):
    cfg = yaml.safe_load(open(p))
    d = cfg["data"]
    suffix = "_cluster" if Path("/projects/swglab").exists() else "_local"
    for key in ("features_dir", "eeg_preproc_dir", "confounds_dir"):
        d[key] = str(Path(d[key + suffix]).expanduser())
    return cfg


def load_rest_eeg(cfg, sub):
    """Load and preprocess resting-state EEG for HMM: 1-45Hz bandpass, 250Hz."""
    d = cfg["data"]
    ses = d["session"]
    eroot = Path(d["eeg_preproc_dir"])
    eeg_cfg = cfg["data"]["eeg"]
    sfreq_target = eeg_cfg["sfreq_hmm"]
    lo, hi = eeg_cfg["bandpass"]
    desc = eeg_cfg["desc"]

    runs_data = []
    for run in [1, 2]:
        fif = (eroot / f"sub-{sub}" / ses / "eeg" /
               f"sub-{sub}_{ses}_task-rest_run-{run:02d}_desc-{desc}_eeg.fif")
        if not fif.exists():
            continue
        raw = mne.io.read_raw_fif(str(fif), preload=True, verbose=False)
        picks = mne.pick_types(raw.info, eeg=True, exclude=[])
        raw.pick(picks)
        raw.filter(lo, hi, verbose=False)
        raw.resample(sfreq_target, verbose=False)
        data = raw.get_data().T  # (n_samples, n_channels) — osl_dynamics wants time x channels
        runs_data.append(data)

    return runs_data


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(CONFIG_PATH))
    ap.add_argument("--subjects", nargs="+", default=None)
    ap.add_argument("--group", action="store_true", help="Use all subjects from config")
    ap.add_argument("--n_states", type=int, default=None)
    ap.add_argument("--n_epochs", type=int, default=None)
    ap.add_argument("--out_name", default="hmm_model")
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.group:
        subjects = cfg["data"]["subjects"]["all"]
    elif args.subjects:
        subjects = args.subjects
    else:
        subjects = cfg["data"]["subjects"]["pilot"]

    n_states = args.n_states or cfg["hmm"]["n_states"]
    n_epochs = args.n_epochs or cfg["hmm"]["n_epochs"]

    print(f"Fitting TIDE-HMM")
    print(f"  Subjects: {subjects}")
    print(f"  States: {n_states}")
    print()

    # ── Load and concatenate resting EEG across subjects ──
    print("Loading resting-state EEG...")
    all_runs = []
    subject_run_labels = []
    for sub in subjects:
        runs = load_rest_eeg(cfg, sub)
        for ri, run_data in enumerate(runs):
            all_runs.append(run_data)
            subject_run_labels.append((sub, ri + 1))
            print(f"  {sub} run {ri+1}: {run_data.shape}")

    if not all_runs:
        print("ERROR: no data loaded")
        return

    # ── Set up osl_dynamics Data object ──
    from osl_dynamics.data import Data
    from osl_dynamics.models.hmm import Config, Model

    out_dir = PROJ_DIR / "results" / args.out_name
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save raw arrays for osl_dynamics to load (it works off .npy files or arrays directly)
    data_obj = Data(all_runs, sampling_frequency=cfg["data"]["eeg"]["sfreq_hmm"])

    # Time-delay embedding + PCA (causal: real-time variant uses only past samples)
    n_channels = all_runs[0].shape[1]
    embedding_lag = cfg["hmm"]["embedding_lag"]
    # n_pca_components: paper targets 90% variance; with n_channels=31 and
    # n_embeddings=7 (window W=l+1=8), embedded dim = n_channels * W ≈ 248.
    # Use a fixed component count as a starting point (refined after inspecting
    # explained variance in practice).
    n_pca_components = cfg["hmm"].get("n_pca_components", min(80, n_channels * (embedding_lag + 1)))

    print("\nPreparing data (time-delay embed + PCA)...")
    data_obj.tde_pca(
        n_embeddings=embedding_lag,
        n_pca_components=n_pca_components,
        whiten=True,
    )
    data_obj.standardize()

    n_features = data_obj.n_channels
    print(f"  Embedded feature dim: {n_features}")

    # ── Configure and fit HMM ──
    hmm_config = Config(
        n_states=n_states,
        n_channels=n_features,
        sequence_length=cfg["hmm"]["sequence_length"],
        learn_means=False,
        learn_covariances=True,
        batch_size=16,
        learning_rate=cfg["hmm"]["learning_rate"],
        n_epochs=n_epochs,
    )

    model = Model(hmm_config)
    model.summary()

    print("\nTraining HMM...")
    n_init = 10 if len(subjects) > 3 else 3
    # random_subset_initialization is more robust than
    # random_state_time_course_initialization for K=12 on group data
    # (the latter fails when a simulated init sequence doesn't visit all states).
    init_history = model.random_subset_initialization(
        data_obj, n_init=n_init, n_epochs=2, take=0.25
    )
    history = model.fit(data_obj)

    # ── Get state probabilities/time courses ──
    alpha = model.get_alpha(data_obj)  # list of (n_samples, n_states) per run

    # Save model and results
    model.save(str(out_dir / "trained_model"))
    np.savez_compressed(
        out_dir / "state_probabilities.npz",
        alpha=np.array(alpha, dtype=object),
        subject_run_labels=subject_run_labels,
        n_states=n_states,
    )

    print(f"\nSaved model to {out_dir / 'trained_model'}")
    print(f"Saved state probabilities to {out_dir / 'state_probabilities.npz'}")

    # Print state occupancy summary.
    # argmax_time_courses returns ONE-HOT time courses (n_samples, n_states),
    # so fractional occupancy is just the mean of the one-hot column per state.
    from osl_dynamics.inference import modes
    stc = modes.argmax_time_courses(alpha)
    all_stc = np.concatenate([np.asarray(s) for s in stc], axis=0)
    occ = all_stc.mean(axis=0)
    for k in range(n_states):
        print(f"  State {k+1}: fractional occupancy = {occ[k]:.3f}")


if __name__ == "__main__":
    main()
