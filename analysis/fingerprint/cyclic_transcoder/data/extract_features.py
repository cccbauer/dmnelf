"""
extract_features.py
-------------------
Run once per subject before training.
Writes one .npz per (subject, task, run) to features_dir:
    fmri_features : (T, 66)   — DiFuMo-64 + [DMN_mean, CEN_mean], z-scored
    eeg_block     : (T, 31)   — EEG block-means at TR, z-scored per channel
    pda           : (T,)      — CEN_mean - DMN_mean
where T = number of volumes in that run.

Usage:
    python extract_features.py --subject dmnelf001 --config ../config.yaml
    python extract_features.py --all --config ../config.yaml      # via SLURM array
"""

import argparse
import os
import warnings
from pathlib import Path

import numpy as np
import yaml
import mne
from nilearn import image, maskers
from nilearn.datasets import fetch_atlas_difumo
from nilearn.signal import clean as nilearn_clean

warnings.filterwarnings("ignore", category=UserWarning)


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def subject_list(cfg):
    excluded = set(cfg["data"]["subjects"].get("exclude", []))
    return [s for s in cfg["data"]["subjects"]["all"] if s not in excluded]


# ---------------------------------------------------------------------------
# fMRI helpers
# ---------------------------------------------------------------------------

def get_bold_path(cfg, subject, task, run=None):
    """Return path to fMRIPrep preproc BOLD and matching confounds TSV."""
    fprep = Path(cfg["data"]["fmriprep_dir"])
    sub = f"sub-{subject}"
    space = cfg["data"]["fmri"]["space"]
    res = cfg["data"]["fmri"]["res"]
    session = cfg["data"]["session"]          # ses-dmnelf

    func_dir = fprep / sub / session / "func"
    if not func_dir.exists():
        return None, None, None

    run_tag = f"_run-{run:02d}" if isinstance(run, int) else (f"_run-{run}" if run else "")
    pattern = (
        f"{sub}_{session}_task-{task}{run_tag}"
        f"_space-{space}_res-{res}_desc-preproc_bold.nii.gz"
    )
    bold_path = func_dir / pattern
    conf_path = func_dir / bold_path.name.replace(
        f"_space-{space}_res-{res}_desc-preproc_bold.nii.gz",
        "_desc-confounds_timeseries.tsv",
    )
    if bold_path.exists():
        return bold_path, conf_path if conf_path.exists() else None, session
    return None, None, None


def load_confounds(conf_path, columns):
    """Load selected confound columns, fill NaNs with 0."""
    import pandas as pd
    df = pd.read_csv(conf_path, sep="\t")
    available = [c for c in columns if c in df.columns]
    missing = set(columns) - set(available)
    if missing:
        print(f"    [warn] confounds not found: {missing}")
    conf = df[available].fillna(0).values
    return conf


def extract_difumo_timeseries(bold_img, conf_mat, cfg):
    """
    Apply DiFuMo-64 masker, regress confounds, return (T, 64) z-scored array.
    """
    atlas = fetch_atlas_difumo(
        dimension=64,
        resolution_mm=2,
        data_dir=cfg["data"]["difumo_cache_dir"],
        legacy_format=False,
    )
    masker = maskers.NiftiMapsMasker(
        maps_img=atlas.maps,
        standardize=True,       # z-score per parcel
        detrend=True,
        high_pass=None,
        low_pass=None,
        t_r=cfg["data"]["fmri"]["tr"],
        memory_level=0,
        verbose=0,
    )
    ts = masker.fit_transform(bold_img, confounds=conf_mat)  # (T, 64)
    return ts.astype(np.float32)


def extract_personal_roi_timeseries(bold_img, subject, cfg):
    """
    Apply subject-specific DMN and CEN binary masks.
    Returns (T, 2) — columns: [DMN_mean, CEN_mean].

    Actual filenames:
        sub-{subject}_space-MNI152NLin6Asym_res-2_dmn_mask.nii.gz
        sub-{subject}_space-MNI152NLin6Asym_res-2_cen_mask.nii.gz
    """
    mask_dir = Path(cfg["data"]["personal_masks_dir"]) / f"sub-{subject}"
    m_cfg = cfg["data"]["masks"]
    space = m_cfg["space"]   # MNI152NLin6Asym
    res   = m_cfg["res"]     # 2
    prefix = f"sub-{subject}_space-{space}_res-{res}"

    dmn_path = mask_dir / f"{prefix}_{m_cfg['dmn_suffix']}"
    cen_path = mask_dir / f"{prefix}_{m_cfg['cen_suffix']}"

    results = []
    for mask_path, label in [(dmn_path, "DMN"), (cen_path, "CEN")]:
        if not mask_path.exists():
            raise FileNotFoundError(
                f"Personal {label} mask not found: {mask_path}"
            )
        masker = maskers.NiftiMasker(
            mask_img=str(mask_path),
            standardize=True,
            detrend=True,
            t_r=cfg["data"]["fmri"]["tr"],
            memory_level=0,
        )
        ts = masker.fit_transform(bold_img)  # (T, n_voxels_in_mask)
        results.append(ts.mean(axis=1))      # (T,)

    return np.stack(results, axis=1).astype(np.float32)  # (T, 2)


# ---------------------------------------------------------------------------
# EEG helpers
# ---------------------------------------------------------------------------

def find_eeg_path(cfg, subject, task, run=None):
    """Return path to preprocessed MNE .fif file (500Hz version)."""
    eeg_root = Path(cfg["data"]["eeg_preproc_dir"])
    ses = cfg["data"]["session"]                      # ses-dmnelf
    desc = cfg["data"]["eeg"].get("desc", "preproc500Hz")
    sub = f"sub-{subject}"
    run_tag = f"_run-{run:02d}" if isinstance(run, int) else (f"_run-{run}" if run else "")
    eeg_dir = eeg_root / sub / ses / "eeg"
    if not eeg_dir.exists():
        return None, None
    pattern = f"{sub}_{ses}_task-{task}{run_tag}_desc-{desc}_eeg.fif"
    p = eeg_dir / pattern
    if p.exists():
        return p, ses
    return None, None


def eeg_block_mean(raw, sfreq, samples_per_tr, n_volumes):
    """
    Block-average continuous EEG to TR grid.
    Returns (n_volumes, n_channels) float32, z-scored per channel.
    """
    data = raw.get_data()  # (n_channels, n_samples)
    n_ch = data.shape[0]

    # Crop to exactly n_volumes * samples_per_tr
    n_samples_needed = n_volumes * samples_per_tr
    if data.shape[1] < n_samples_needed:
        raise ValueError(
            f"EEG has {data.shape[1]} samples, need {n_samples_needed} "
            f"({n_volumes} vols × {samples_per_tr} samples/TR)"
        )
    data = data[:, :n_samples_needed]

    # Reshape and average: (n_ch, n_volumes, samples_per_tr) → (n_volumes, n_ch)
    data = data.reshape(n_ch, n_volumes, samples_per_tr)
    block = data.mean(axis=2).T  # (n_volumes, n_ch)

    # Z-score per channel
    mu = block.mean(axis=0, keepdims=True)
    sigma = block.std(axis=0, keepdims=True) + 1e-8
    block = (block - mu) / sigma

    return block.astype(np.float32)


# ---------------------------------------------------------------------------
# Main extraction
# ---------------------------------------------------------------------------

def extract_subject(subject, cfg):
    features_dir = Path(cfg["data"]["features_dir"])
    tasks = (
        cfg["data"]["tasks"]["train"]
        + cfg["data"]["tasks"]["validate"]
        + cfg["data"]["tasks"]["apply"]
    )
    tr = cfg["data"]["fmri"]["tr"]
    samples_per_tr = cfg["data"]["eeg"]["samples_per_tr"]
    confound_cols = cfg["data"]["fmri"]["confound_columns"]

    for task in tasks:
        # Try runs 1..5, then no run tag
        run_tags = list(range(1, 6)) + [None]
        found_any = False

        for run in run_tags:
            bold_path, conf_path, ses = get_bold_path(cfg, subject, task, run)
            if bold_path is None:
                continue

            eeg_path, _ = find_eeg_path(cfg, subject, task, run)
            if eeg_path is None:
                print(f"  [skip] {subject} task={task} run={run}: EEG not found")
                continue

            run_label = f"run-{run}" if run is not None else "run-1"
            out_name = f"sub-{subject}_task-{task}_{run_label}_features.npz"
            out_dir = features_dir / f"sub-{subject}"
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / out_name

            if out_path.exists():
                print(f"  [skip] {out_path.name} already exists")
                found_any = True
                continue

            print(f"  Processing {subject} task={task} run={run} ses={ses}")

            # --- fMRI ---
            bold_img = image.load_img(str(bold_path))
            n_vols = bold_img.shape[3]

            conf_mat = None
            if conf_path is not None:
                conf_mat = load_confounds(conf_path, confound_cols)
                # Trim to n_vols if needed
                if conf_mat.shape[0] > n_vols:
                    conf_mat = conf_mat[:n_vols]

            difumo_ts = extract_difumo_timeseries(bold_img, conf_mat, cfg)  # (T, 64)

            try:
                personal_ts = extract_personal_roi_timeseries(bold_img, subject, cfg)  # (T, 2)
            except FileNotFoundError as e:
                print(f"    [warn] {e}")
                print("    Filling personal ROIs with zeros — fix masks before training")
                personal_ts = np.zeros((n_vols, 2), dtype=np.float32)

            fmri_features = np.concatenate([difumo_ts, personal_ts], axis=1)  # (T, 66)
            pda = personal_ts[:, 1] - personal_ts[:, 0]  # CEN - DMN

            # --- EEG ---
            raw = mne.io.read_raw_fif(str(eeg_path), preload=True, verbose=False)
            eeg_block = eeg_block_mean(raw, tr, samples_per_tr, n_vols)  # (T, 31)

            # --- Save ---
            np.savez(
                out_path,
                fmri_features=fmri_features,  # (T, 66)
                eeg_block=eeg_block,           # (T, 31)
                pda=pda,                       # (T,)
                subject=subject,
                task=task,
                run=str(run),
                tr=tr,
                n_difumo=64,
                n_personal=2,
            )
            print(f"    Saved {out_path.name} — fmri {fmri_features.shape}, eeg {eeg_block.shape}")
            found_any = True

        if not found_any:
            print(f"  [warn] No data found for {subject} task={task}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", type=str, default=None)
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--config", type=str, default="config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    subjects = subject_list(cfg)

    if args.all:
        for subj in subjects:
            print(f"\n=== {subj} ===")
            extract_subject(subj, cfg)
    elif args.subject:
        extract_subject(args.subject, cfg)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()