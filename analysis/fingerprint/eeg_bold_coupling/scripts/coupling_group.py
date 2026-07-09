#!/usr/bin/env python
"""
coupling_group.py
-----------------
Group-level analysis of EEG-BOLD coupling after per-subject feature extraction.

Computes:
- Within-subject correlations (EEG band power → DMN/CEN/PDA)
- Group-level t-tests against zero (sign-flip permutation null)
- Multiple comparison correction (max-statistic across bands/channels/targets)

Run after bandpower.py has extracted features for all subjects.
"""
import sys
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import yaml
import mne
from scipy.stats import ttest_1samp, t
from joblib import Parallel, delayed

# Add parent to path for shared utils
sys.path.insert(0, str(Path(__file__).resolve().parent))
from bandpower import load_config, gather_subject, canonical_hrf

mne.set_log_level("ERROR")


# ----------------------------------------------------------------------
# Within-subject coupling (correlation between EEG band power and target)
# ----------------------------------------------------------------------
def subject_coupling(runs, target_name, bands, zscore_eeg=True, zscore_target=True):
    """
    Compute correlation for each (band, channel) across all runs of one subject.

    Parameters
    ----------
    runs : list of dicts from gather_subject
    target_name : str, 'DMN', 'CEN', or 'PDA'
    bands : list of band names (e.g., ['delta','theta',...])
    zscore_eeg : bool, z-score each band's power per run (within-run)
    zscore_target : bool, z-score target per run

    Returns
    -------
    r_mat : dict band -> np.ndarray (n_channels,)
        Correlation coefficients per channel
    """
    # Collect all TRs across runs into one array per band
    # We'll concatenate runs (preserving time order but runs are independent)
    all_eeg = {b: [] for b in bands}
    all_target = []

    for run in runs:
        n_tr = run["n_tr"]
        target = run["targets"][target_name]
        if zscore_target:
            target = (target - target.mean()) / (target.std() + 1e-12)
        all_target.append(target)

        for band in bands:
            bp = run["bp"][band]          # (n_tr, n_ch)
            if zscore_eeg:
                bp = (bp - bp.mean(axis=0)) / (bp.std(axis=0) + 1e-12)
            all_eeg[band].append(bp)

    # Concatenate across runs
    all_target = np.concatenate(all_target, axis=0)
    r_out = {}
    for band in bands:
        X = np.concatenate(all_eeg[band], axis=0)   # (total_tr, n_ch)
        # Correlation with target (vector) per channel
        r = np.corrcoef(X.T, all_target)[:-1, -1]   # (n_ch,)
        r_out[band] = r
    return r_out


# ----------------------------------------------------------------------
# Group-level stats with sign-flip permutation (max-statistic corrected)
# ----------------------------------------------------------------------
def group_coupling_all_subjects(cfg, subjects, target_name, bands, n_perm=2000):
    """
    For each subject, get correlation maps (band, channel).
    Then group t-test against zero, corrected with sign-flip permutation.

    Returns
    -------
    results : dict
        t_obs : dict band -> np.ndarray (n_channels,)
        p_fwer : dict band -> np.ndarray (n_channels,)
        t_thresh : float (FWER threshold)
    """
    tr = cfg["data"]["fmri"]["tr"]
    hrf_params = cfg["hrf"]
    hrf = canonical_hrf(tr, hrf_params["length_s"], hrf_params["delay"], hrf_params["undershoot"])

    # Collect per-subject correlation maps (n_subj, n_ch)
    n_ch = None
    subj_maps = {b: [] for b in bands}

    for subj in subjects:
        print(f"  Processing {subj}...")
        runs = gather_subject(cfg, subj, hrf)
        if not runs:
            print(f"    No runs found for {subj}, skipping")
            continue
        r_map = subject_coupling(runs, target_name, bands, zscore_eeg=True, zscore_target=True)
        for b in bands:
            subj_maps[b].append(r_map[b])
            if n_ch is None:
                n_ch = len(r_map[b])

    # Convert to arrays (n_subj, n_ch)
    for b in bands:
        subj_maps[b] = np.array(subj_maps[b])
    n_subj = len(subj_maps[bands[0]])

    # Observed t-values (one-sample t-test against 0)
    t_obs = {}
    for b in bands:
        t_obs[b], _ = ttest_1samp(subj_maps[b], 0, axis=0)

    # Permutation null: sign-flip within each subject
    # Max-statistic over all bands and channels
    max_t_perm = np.zeros(n_perm)
    for perm in range(n_perm):
        # Random sign flips for each subject (per channel, but we flip entire subject's map)
        signs = np.random.choice([-1, 1], size=(n_subj, 1))
        t_perm_band = []
        for b in bands:
            flipped = subj_maps[b] * signs
            t_perm, _ = ttest_1samp(flipped, 0, axis=0)
            t_perm_band.append(t_perm)
        t_perm_all = np.concatenate(t_perm_band)  # (n_bands * n_ch,)
        max_t_perm[perm] = np.max(np.abs(t_perm_all))
        if (perm+1) % 500 == 0:
            print(f"    Permutation {perm+1}/{n_perm}")

    # FWER threshold at alpha=0.05
    t_thresh = np.percentile(max_t_perm, 95)
    # Compute FWER-corrected p-values per channel/band
    p_fwer = {}
    for b in bands:
        p_unc = 2 * (1 - t.cdf(np.abs(t_obs[b]), df=n_subj-1))
        # Adjust: p_fwer = proportion of permutations where max_t > |t_obs|
        # But to get per-channel we need to compare each t_obs against null distribution of max
        # Standard approach: p_fwer = (number of permutations with max_t >= |t_obs[b][c]|) / n_perm
        # We'll compute per channel by counting how many permutations had max_t >= that channel's t
        p_fwer_b = np.zeros_like(t_obs[b])
        for c in range(n_ch):
            p_fwer_b[c] = np.mean(max_t_perm >= np.abs(t_obs[b][c]))
        p_fwer[b] = p_fwer_b

    return {"t_obs": t_obs, "p_fwer": p_fwer, "t_thresh": t_thresh, "n_subj": n_subj}


# ----------------------------------------------------------------------
# Main entry point
# ----------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Group-level EEG-BOLD coupling analysis")
    parser.add_argument("--offline", action="store_true", help="Use local offline data paths")
    parser.add_argument("--subject", type=str, default="group",
                        help="Ignored; kept for compatibility (always runs group)")
    parser.add_argument("--target", type=str, default="PDA",
                        choices=["DMN", "CEN", "PDA"],
                        help="Target network to decode")
    parser.add_argument("--n_perm", type=int, default=2000,
                        help="Number of sign-flip permutations for FWER correction")
    args = parser.parse_args()

    config_path = Path(__file__).resolve().parent.parent / "config.yaml"
    cfg = load_config(config_path, offline=args.offline)

    # Override output directory for offline mode
    if args.offline:
        # Set base_dir to local repo results folder
        repo_root = Path(__file__).resolve().parent.parent.parent  # up to dmnelf/
        cfg["project"]["base_dir"] = str(repo_root / "analysis/fingerprint/eeg_bold_coupling")
        print(f"Offline mode: saving results to {cfg['project']['base_dir']}")
    else:
        # Keep cluster base_dir as is (already loaded from config)
        pass

    subjects = cfg["data"]["subjects"]["all"]
    bands = list(cfg["bands"].keys())

    print(f"Running group coupling for target={args.target}")
    print(f"Subjects: {subjects}")
    print(f"Bands: {bands}")

    results = group_coupling_all_subjects(cfg, subjects, args.target, bands, n_perm=args.n_perm)

    # Save results
    out_dir = Path(cfg["project"]["base_dir"]) / "results"
    out_dir.mkdir(exist_ok=True, parents=True)
    for band in bands:
        t_arr = results["t_obs"][band]
        p_arr = results["p_fwer"][band]
        # Save as numpy
        np.savez(out_dir / f"group_{args.target}_{band}_coupling.npz",
                 t_obs=t_arr, p_fwer=p_arr, t_thresh=results["t_thresh"], n_subj=results["n_subj"])

    print(f"\nResults saved to {out_dir}")
    print(f"FWER threshold t = {results['t_thresh']:.3f}")
    for band in bands:
        n_sig = np.sum(results["p_fwer"][band] < 0.05)
        print(f"{band}: {n_sig} significant channels (FWER p<0.05)")