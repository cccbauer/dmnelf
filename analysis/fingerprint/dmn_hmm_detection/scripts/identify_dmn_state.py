#!/usr/bin/env python3
"""
identify_dmn_state.py
---------------------
Identify which of the K HMM states is the "DMN state" by correlating each
state's occupancy timeseries (HRF-convolved, TR-binned) against the SIMULTANEOUS
fMRI DMN timeseries — the validation Cooray et al. 2024 could not do (they only
had spectral matching, no concurrent fMRI).

Uses the saved state probabilities (results/<model>/state_probabilities.npz);
requires only numpy/scipy/pandas (no osl_dynamics / TensorFlow).

For each rest run:
  1. Load per-state probability alpha (n_eeg_samples, K) @ 250 Hz.
  2. Bin to TR resolution (300 EEG samples / TR at 250 Hz, 1.2 s TR).
  3. HRF-convolve each state's TR-occupancy (state activity -> BOLD).
  4. Load simultaneous fMRI DMN/CEN/PDA (+ GSR'd variants) for the same run.
  5. Pearson-correlate each state vs each target.

Aggregate mean r across runs; the DMN state = state most positively correlated
with fMRI DMN (and, as a cross-check, most negatively correlated with PDA=CEN-DMN).
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import yaml
from scipy.stats import gamma, pearsonr

SCRIPT_DIR = Path(__file__).resolve().parent
PROJ_DIR = SCRIPT_DIR.parent
CONFIG_PATH = PROJ_DIR / "config.yaml"

EEG_SFREQ = 250.0
TR = 1.2
SAMPLES_PER_TR = int(round(EEG_SFREQ * TR))  # 300


def load_config(p):
    cfg = yaml.safe_load(open(p))
    d = cfg["data"]
    suffix = "_cluster" if Path("/projects/swglab").exists() else "_local"
    for key in ("features_dir", "eeg_preproc_dir", "confounds_dir"):
        d[key] = str(Path(d[key + suffix]).expanduser())
    return cfg


def canonical_hrf(tr, length_s=32, delay=6, undershoot=16):
    """SPM-style double-gamma HRF sampled at the TR. Sums to 1."""
    t = np.arange(0, length_s, tr)
    h = gamma.pdf(t, delay) - gamma.pdf(t, undershoot) / 6.0
    return h / h.sum()


def hrf_convolve(x, hrf):
    return np.convolve(x, hrf, mode="full")[:len(x)]


def residualize(y, confound):
    """OLS residualize y on [1, confound]."""
    X = np.column_stack([np.ones_like(confound), confound])
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    return y - X @ beta


def bin_to_tr(alpha_run, n_tr):
    """Bin (n_samples, K) @250Hz into (n_tr, K) by averaging SAMPLES_PER_TR windows."""
    K = alpha_run.shape[1]
    usable = min(alpha_run.shape[0] // SAMPLES_PER_TR, n_tr)
    binned = alpha_run[:usable * SAMPLES_PER_TR].reshape(usable, SAMPLES_PER_TR, K).mean(axis=1)
    return binned  # (usable, K)


def load_fmri_targets(cfg, sub, run):
    """Load DMN/CEN/PDA + GSR'd variants for one rest run. Returns dict or None."""
    d = cfg["data"]
    fdir = Path(d["features_dir"]) / f"sub-{sub}"
    npz = fdir / f"sub-{sub}_task-{d['task_rest']}_run-{int(run):02d}_features.npz"
    if not npz.exists():
        # some datasets store rest without run in name; try glob
        cands = sorted(fdir.glob(f"sub-{sub}_task-{d['task_rest']}_run-*_features.npz"))
        if not cands:
            return None
        npz = cands[min(int(run) - 1, len(cands) - 1)]
    z = np.load(npz, allow_pickle=True)
    fm = np.asarray(z["fmri_features"], float)  # (n_tr, 66)
    dmn = fm[:, d["fmri"]["dmn_idx"]]
    cen = fm[:, d["fmri"]["cen_idx"]]
    pda = cen - dmn

    gs = load_global_signal(cfg, sub, run, n_tr=fm.shape[0])
    if gs is not None:
        gdmn = residualize(dmn.copy(), gs)
        gcen = residualize(cen.copy(), gs)
        gpda = gcen - gdmn
    else:
        gdmn = gcen = gpda = None
    return dict(DMN=dmn, CEN=cen, PDA=pda,
                GSR_DMN=gdmn, GSR_CEN=gcen, GSR_PDA=gpda)


def load_global_signal(cfg, sub, run, n_tr):
    d = cfg["data"]
    cdir = Path(d["confounds_dir"])
    ses = d["session"]
    tsv = (cdir / f"sub-{sub}" / ses / "func" /
           f"sub-{sub}_{ses}_task-{d['task_rest']}_run-{int(run):02d}_desc-confounds_timeseries.tsv")
    if not tsv.exists():
        return None
    df = pd.read_csv(tsv, sep="\t")
    if "global_signal" not in df.columns:
        return None
    gs = df["global_signal"].values.astype(float)
    if len(gs) and np.isnan(gs[0]):
        gs[0] = gs[1]
    return gs[:n_tr]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(CONFIG_PATH))
    ap.add_argument("--model", default="group_k12", help="results/<model>/ dir name")
    args = ap.parse_args()

    cfg = load_config(args.config)
    res_dir = PROJ_DIR / "results" / args.model
    npz = np.load(res_dir / "state_probabilities.npz", allow_pickle=True)
    alpha = npz["alpha"]
    labels = [tuple(x) for x in npz["subject_run_labels"]]
    K = int(npz["n_states"]) if "n_states" in npz else alpha[0].shape[1]

    hrf = canonical_hrf(tr=TR, length_s=cfg["hrf"]["length_s"],
                        delay=cfg["hrf"]["delay"], undershoot=cfg["hrf"]["undershoot"])

    targets = ["DMN", "CEN", "PDA", "GSR_DMN", "GSR_CEN", "GSR_PDA"]
    # collect per (state, target) list of per-run r
    rvals = {t: [[] for _ in range(K)] for t in targets}
    n_runs_used = 0
    n_runs_skipped = 0

    for ai, (sub, run) in enumerate(labels):
        tgt = load_fmri_targets(cfg, sub, run)
        if tgt is None:
            n_runs_skipped += 1
            continue
        a = np.asarray(alpha[ai], dtype=np.float64)  # (n_samp, K)
        n_tr_fmri = len(tgt["DMN"])
        occ = bin_to_tr(a, n_tr_fmri)                 # (usable, K)
        usable = occ.shape[0]
        # HRF-convolve each state occupancy at TR resolution
        occ_h = np.column_stack([hrf_convolve(occ[:, k], hrf) for k in range(K)])
        n_runs_used += 1
        for t in targets:
            y = tgt[t]
            if y is None:
                continue
            y = y[:usable]
            if np.std(y) < 1e-9:
                continue
            for k in range(K):
                x = occ_h[:usable, k]
                if np.std(x) < 1e-9:
                    continue
                r, _ = pearsonr(x, y)
                rvals[t][k].append(r)

    # Aggregate
    rows = []
    for t in targets:
        for k in range(K):
            arr = np.array(rvals[t][k])
            if arr.size == 0:
                rows.append(dict(target=t, state=k + 1, mean_r=np.nan, sem_r=np.nan, n=0))
            else:
                rows.append(dict(target=t, state=k + 1, mean_r=arr.mean(),
                                 sem_r=arr.std(ddof=1) / np.sqrt(arr.size) if arr.size > 1 else np.nan,
                                 n=arr.size))
    df = pd.DataFrame(rows)
    out_csv = res_dir / "state_fmri_correlations.csv"
    df.to_csv(out_csv, index=False)

    print(f"Runs used: {n_runs_used}, skipped (no fMRI): {n_runs_skipped}\n")

    def top_states(target, ascending=False):
        sub = df[df.target == target].sort_values("mean_r", ascending=ascending)
        return sub.head(3)

    print("=== Mean state-vs-fMRI correlations (per state, HRF-convolved) ===")
    for t in targets:
        sub = df[df.target == t].set_index("state")["mean_r"]
        print(f"\n{t}:")
        print("  " + "  ".join(f"S{k+1}:{sub.get(k+1, np.nan):+.3f}" for k in range(K)))

    print("\n=== DMN-state candidates ===")
    print("Top +corr with fMRI DMN (raw):")
    print(top_states("DMN", ascending=False)[["state", "mean_r", "n"]].to_string(index=False))
    print("\nTop +corr with GSR_DMN:")
    print(top_states("GSR_DMN", ascending=False)[["state", "mean_r", "n"]].to_string(index=False))
    print("\nTop -corr with PDA (CEN-DMN), i.e. DMN>CEN:")
    print(top_states("PDA", ascending=True)[["state", "mean_r", "n"]].to_string(index=False))

    # Best DMN state by GSR_DMN if available else raw DMN
    key = "GSR_DMN" if df[(df.target == "GSR_DMN")]["n"].sum() > 0 else "DMN"
    best = df[df.target == key].sort_values("mean_r", ascending=False).iloc[0]
    print(f"\n>>> DMN state = State {int(best.state)} "
          f"({key} r={best.mean_r:+.3f}, n={int(best.n)})")
    print(f"Saved correlations to {out_csv}")


if __name__ == "__main__":
    main()
