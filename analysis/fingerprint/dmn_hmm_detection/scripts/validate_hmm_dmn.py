#!/usr/bin/env python3
"""
validate_hmm_dmn.py
-------------------
Apply the trained group TIDE-HMM to FEEDBACK-run EEG and test whether the
fMRI-identified DMN state (State 7) occupancy tracks the simultaneous fMRI DMN
during feedback — then compare against the multivariate regression benchmark.

Applying to new data requires the SAME TDE-PCA transform as training, so we
rebuild the training Data to recover its fitted pca_components and pass them to
the feedback Data's tde_pca (fit once, apply to new data).

Outputs:
  results/<model>/feedback_state_correlations.csv
"""
import argparse, warnings
from pathlib import Path
import numpy as np, pandas as pd, yaml, mne
from scipy.stats import gamma, pearsonr

warnings.filterwarnings("ignore")
mne.set_log_level("ERROR")

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
    t = np.arange(0, length_s, tr)
    h = gamma.pdf(t, delay) - gamma.pdf(t, undershoot) / 6.0
    return h / h.sum()


def hrf_convolve(x, hrf):
    return np.convolve(x, hrf, mode="full")[:len(x)]


def residualize(y, confound):
    X = np.column_stack([np.ones_like(confound), confound])
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    return y - X @ beta


def load_task_eeg(cfg, sub, task, runs=(1, 2)):
    """Load preprocessed EEG runs for a task -> list of (run_number, (n_samp, n_ch))."""
    d = cfg["data"]; ses = d["session"]; eroot = Path(d["eeg_preproc_dir"])
    ec = d["eeg"]; lo, hi = ec["bandpass"]; desc = ec["desc"]; sf = ec["sfreq_hmm"]
    out = []
    for run in runs:
        fif = (eroot / f"sub-{sub}" / ses / "eeg" /
               f"sub-{sub}_{ses}_task-{task}_run-{run:02d}_desc-{desc}_eeg.fif")
        if not fif.exists():
            continue
        raw = mne.io.read_raw_fif(str(fif), preload=True, verbose=False)
        raw.pick(mne.pick_types(raw.info, eeg=True, exclude=[]))
        raw.filter(lo, hi, verbose=False)
        raw.resample(sf, verbose=False)
        out.append((run, raw.get_data().T.astype(np.float32)))
    return out


def bin_to_tr(alpha_run, n_tr):
    K = alpha_run.shape[1]
    usable = min(alpha_run.shape[0] // SAMPLES_PER_TR, n_tr)
    return alpha_run[:usable * SAMPLES_PER_TR].reshape(usable, SAMPLES_PER_TR, K).mean(axis=1)


def load_fmri_targets(cfg, sub, task, run):
    d = cfg["data"]
    fdir = Path(d["features_dir"]) / f"sub-{sub}"
    cands = sorted(fdir.glob(f"sub-{sub}_task-{task}_run-*_features.npz"))
    npz = None
    for c in cands:
        if f"run-{int(run)}_" in c.name or f"run-{int(run):02d}_" in c.name:
            npz = c; break
    if npz is None:
        return None
    z = np.load(npz, allow_pickle=True)
    fm = np.asarray(z["fmri_features"], float)
    dmn = fm[:, d["fmri"]["dmn_idx"]]; cen = fm[:, d["fmri"]["cen_idx"]]
    tsv = (Path(d["confounds_dir"]) / f"sub-{sub}" / d["session"] / "func" /
           f"sub-{sub}_{d['session']}_task-{task}_run-{int(run):02d}_desc-confounds_timeseries.tsv")
    gs = None
    if tsv.exists():
        df = pd.read_csv(tsv, sep="\t")
        if "global_signal" in df.columns:
            gs = df["global_signal"].values.astype(float)
            if len(gs) and np.isnan(gs[0]):
                gs[0] = gs[1]
            gs = gs[:len(dmn)]
    out = dict(DMN=dmn, CEN=cen, PDA=cen - dmn)
    if gs is not None:
        gdmn = residualize(dmn.copy(), gs); gcen = residualize(cen.copy(), gs)
        out.update(GSR_DMN=gdmn, GSR_CEN=gcen, GSR_PDA=gcen - gdmn)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(CONFIG_PATH))
    ap.add_argument("--model", default="group_k12")
    ap.add_argument("--dmn_state", type=int, default=7)
    args = ap.parse_args()

    cfg = load_config(args.config)
    res_dir = PROJ_DIR / "results" / args.model
    subjects = cfg["data"]["subjects"]["all"]
    task_rest = cfg["data"]["task_rest"]; task_fb = cfg["data"]["task_feedback"]
    K = cfg["hmm"]["n_states"]
    emb = cfg["hmm"]["embedding_lag"]
    n_pca = cfg["hmm"].get("n_pca_components", 80)

    from osl_dynamics.data import Data
    from osl_dynamics.models import load

    # ── Rebuild training Data to recover fitted PCA components ──
    print("Rebuilding training (rest) data to recover PCA transform...")
    rest_runs = []
    for sub in subjects:
        for _, d in load_task_eeg(cfg, sub, task_rest):
            rest_runs.append(d)
    train = Data(rest_runs, sampling_frequency=EEG_SFREQ)
    train.tde_pca(n_embeddings=emb, n_pca_components=n_pca, whiten=True)
    train.standardize()
    pca_components = train.pca_components
    print(f"  Recovered PCA components: {pca_components.shape}")

    model = load(str(res_dir / "trained_model"))
    hrf = canonical_hrf(TR, length_s=cfg["hrf"]["length_s"],
                        delay=cfg["hrf"]["delay"], undershoot=cfg["hrf"]["undershoot"])
    targets = ["DMN", "CEN", "PDA", "GSR_DMN", "GSR_CEN", "GSR_PDA"]
    ds = args.dmn_state

    rows = []
    for sub in subjects:
        fb = load_task_eeg(cfg, sub, task_fb, runs=(1, 2, 3, 4))
        if not fb:
            continue
        run_nums = [r for r, _ in fb]
        fb_data = [d for _, d in fb]
        # prepare feedback data with the SAME transform (apply training PCA)
        fdata = Data(fb_data, sampling_frequency=EEG_SFREQ)
        # reuse the training PCA: pass ONLY pca_components (not n_pca_components)
        fdata.tde_pca(n_embeddings=emb, pca_components=pca_components, whiten=True)
        fdata.standardize()
        alpha = model.get_alpha(fdata)           # list per run
        if not isinstance(alpha, list):
            alpha = [alpha]
        for ri, run in enumerate(run_nums):
            tgt = load_fmri_targets(cfg, sub, task_fb, run)
            if tgt is None:
                continue
            a = np.asarray(alpha[ri], dtype=np.float64)
            n_tr = len(tgt["DMN"])
            occ = bin_to_tr(a, n_tr)
            usable = occ.shape[0]
            occ_h = hrf_convolve(occ[:, ds - 1], hrf)[:usable]
            row = dict(subject=sub, run=run, n_tr=usable)
            for t in targets:
                y = tgt.get(t)
                if y is None:
                    row[t] = np.nan; continue
                y = y[:usable]
                if np.std(y) < 1e-9 or np.std(occ_h) < 1e-9:
                    row[t] = np.nan; continue
                row[t], _ = pearsonr(occ_h, y)
            rows.append(row)
            print(f"  {sub} fb run {run}: DMN r={row.get('DMN', np.nan):+.3f} "
                  f"GSR_DMN r={row.get('GSR_DMN', np.nan):+.3f}")

    df = pd.DataFrame(rows)
    out_csv = res_dir / "feedback_state_correlations.csv"
    df.to_csv(out_csv, index=False)

    print(f"\n=== Feedback validation: State {ds} occupancy vs fMRI (n={len(df)} runs) ===")
    for t in targets:
        if t in df:
            vals = df[t].dropna().values
            if len(vals):
                m = vals.mean(); sem = vals.std(ddof=1) / np.sqrt(len(vals))
                # one-sample t vs 0
                from scipy.stats import ttest_1samp
                tstat, p = ttest_1samp(vals, 0.0)
                print(f"  {t:8s}: mean r={m:+.3f} ± {sem:.3f}  (t={tstat:+.2f}, p={p:.3f}, n={len(vals)})")
    print(f"\nRegression benchmark (eeg_bold_coupling multivariate): "
          f"GSR_DMN r≈0.12, GSR_CEN r≈0.17")
    print(f"Saved to {out_csv}")


if __name__ == "__main__":
    main()
