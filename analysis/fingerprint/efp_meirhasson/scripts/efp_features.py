#!/usr/bin/env python3
"""
efp_features.py
---------------
EEG Finger-Print feature construction (Meir-Hasson 2014), per subject:

  raw 250 Hz EEG (single channel)
    -> Stockwell power (1 Hz, 1..40 Hz)
    -> 10 data-driven equal-energy frequency bands (per channel)
    -> down-sample band power to target resolution (TR or 4 Hz)
    -> [cached]

The [frequency x sliding-delay] design matrix and targets are assembled at decode
time from the cached band power (see make_delay_design / load_subject_features).

Targets (DMN/CEN/PDA + GSR variants) are built here too:
  - TR mode: native TR
  - 4 Hz mode: cubic-spline up-sampled to 4 Hz (paper's step)
"""
import argparse
from pathlib import Path
import numpy as np, pandas as pd, yaml, mne
from scipy.interpolate import interp1d

from stockwell import stockwell_power

mne.set_log_level("ERROR")

SCRIPT_DIR = Path(__file__).resolve().parent
PROJ_DIR = SCRIPT_DIR.parent
CONFIG_PATH = PROJ_DIR / "config.yaml"


# ── config / io ────────────────────────────────────────────────────────────
def load_config(p=CONFIG_PATH):
    cfg = yaml.safe_load(open(p))
    d = cfg["data"]
    suffix = "_cluster" if Path("/projects/swglab").exists() else "_local"
    for key in ("features_dir", "eeg_preproc_dir", "confounds_dir", "network_masks_dir"):
        if (key + suffix) in d:
            d[key] = str(Path(d[key + suffix]).expanduser())
    return cfg


def residualize(y, confound):
    """OLS residualize y on [1, confound] (global signal regression)."""
    X = np.column_stack([np.ones_like(confound), confound])
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    return y - X @ beta


def load_confounds_run(cfg, sub, run):
    d = cfg["data"]
    ses = d.get("session_fmri", d["session"])   # rtBPD: fMRI in ses-nf1
    tsv = (Path(d["confounds_dir"]) / f"sub-{sub}" / ses / "func" /
           f"sub-{sub}_{ses}_task-{d['task']}_run-{int(run):02d}_desc-confounds_timeseries.tsv")
    if not tsv.exists():
        return None
    df = pd.read_csv(tsv, sep="\t")
    if "global_signal" not in df.columns:
        return None
    gs = df["global_signal"].values.astype(float)
    if len(gs) and np.isnan(gs[0]):
        gs[0] = gs[1]
    return gs


def run_ids(cfg, sub):
    d = cfg["data"]
    fdir = Path(d["features_dir"]) / f"sub-{sub}"
    ids = []
    for npz in sorted(fdir.glob(f"sub-{sub}_task-{d['task']}_run-*_features.npz")):
        ids.append(npz.name.split("run-")[1].split("_")[0])
    return ids


def _load_visual_sphere(sub, run, n_tr):
    """Load the precomputed calcarine-sphere VIS timeseries for one run, or None."""
    f = PROJ_DIR / "results" / "visual_sphere" / f"{sub}.npz"
    if not f.exists():
        return None
    z = np.load(f, allow_pickle=True)
    key = str(run)
    if key not in z.files:
        return None
    return np.asarray(z[key], float)[:n_tr]


def load_targets_run(cfg, sub, run):
    """DMN/CEN/PDA + GSR variants at native TR for one run. Returns (dict, n_tr)."""
    d = cfg["data"]
    fdir = Path(d["features_dir"]) / f"sub-{sub}"
    cands = sorted(fdir.glob(f"sub-{sub}_task-{d['task']}_run-*_features.npz"))
    npz = None
    for c in cands:
        rid = c.name.split("run-")[1].split("_")[0]
        if str(rid) == str(run):
            npz = c; break
    if npz is None:
        return None, 0
    z = np.load(npz, allow_pickle=True)
    fm = np.asarray(z["fmri_features"], float)
    dmn = fm[:, d["fmri"]["dmn_idx"]]; cen = fm[:, d["fmri"]["cen_idx"]]
    n_tr = fm.shape[0]
    out = dict(DMN=dmn, CEN=cen, PDA=cen - dmn)
    # visual-cortex positive control: focal 6mm calcarine sphere (paper Fig 5a),
    # precomputed by extract_visual_sphere.py -> results/visual_sphere/{sub}.npz
    vis = _load_visual_sphere(sub, run, n_tr)
    if vis is not None:
        out["VIS"] = vis
    gs = load_confounds_run(cfg, sub, run)
    if gs is not None:
        gs = gs[:n_tr]
        gdmn = residualize(dmn.copy(), gs); gcen = residualize(cen.copy(), gs)
        out.update(GSR_DMN=gdmn, GSR_CEN=gcen, GSR_PDA=gcen - gdmn)
    return out, n_tr


def load_eeg_run(cfg, sub, run):
    """Load preprocessed 250 Hz EEG for one run. Returns (data (n_ch, n_samp), ch_names)."""
    d = cfg["data"]
    ses = d.get("session_eeg", d["session"])    # rtBPD: EEG in ses-nf
    fif = (Path(d["eeg_preproc_dir"]) / f"sub-{sub}" / ses / "eeg" /
           f"sub-{sub}_{ses}_task-{d['task']}_run-{int(run):02d}_desc-{d['eeg']['desc']}_eeg.fif")
    if not fif.exists():
        return None, None
    raw = mne.io.read_raw_fif(str(fif), preload=True, verbose=False)
    raw.pick(mne.pick_types(raw.info, eeg=True, exclude=[]))
    sf = d["eeg"]["sfreq"]
    if abs(raw.info["sfreq"] - sf) > 1e-3:
        raw.resample(sf, verbose=False)
    return raw.get_data(), raw.ch_names


# ── EFP feature ops ─────────────────────────────────────────────────────────
def equal_energy_edges(spectrum, freqs, n_bands):
    """Split freqs into n_bands contiguous bands with equal cumulative log-power.

    Returns list of (lo_idx, hi_idx) inclusive index ranges into `freqs`.
    """
    dens = np.log(spectrum + 1e-20)
    dens = dens - dens.min() + 1e-6          # make a positive density over frequency
    cum = np.cumsum(dens); cum /= cum[-1]
    targets = np.linspace(0, 1, n_bands + 1)[1:-1]
    cut_idx = np.searchsorted(cum, targets)
    bounds = [0] + list(cut_idx) + [len(freqs) - 1]
    edges = []
    for b in range(n_bands):
        lo = bounds[b] if b == 0 else bounds[b] + 1
        hi = bounds[b + 1]
        if hi < lo:
            hi = lo
        edges.append((lo, hi))
    return edges


def bin_average(power, n_out):
    """Average a (n_bands, n_samp) array into (n_bands, n_out) contiguous time bins."""
    n_bands, n_samp = power.shape
    edges = np.linspace(0, n_samp, n_out + 1).astype(int)
    out = np.empty((n_bands, n_out))
    for i in range(n_out):
        a, b = edges[i], max(edges[i + 1], edges[i] + 1)
        out[:, i] = power[:, a:b].mean(axis=1)
    return out


def channel_bandpower(x, sfreq, fmin, fmax, n_bands, ta_bands=None):
    """S-transform -> equal-energy bands -> (n_bands, n_samp) power, band edges (Hz),
    plus fixed theta/alpha power (n_ta, n_samp) for the T/A baseline if ta_bands given."""
    freqs, power = stockwell_power(x, sfreq, fmin, fmax)   # (n_freq, n_samp)
    spec = power.mean(axis=1)                              # mean spectrum
    edges = equal_energy_edges(spec, freqs, n_bands)
    bp = np.empty((n_bands, power.shape[1]))
    for i, (lo, hi) in enumerate(edges):
        bp[i] = power[lo:hi + 1].mean(axis=0)
    band_hz = [(int(freqs[lo]), int(freqs[hi])) for lo, hi in edges]
    ta = None
    if ta_bands is not None:
        ta = np.empty((len(ta_bands), power.shape[1]))
        for i, (lo, hi) in enumerate(ta_bands):
            m = (freqs >= lo) & (freqs <= hi)
            ta[i] = power[m].mean(axis=0)
    return bp, band_hz, ta


def make_delay_design(bp_run, n_delays):
    """Build sliding-delay design for one run.

    bp_run : (n_bands, n_out) band power at target resolution.
    Returns X (n_out - n_delays + 1, n_bands * n_delays); row t uses lags
    [t, t-1, ..., t-(n_delays-1)] (0..-max delay). Also returns the valid index
    offset (n_delays-1) so targets can be trimmed to match.
    """
    n_bands, n_out = bp_run.shape
    n_valid = n_out - (n_delays - 1)
    if n_valid <= 0:
        return np.empty((0, n_bands * n_delays)), n_delays - 1
    X = np.empty((n_valid, n_bands * n_delays))
    for d in range(n_delays):
        # lag d: value at time (t - d) for t in [n_delays-1 .. n_out-1]
        seg = bp_run[:, (n_delays - 1 - d):(n_out - d)].T   # (n_valid, n_bands)
        X[:, d * n_bands:(d + 1) * n_bands] = seg
    return X, n_delays - 1


# ── build + cache per subject ───────────────────────────────────────────────
def build_subject_features(cfg, sub, cache_dir):
    cache = cache_dir / f"{sub}_efp.npz"
    if cache.exists():
        return cache
    d = cfg["data"]; e = cfg["efp"]
    sf = d["eeg"]["sfreq"]; tr = d["fmri"]["tr"]; hz4 = e["hz4"]
    fmin, fmax, n_bands = e["freq_min"], e["freq_max"], e["n_bands"]

    rids = run_ids(cfg, sub)
    runs_out = []
    ch_names = None
    for run in rids:
        eeg, chs = load_eeg_run(cfg, sub, run)
        tgt, n_tr = load_targets_run(cfg, sub, run)
        if eeg is None or tgt is None or n_tr == 0:
            continue
        ch_names = chs
        n_samp = eeg.shape[1]
        n_hz4 = int(round(n_samp / sf * hz4))
        ta_bands = [tuple(cfg["baseline"]["theta"]), tuple(cfg["baseline"]["alpha"])]
        # per-channel band power at both resolutions
        bp_tr = np.empty((len(chs), n_bands, n_tr))
        bp_hz4 = np.empty((len(chs), n_bands, n_hz4))
        ta_tr = np.empty((len(chs), 2, n_tr))
        ta_hz4 = np.empty((len(chs), 2, n_hz4))
        band_hz = None
        for ci in range(len(chs)):
            bp, bhz, ta = channel_bandpower(eeg[ci], sf, fmin, fmax, n_bands, ta_bands)
            band_hz = bhz
            bp_tr[ci] = bin_average(bp, n_tr)
            bp_hz4[ci] = bin_average(bp, n_hz4)
            ta_tr[ci] = bin_average(ta, n_tr)
            ta_hz4[ci] = bin_average(ta, n_hz4)
        # targets: TR native; 4 Hz cubic-spline upsample
        tgt_tr = {k: v[:n_tr] for k, v in tgt.items()}
        t_src = np.linspace(0, 1, n_tr)
        t_dst = np.linspace(0, 1, n_hz4)
        tgt_hz4 = {k: interp1d(t_src, v[:n_tr], kind="cubic")(t_dst) for k, v in tgt.items()}
        runs_out.append(dict(run=run, n_tr=n_tr, n_hz4=n_hz4,
                             bp_tr=bp_tr, bp_hz4=bp_hz4,
                             ta_tr=ta_tr, ta_hz4=ta_hz4,
                             tgt_tr=tgt_tr, tgt_hz4=tgt_hz4, band_hz=band_hz))
        print(f"  {sub} run {run}: n_tr={n_tr} n_hz4={n_hz4} bands={band_hz}")

    if not runs_out:
        print(f"  {sub}: no runs")
        return None
    cache_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, runs=np.array(runs_out, dtype=object),
                        ch_names=np.array(ch_names))
    print(f"  saved {cache}")
    return cache


def load_subject_features(cache_dir, sub):
    z = np.load(cache_dir / f"{sub}_efp.npz", allow_pickle=True)
    return list(z["runs"]), list(z["ch_names"])


def refresh_targets_in_cache(cfg, sub, cache_dir):
    """Recompute the fMRI targets (incl. any newly-added ones like VIS) from
    cyclic_features and update an existing cache in place — WITHOUT recomputing
    the EEG band power. Cheap way to add a target to already-built caches."""
    cache = cache_dir / f"{sub}_efp.npz"
    if not cache.exists():
        print(f"  {sub}: no cache to refresh")
        return
    z = np.load(cache, allow_pickle=True)
    runs = list(z["runs"]); ch_names = list(z["ch_names"])
    changed = 0
    for rd in runs:
        tgt, n_tr = load_targets_run(cfg, sub, rd["run"])
        if tgt is None:
            continue
        m = min(n_tr, rd["n_tr"])
        t_src = np.linspace(0, 1, rd["n_tr"]); t_dst = np.linspace(0, 1, rd["n_hz4"])
        rd["tgt_tr"] = {k: v[:rd["n_tr"]] for k, v in tgt.items()}
        rd["tgt_hz4"] = {k: interp1d(t_src, v[:rd["n_tr"]], kind="cubic")(t_dst)
                         for k, v in tgt.items()}
        changed += 1
    np.savez_compressed(cache, runs=np.array(runs, dtype=object),
                        ch_names=np.array(ch_names))
    print(f"  {sub}: refreshed targets in {changed} runs "
          f"({', '.join(runs[0]['tgt_tr'].keys()) if runs else ''})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subjects", nargs="+", default=None)
    ap.add_argument("--group", action="store_true")
    ap.add_argument("--refresh-targets", action="store_true",
                    help="update fMRI targets in existing caches (e.g. add VIS) "
                         "without recomputing EEG band power")
    ap.add_argument("--cache", default=str(PROJ_DIR / "results" / "features_cache"))
    ap.add_argument("--config", default=str(CONFIG_PATH))
    args = ap.parse_args()
    cfg = load_config(args.config)
    if args.group:
        subs = cfg["data"]["subjects"]["all"]
    elif args.subjects:
        subs = args.subjects
    else:
        subs = cfg["data"]["subjects"]["pilot"]
    cache_dir = Path(args.cache)
    for sub in subs:
        print(f"[{sub}]")
        if args.refresh_targets:
            refresh_targets_in_cache(cfg, sub, cache_dir)
        else:
            build_subject_features(cfg, sub, cache_dir)


if __name__ == "__main__":
    main()
