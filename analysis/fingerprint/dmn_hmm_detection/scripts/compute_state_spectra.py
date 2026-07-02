#!/usr/bin/env python3
"""
compute_state_spectra.py
------------------------
Cross-check the fMRI-identified DMN state (State 7) against the spectral method
used by Cooray et al. 2024, who lacked simultaneous fMRI and identified the DMN
state purely by its spectral signature (posterior alpha + frontal delta/theta).

For each rest run we reload the raw 31-channel EEG (250 Hz, 1-45 Hz), align it to
the saved HMM state probabilities, and run osl_dynamics multitaper to get
state-specific power spectra (n_states, n_channels, n_freqs). We then report, per
state, the band power per channel and flag whether the fMRI-DMN state matches the
posterior-alpha / frontal-theta signature.

Outputs:
  results/<model>/state_spectra.npz      (f, psd[K,ch,F], channel names)
  results/<model>/state_band_power.csv   (state x band x region mean power)
  results/<model>/figures/state_topomaps_alpha.png   (alpha topomap per state)
  results/<model>/figures/dmn_state_spectrum.png     (DMN-state spectra by region)
"""
import argparse, warnings
from pathlib import Path
import numpy as np, pandas as pd, yaml, mne
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")
mne.set_log_level("ERROR")

SCRIPT_DIR = Path(__file__).resolve().parent
PROJ_DIR = SCRIPT_DIR.parent
CONFIG_PATH = PROJ_DIR / "config.yaml"

BANDS = {"delta": (1, 4), "theta": (4, 8), "alpha": (8, 13), "beta": (13, 30)}
# rough scalp regions by 10-20 label prefix
FRONTAL = ("Fp", "AF", "F")
POSTERIOR = ("P", "PO", "O")


def load_config(p):
    cfg = yaml.safe_load(open(p))
    d = cfg["data"]
    suffix = "_cluster" if Path("/projects/swglab").exists() else "_local"
    for key in ("features_dir", "eeg_preproc_dir", "confounds_dir"):
        d[key] = str(Path(d[key + suffix]).expanduser())
    return cfg


def load_rest_raw(cfg, sub):
    """Reload raw rest EEG runs (1-45 Hz, 250 Hz), returns list of (raw_obj, data)."""
    d = cfg["data"]; ses = d["session"]; eroot = Path(d["eeg_preproc_dir"])
    ec = d["eeg"]; lo, hi = ec["bandpass"]; desc = ec["desc"]; sf = ec["sfreq_hmm"]
    out = []
    for run in [1, 2]:
        fif = (eroot / f"sub-{sub}" / ses / "eeg" /
               f"sub-{sub}_{ses}_task-rest_run-{run:02d}_desc-{desc}_eeg.fif")
        if not fif.exists():
            continue
        raw = mne.io.read_raw_fif(str(fif), preload=True, verbose=False)
        raw.pick(mne.pick_types(raw.info, eeg=True, exclude=[]))
        raw.filter(lo, hi, verbose=False)
        raw.resample(sf, verbose=False)
        out.append((raw, raw.get_data().T))  # (n_samp, n_ch)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(CONFIG_PATH))
    ap.add_argument("--model", default="group_k12")
    ap.add_argument("--dmn_state", type=int, default=7)
    args = ap.parse_args()

    cfg = load_config(args.config)
    res_dir = PROJ_DIR / "results" / args.model
    fig_dir = res_dir / "figures"; fig_dir.mkdir(parents=True, exist_ok=True)

    npz = np.load(res_dir / "state_probabilities.npz", allow_pickle=True)
    alpha_all = npz["alpha"]
    labels = [tuple(x) for x in npz["subject_run_labels"]]
    K = int(npz["n_states"]) if "n_states" in npz else alpha_all[0].shape[1]

    # Reload raw per subject (cache), align to alpha length, collect lists
    raw_cache = {}
    data_list, alpha_list = [], []
    ref_raw = None
    for ai, (sub, run) in enumerate(labels):
        if sub not in raw_cache:
            raw_cache[sub] = load_rest_raw(cfg, sub)
        runs = raw_cache[sub]
        idx = int(run) - 1
        if idx >= len(runs):
            continue
        raw, data = runs[idx]
        if ref_raw is None:
            ref_raw = raw
        a = np.asarray(alpha_all[ai], dtype=np.float64)
        n = a.shape[0]
        # align: TDE trims symmetrically; take centered window of raw to match alpha
        off = max((data.shape[0] - n) // 2, 0)
        d = data[off:off + n]
        m = min(d.shape[0], n)
        data_list.append(d[:m].astype(np.float32))
        alpha_list.append(a[:m])

    print(f"Runs for spectra: {len(data_list)}")
    ch_names = ref_raw.ch_names
    sf = cfg["data"]["eeg"]["sfreq_hmm"]

    from osl_dynamics.analysis import spectral
    f, psd, coh = spectral.multitaper_spectra(
        data=data_list, alpha=alpha_list, sampling_frequency=sf,
        frequency_range=[1, 45], return_weights=False, n_jobs=4,
    )
    # psd shape (n_runs, K, n_ch, F) -> average over runs
    psd = np.asarray(psd)
    if psd.ndim == 4:
        psd_m = psd.mean(axis=0)
    else:
        psd_m = psd  # already (K, ch, F)
    print("psd_m shape:", psd_m.shape)

    np.savez_compressed(res_dir / "state_spectra.npz",
                        f=f, psd=psd_m, channels=np.array(ch_names))

    # ── Band power per state per channel; region summaries ──
    def band_mask(band):
        lo, hi = BANDS[band]
        return (f >= lo) & (f < hi)

    def region_idx(prefixes):
        return [i for i, c in enumerate(ch_names)
                if c.upper().startswith(tuple(p.upper() for p in prefixes))]

    front_i = region_idx(FRONTAL); post_i = region_idx(POSTERIOR)
    rows = []
    for k in range(K):
        for band in BANDS:
            bm = band_mask(band)
            bp = psd_m[k][:, bm].mean(axis=1)  # per channel
            rows.append(dict(state=k + 1, band=band,
                             frontal=bp[front_i].mean() if front_i else np.nan,
                             posterior=bp[post_i].mean() if post_i else np.nan,
                             whole=bp.mean()))
    bp_df = pd.DataFrame(rows)
    bp_df.to_csv(res_dir / "state_band_power.csv", index=False)

    # relative (state minus across-state mean) to expose signature
    print("\n=== Relative band power (state - mean across states) ===")
    for band in BANDS:
        sub = bp_df[bp_df.band == band].set_index("state")
        fr = sub["frontal"]; po = sub["posterior"]
        fr_rel = fr - fr.mean(); po_rel = po - po.mean()
        print(f"\n{band}:")
        print("  frontal Δ : " + " ".join(f"S{k}:{fr_rel[k]:+.2e}" for k in range(1, K + 1)))
        print("  posterior Δ: " + " ".join(f"S{k}:{po_rel[k]:+.2e}" for k in range(1, K + 1)))

    ds = args.dmn_state
    print(f"\n=== DMN state (State {ds}) spectral signature ===")
    for band in BANDS:
        sub = bp_df[(bp_df.band == band)].set_index("state")
        fr = sub["frontal"]; po = sub["posterior"]
        fr_rank = int((fr.rank(ascending=False))[ds]); po_rank = int((po.rank(ascending=False))[ds])
        print(f"  {band}: frontal rank {fr_rank}/{K}, posterior rank {po_rank}/{K}")
    print("(Paper's DMN signature: posterior ALPHA high + frontal DELTA/THETA high)")

    # ── Topomap of alpha power per state ──
    try:
        alpha_bm = band_mask("alpha")
        alpha_bp = np.array([psd_m[k][:, alpha_bm].mean(axis=1) for k in range(K)])  # (K, ch)
        info = ref_raw.info
        ncol = 4; nrow = int(np.ceil(K / ncol))
        fig, axes = plt.subplots(nrow, ncol, figsize=(3 * ncol, 3 * nrow))
        vmin, vmax = alpha_bp.min(), alpha_bp.max()
        for k in range(K):
            ax = axes.flat[k]
            mne.viz.plot_topomap(alpha_bp[k], info, axes=ax, show=False,
                                 vlim=(vmin, vmax), cmap="RdBu_r")
            ttl = f"State {k+1}" + ("  (DMN)" if (k + 1) == ds else "")
            ax.set_title(ttl, fontsize=10, fontweight="bold" if (k + 1) == ds else "normal")
        for j in range(K, nrow * ncol):
            axes.flat[j].axis("off")
        fig.suptitle("Alpha (8-13 Hz) power per HMM state", fontsize=13)
        fig.tight_layout()
        fig.savefig(fig_dir / "state_topomaps_alpha.png", dpi=140)
        plt.close(fig)
        print(f"\nSaved topomaps to {fig_dir / 'state_topomaps_alpha.png'}")
    except Exception as e:
        print("[topomap skipped]:", repr(e))

    # ── DMN-state spectrum by region ──
    try:
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(f, psd_m[ds - 1][front_i].mean(0), label="frontal", color="C3")
        ax.plot(f, psd_m[ds - 1][post_i].mean(0), label="posterior", color="C0")
        ax.plot(f, psd_m[ds - 1].mean(0), label="whole-head", color="k", ls="--", alpha=.6)
        for band, (lo, hi) in BANDS.items():
            ax.axvspan(lo, hi, alpha=0.06)
        ax.set_xlabel("Frequency (Hz)"); ax.set_ylabel("Power")
        ax.set_title(f"State {ds} (fMRI-DMN state) power spectrum by region")
        ax.legend(); fig.tight_layout()
        fig.savefig(fig_dir / "dmn_state_spectrum.png", dpi=140)
        plt.close(fig)
        print(f"Saved DMN-state spectrum to {fig_dir / 'dmn_state_spectrum.png'}")
    except Exception as e:
        print("[spectrum plot skipped]:", repr(e))

    print(f"\nSaved band power CSV to {res_dir / 'state_band_power.csv'}")


if __name__ == "__main__":
    main()
