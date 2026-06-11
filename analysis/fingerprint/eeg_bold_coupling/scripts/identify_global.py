"""
identify_global.py  (STEP A follow-up)
--------------------------------------
The robust EEG->DMN coupling is a GLOBAL broadband EEG-power factor. Is it a real
brain state (arousal/vigilance) or a motion/physiological artifact? This decides
whether it is a usable (if non-specific) DMN decoder or junk.

For each subject (feedback runs, per-run z-scored, pooled) we compute:
  G_raw  = mean over all band x channel RAW (pre-HRF) log power  -> instantaneous
  G_hrf  = same but HRF-convolved                                -> BOLD-aligned
  per-band G_hrf (which band drives the DMN coupling)
and from fMRIPrep confounds: FD (framewise displacement), global_signal, dvars;
from the npz: DMN, CEN, PDA, and GFMRI = mean of the 64 DiFuMo parcels (global BOLD).

Diagnostic correlations (+ partial correlations):
  eeg_motion      = corr(G_raw, FD)                 motion in EEG <-> fMRI motion
  eeg_globalBOLD  = corr(G_hrf, GFMRI)              global EEG power <-> global BOLD
  eeg_DMN         = corr(G_hrf, DMN)                the coupling to explain
  DMN_globalBOLD  = corr(DMN, GFMRI)                is the DMN ROI ~ global signal?
  DMN_motion      = corr(DMN, FD)
  eeg_DMN|globalBOLD, eeg_DMN|FD  = partial corr   does the coupling SURVIVE control?
  band profile    = corr(G_hrf_band, DMN) per band  arousal(alpha/delta) vs broadband

Group: one-sample t (Fisher-z) across subjects. Figure + CSV.

Usage: sbatch scripts/identify_job.sh   (band-power extraction is heavy; use sharing)
"""
import argparse, sys, warnings, csv
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import ttest_1samp

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bandpower import load_config, canonical_hrf, hrf_convolve, gather_subject, zscore  # noqa: E402
import mne  # noqa: E402
warnings.simplefilter("ignore"); mne.set_log_level("ERROR")

CONF = ("/projects/swglab/data/DMNELF/derivatives/fmriprep_25.2.5_fmap/"
        "sub-{s}/ses-dmnelf/func/sub-{s}_ses-dmnelf_task-feedback_run-{r:02d}_"
        "desc-confounds_timeseries.tsv")


def corr(a, b):
    a = (a - a.mean()) / (a.std() + 1e-12); b = (b - b.mean()) / (b.std() + 1e-12)
    return float((a * b).mean())


def partial(a, b, c):
    """corr(a,b | c): residualize a and b on [1,c], correlate residuals."""
    X = np.column_stack([np.ones_like(c), c])
    ra = a - X @ np.linalg.lstsq(X, a, rcond=None)[0]
    rb = b - X @ np.linalg.lstsq(X, b, rcond=None)[0]
    return corr(ra, rb)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    a = ap.parse_args(); cfg = load_config(a.config)
    bands = list(cfg["bands"]); hcfg = cfg["hrf"]
    hrf = canonical_hrf(cfg["data"]["fmri"]["tr"], hcfg["length_s"], hcfg["delay"], hcfg["undershoot"])
    ident = np.array([1.0])
    subs = [s for s in cfg["data"]["subjects"]["all"]
            if s not in set(cfg["data"]["subjects"].get("exclude", []))]

    rows = []
    band_prof = {b: [] for b in bands}
    for s in subs:
        runs = gather_subject(cfg, s, ident)            # RAW per-TR log band power
        if len(runs) < 2:
            print(f"  skip {s}"); continue
        Graw, Ghrf, DMN, CEN, PDA, GF, FD, GS, DV = ([] for _ in range(9))
        Gband = {b: [] for b in bands}
        ok = True
        for r in runs:
            cf_path = CONF.format(s=s, r=int(r["run"]))
            if not Path(cf_path).exists():
                ok = False; break
            cf = pd.read_csv(cf_path, sep="\t")
            n = r["n_tr"]
            fd = np.nan_to_num(cf["framewise_displacement"].values[:n], nan=0.0)
            gs = cf["global_signal"].values[:n]
            dv = np.nan_to_num(cf["dvars"].values[:n], nan=0.0)
            raw_all = np.hstack([r["bp"][b] for b in bands])            # (n,155) RAW
            hrf_all = np.column_stack([hrf_convolve(raw_all[:, j], hrf) for j in range(raw_all.shape[1])])
            Graw.append(zscore(raw_all.mean(1))); Ghrf.append(zscore(hrf_all.mean(1)))
            for b in bands:
                Gband[b].append(zscore(hrf_convolve(r["bp"][b].mean(1), hrf)))
            DMN.append(zscore(r["targets"]["DMN"])); CEN.append(zscore(r["targets"]["CEN"]))
            PDA.append(zscore(r["targets"]["PDA"])); GF.append(zscore(r["parcels"].mean(1)))
            FD.append(zscore(fd)); GS.append(zscore(gs)); DV.append(zscore(dv))
        if not ok:
            print(f"  skip {s} (missing confounds)"); continue
        cat = lambda L: np.concatenate(L)
        Graw, Ghrf = cat(Graw), cat(Ghrf)
        DMN, CEN, PDA, GF = cat(DMN), cat(CEN), cat(PDA), cat(GF)
        FD, GS, DV = cat(FD), cat(GS), cat(DV)
        row = dict(subject=s,
                   eeg_motion=corr(Graw, FD), eeg_dvars=corr(Graw, DV),
                   eeg_globalBOLD=corr(Ghrf, GF), eeg_globalsig=corr(Ghrf, GS),
                   eeg_DMN=corr(Ghrf, DMN), eeg_CEN=corr(Ghrf, CEN), eeg_PDA=corr(Ghrf, PDA),
                   DMN_globalBOLD=corr(DMN, GF), DMN_motion=corr(DMN, FD),
                   DMN_globalsig=corr(DMN, GS),
                   eeg_DMN_pGF=partial(Ghrf, DMN, GF),
                   eeg_DMN_pFD=partial(Ghrf, DMN, FD),
                   eeg_DMN_pGS=partial(Ghrf, DMN, GS))
        rows.append(row)
        for b in bands:
            band_prof[b].append(corr(cat(Gband[b]), DMN))
        print(f"{s}: eeg_DMN={row['eeg_DMN']:+.2f}  eeg_globalBOLD={row['eeg_globalBOLD']:+.2f}  "
              f"eeg_motion={row['eeg_motion']:+.2f}  DMN_globalBOLD={row['DMN_globalBOLD']:+.2f}  "
              f"eeg_DMN|GF={row['eeg_DMN_pGF']:+.2f}  eeg_DMN|FD={row['eeg_DMN_pFD']:+.2f}")

    df = pd.DataFrame(rows)
    keys = ["eeg_motion", "eeg_dvars", "eeg_globalBOLD", "eeg_globalsig", "eeg_DMN",
            "DMN_globalBOLD", "DMN_motion", "DMN_globalsig",
            "eeg_DMN_pGF", "eeg_DMN_pFD", "eeg_DMN_pGS"]
    print("\n=== COHORT (mean r, one-sample t on Fisher-z) ===")
    for k in keys:
        v = df[k].values; z = np.arctanh(np.clip(v, -.999, .999))
        t, p = ttest_1samp(z, 0)
        print(f"  {k:16s} mean r={v.mean():+.3f}  t={t:+.2f} p={p:.4f}")

    # ---- figure ----
    fig, ax = plt.subplots(1, 3, figsize=(18, 5))
    barkeys = ["eeg_motion", "eeg_globalBOLD", "eeg_globalsig", "eeg_DMN", "DMN_globalBOLD", "DMN_motion"]
    means = [df[k].mean() for k in barkeys]
    ax[0].bar(range(len(barkeys)), means, color="C0")
    for i, k in enumerate(barkeys):
        ax[0].scatter(np.full(len(df), i) + np.random.uniform(-.12, .12, len(df)),
                      df[k], s=12, color="k", alpha=.5)
    ax[0].set_xticks(range(len(barkeys))); ax[0].set_xticklabels(barkeys, rotation=30, ha="right", fontsize=8)
    ax[0].axhline(0, color="k", lw=.6); ax[0].set_ylabel("Pearson r"); ax[0].set_title("what does the global factor track?")

    pk = ["eeg_DMN", "eeg_DMN_pGF", "eeg_DMN_pFD", "eeg_DMN_pGS"]
    ax[1].bar(range(len(pk)), [df[k].mean() for k in pk], color=["C3", "C1", "C2", "C5"])
    for i, k in enumerate(pk):
        ax[1].scatter(np.full(len(df), i) + np.random.uniform(-.12, .12, len(df)), df[k], s=12, color="k", alpha=.5)
    ax[1].set_xticks(range(len(pk)))
    ax[1].set_xticklabels(["eeg→DMN", "| cortical GS", "| motion", "| whole-brain GS"], fontsize=8)
    ax[1].axhline(0, color="k", lw=.6); ax[1].set_ylabel("r"); ax[1].set_title("does eeg→DMN survive controls?")

    bm = [np.mean(band_prof[b]) for b in bands]
    ax[2].bar(range(len(bands)), bm, color="C4")
    ax[2].set_xticks(range(len(bands))); ax[2].set_xticklabels(bands)
    ax[2].axhline(0, color="k", lw=.6); ax[2].set_ylabel("mean corr(band global power, DMN)")
    ax[2].set_title("band profile of the DMN coupling\n(alpha/delta=arousal, broadband/gamma=motion)")
    fig.suptitle("Identify the global EEG→DMN factor: arousal vs motion/global-signal", fontsize=13)
    fig.tight_layout()
    outd = Path(cfg["project"]["base_dir"]) / "results"
    (outd / "figures").mkdir(parents=True, exist_ok=True)
    fig.savefig(outd / "figures" / "identify_global.png", dpi=130); plt.close(fig)
    df.to_csv(outd / "identify_global.csv", index=False)
    print(f"saved: {outd/'identify_global.csv'} + figures/identify_global.png")


if __name__ == "__main__":
    main()
