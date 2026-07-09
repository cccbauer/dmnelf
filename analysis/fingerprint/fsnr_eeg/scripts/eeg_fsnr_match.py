#!/usr/bin/env python3
"""
eeg_fsnr_match.py  —  Stream B, Flavor 2 (matching; runs local after extraction)
--------------------------------------------------------------------------------
Uses the specparam features (eeg_fsnr_specparam.py output) to build an oscillatory/
aperiodic EEG f-SNR and match it to the fMRI PDA and fMRI running f-SNR. Also computes the
CLEAN (non-convolved) EEG variability quench — the cross-modal analog of the BOLD result.

EEG f-SNR candidates per channel (specparam):
  periodic        oscillatory power above the 1/f (signal-above-noise, intrinsic f-SNR)
  alpha           oscillatory power in 8-13 Hz
  osc_over_ap     periodic / offset (signal / aperiodic-noise)
  neg_exponent    -aperiodic exponent (flatter 1/f)
  bandpow[b]      non-convolved band power (5)
EEG features are HRF-convolved before matching to BOLD (instantaneous EEG -> BOLD time).
Matching is leak-free nested single-site correlation (same as Flavor 1).
"""
from pathlib import Path
import numpy as np, glob, re
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "fsnr" / "scripts"))
from fsnr_fmri import canonical_hrf, BASELINE_TR, HRF_DROP
from fsnr_proxy import running_fsnr
sys.path.insert(0, str(Path(__file__).resolve().parent))
from eeg_fsnr_bandpower import zs, folds, match_nested, subject_runs as bp_runs

PROJ = Path(__file__).resolve().parent.parent
SPEC = PROJ / "results" / "specparam"
BPDATA = PROJ / "data"
RES = PROJ / "results"
HRF = canonical_hrf()
BANDS = ["delta", "theta", "alpha", "beta", "gamma"]
POST = ["P3", "P4", "P7", "P8", "O1", "O2", "Oz", "Pz", "POz", "PO3", "PO4"]


def hrfconv(x):
    y = np.convolve(np.nan_to_num(x), HRF, mode="full")[:len(x)]
    return y


def pda_of(sub):
    """PDA(t) per run from the band-power cache (aligned)."""
    runs, chs = bp_runs(str(BPDATA / f"{sub}_bandpower.npz"))
    return {rd["run"]: np.asarray(rd["targets"]["PDA"], float) for rd in runs}


def sflip(x, n=10000):
    x = np.asarray([v for v in x if np.isfinite(v)]); rng = np.random.default_rng(0)
    obs = x.mean(); null = (rng.choice([-1, 1], (n, len(x))) * np.abs(x)).mean(1)
    return obs, (np.sum(null >= obs) + 1) / (n + 1)


def main():
    files = sorted(glob.glob(str(SPEC / "*_specparam.npz")))
    if not files:
        print("no specparam output yet — run/await the extraction job."); return
    rows, quench = [], {b: [] for b in BANDS}
    apriori = []
    for f in files:
        sub = re.search(r"(dmnelf\w+)_specparam", f).group(1)
        z = np.load(f, allow_pickle=True)
        runkeys = [str(k) for k in z["_runs"]]
        pda = pda_of(sub)
        Xosc, Xap, ypda, yfs, feat = [], [], [], [], None
        for rk in runkeys:
            d = z[rk].item(); chs = list(d["chs"]); rn = int(rk.replace("run", ""))
            if rn not in pda:
                continue
            p = pda[rn]; n = min(len(p), d["offset"].shape[0])
            per, al, off, exn = d["periodic"][:n], d["alpha"][:n], d["offset"][:n], d["exponent"][:n]
            bp = d["bandpow"][:n]
            # per channel EEG f-SNR feature bank (HRF-convolved)
            cols, names = [], []
            for ci in range(len(chs)):
                for arr, nm in [(per[:, ci], "periodic"), (al[:, ci], "alpha"),
                                (per[:, ci] / (off[:, ci] + 1e-9), "osc_over_ap"),
                                (-exn[:, ci], "neg_exp")]:
                    cols.append(hrfconv(arr)); names.append((nm, chs[ci]))
                for bi, b in enumerate(BANDS):
                    cols.append(hrfconv(bp[:, ci, bi])); names.append((b, chs[ci]))
            X = np.column_stack(cols)
            Xosc.append(X); ypda.append(zs(p[:n])); yfs.append(zs(running_fsnr(p[:n])[1]))
            if feat is None: feat = names
            # clean EEG quench (non-convolved band power, posterior)
            pi = [i for i, c in enumerate(chs) if c in POST]
            for bi, b in enumerate(BANDS):
                vr = bp[:BASELINE_TR][:, pi, bi]; vf = bp[BASELINE_TR+HRF_DROP:][:, pi, bi]
                vr, vf = np.nanvar(vr, 0).mean(), np.nanvar(vf, 0).mean()
                quench[b].append(10*np.log10(vr/vf) if vf > 1e-12 and vr > 0 else np.nan)
        if not Xosc:
            continue
        X = np.vstack(Xosc); ypda = np.concatenate(ypda); yfs = np.concatenate(yfs)
        r = dict(subject=sub)
        r["eegfsnr_vs_PDA"], ch = match_nested(X, ypda)
        r["eegfsnr_vs_fMRIfsnr"], _ = match_nested(X, yfs)
        # which feature type chosen most for PDA
        if ch:
            from collections import Counter
            top = Counter(feat[j][0] for j in ch).most_common(1)[0][0]
            r["top_feature"] = top
        rows.append(r)
        # a-priori: posterior alpha oscillatory power (HRF-conv) vs PDA, no fitting
        ap = []
        for rk in runkeys:
            d = z[rk].item(); chs = list(d["chs"]); rn = int(rk.replace("run", ""))
            if rn not in pda: continue
            pi = [i for i, c in enumerate(chs) if c in POST]
            p = pda[rn]; n = min(len(p), d["alpha"].shape[0])
            aeeg = zs(hrfconv(np.nanmean(d["alpha"][:n][:, pi], 1)))
            pp = zs(p[:n]); m = np.isfinite(aeeg) & np.isfinite(pp)
            if m.sum() > 20: ap.append(np.corrcoef(aeeg[m], pp[m])[0, 1])
        apriori.append(np.nanmean(ap) if ap else np.nan)

    import pandas as pd
    df = pd.DataFrame(rows); df.to_csv(RES / "eeg_fsnr_specparam_match.csv", index=False)
    print(f"{len(df)} subjects\n")
    print("=== Flavor 2 (specparam) within-subject matched r ===")
    for c in ["eegfsnr_vs_PDA", "eegfsnr_vs_fMRIfsnr"]:
        o, p = sflip(df[c].values); print(f"  {c:22s} r={o:+.3f}  p={p:.4f}")
    o, p = sflip(np.array(apriori))
    print(f"  {'aprioriPostAlphaOsc_PDA':22s} r={o:+.3f}  p={p:.4f}  (construct, no fitting)")
    if "top_feature" in df:
        print("  top feature chosen:", df["top_feature"].value_counts().to_dict())
    print("\n=== CLEAN EEG variability quench (non-convolved; posterior; +dB=declutter) ===")
    for b in BANDS:
        o, p = sflip(quench[b]); print(f"  {b:6s} {o:+.2f} dB  p={p:.4f}")


if __name__ == "__main__":
    main()
