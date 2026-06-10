"""
coupling_specificity.py  (STEP A, control)
-------------------------------------------
The raw band-power coupling screen showed a UNIFORM band x channel heatmap
(~+0.69 everywhere for DMN) = a single GLOBAL broadband power factor, confirmed
by corr(global EEG power, DMN)=+0.68 (008), which survives detrending. This is an
arousal/motion-type confound, not spatially-specific neural coupling.

This script tests whether ANY specific coupling survives removing that global
factor, three feature versions compared side by side:
  RAW     : per-run z-scored log band power (HRF-convolved).
  CAR     : common-average-reference on the power -> per band, subtract the
            across-channel mean per TR. Removes the global spatial factor, keeps
            topographic contrast (e.g. posterior vs frontal alpha).
  PARTIAL : residualize each feature AND each target against [1, t, global] per
            run (global = mean across all features). Removes the shared global +
            linear trend.

For each target (DMN/CEN/PDA) x version: best channel x band coupling with a
max-stat circular-shift null. If specific coupling survives CAR/PARTIAL, it is
real; if it collapses to ~0, the whole effect was the global confound.

Usage: python scripts/coupling_specificity.py --config config.yaml --subjects dmnelf008 dmnelf007
"""
import argparse, sys, warnings
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bandpower import load_config, canonical_hrf, gather_subject, zscore  # noqa: E402
import mne  # noqa: E402
warnings.simplefilter("ignore"); mne.set_log_level("ERROR")


def zc(x):
    return (x - x.mean(0)) / (x.std(0) + 1e-12)


def resid(Y, nuis):
    """Remove [1, nuis...] from Y (1D or 2D columns), least squares."""
    X = np.column_stack([np.ones(len(nuis)), nuis])
    beta = np.linalg.lstsq(X, Y, rcond=None)[0]
    return Y - X @ beta


def maxstat(F, y, run_lens, nperm, rng):
    Fz = zc(F); yz = zc(y)
    r = (Fz * yz[:, None]).mean(0)
    obs = np.abs(r); b = int(obs.argmax()); om = obs[b]; ge = 0
    for _ in range(nperm):
        sh, off = [], 0
        for L in run_lens:
            seg = y[off:off + L]; k = int(rng.integers(5, max(6, L - 5)))
            sh.append(np.roll(seg, k)); off += L
        if np.abs((Fz * zc(np.concatenate(sh))[:, None]).mean(0)).max() >= om:
            ge += 1
    return r, b, (ge + 1) / (nperm + 1)


def build_versions(runs, bands):
    """Return dict version-> (F[T,J]), Jlab, targets dict, run_lens, chs, global_r."""
    chs = runs[0]["chs"]; nc = len(chs)
    Jlab = [(c, b) for b in bands for c in chs]
    raw_runs, car_runs, glob_runs, tgt_runs = [], [], [], {k: [] for k in ("DMN", "CEN", "PDA")}
    lens = []
    for r in runs:
        zb = {b: zscore(r["bp"][b]) for b in bands}             # per band (n_tr,nc) z
        raw = np.hstack([zb[b] for b in bands])                 # (n_tr, nb*nc)
        car = np.hstack([zb[b] - zb[b].mean(1, keepdims=True) for b in bands])
        g = raw.mean(1)                                         # global factor
        t = np.arange(r["n_tr"])
        raw_runs.append(raw); car_runs.append(car)
        glob_runs.append(resid(raw, np.column_stack([t, g])))   # partial out trend+global
        lens.append(r["n_tr"])
        for k in tgt_runs:
            tgt_runs[k].append((zscore(r["targets"][k]), t, g))
    # assemble targets (raw z) and partial targets
    tgt = {k: np.concatenate([v[0] for v in tgt_runs[k]]) for k in tgt_runs}
    tgt_partial = {}
    for k in tgt_runs:
        segs = []
        for (yz, t, g) in tgt_runs[k]:
            segs.append(resid(yz, np.column_stack([t, g])))
        tgt_partial[k] = np.concatenate(segs)
    versions = {"RAW": np.vstack(raw_runs), "CAR": np.vstack(car_runs),
                "PARTIAL": np.vstack(glob_runs)}
    # global coupling r per target (reference)
    gall = np.concatenate([gr.mean(1) for gr in [v for v in raw_runs]])  # not used; placeholder
    return versions, Jlab, tgt, tgt_partial, lens, chs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--subjects", nargs="+", default=["dmnelf008", "dmnelf007"])
    a = ap.parse_args(); cfg = load_config(a.config)
    bands = list(cfg["bands"]); nb = len(bands)
    hcfg = cfg["hrf"]; hrf = canonical_hrf(cfg["data"]["fmri"]["tr"], hcfg["length_s"],
                                           hcfg["delay"], hcfg["undershoot"])
    nperm = cfg["stats"]["nperm"]; rng = np.random.default_rng(0)
    outd = Path(cfg["project"]["base_dir"]) / "results" / "figures"
    outd.mkdir(parents=True, exist_ok=True)

    for s in a.subjects:
        runs = gather_subject(cfg, s, hrf)
        if len(runs) < 2:
            print(f"{s}: <2 runs"); continue
        versions, Jlab, tgt, tgt_part, lens, chs = build_versions(runs, bands)
        nc = len(chs)
        print(f"\n=== {s} ===")
        fig, ax = plt.subplots(3, 3, figsize=(17, 11))
        for ri, k in enumerate(("DMN", "CEN", "PDA")):
            for ci, ver in enumerate(("RAW", "CAR", "PARTIAL")):
                y = tgt_part[k] if ver == "PARTIAL" else tgt[k]
                r, b, p = maxstat(versions[ver], y, lens, nperm, rng)
                ch, bd = Jlab[b]
                M = r.reshape(nb, nc)
                im = ax[ri, ci].imshow(M, aspect="auto", cmap="RdBu_r", vmin=-.3, vmax=.3)
                ax[ri, ci].set_yticks(range(nb)); ax[ri, ci].set_yticklabels(bands, fontsize=7)
                ax[ri, ci].set_xticks(range(0, nc, 4))
                ax[ri, ci].set_xticklabels([chs[j] for j in range(0, nc, 4)], rotation=90, fontsize=5)
                ax[ri, ci].set_title(f"{k} | {ver}: best {ch}/{bd} r={r[b]:+.2f} p={p:.3f}", fontsize=9)
                fig.colorbar(im, ax=ax[ri, ci], fraction=.046, pad=.02)
                print(f"  {k:3s} {ver:7s}: best {ch}/{bd} r={r[b]:+.3f} p={p:.3f}")
        fig.suptitle(f"{s}: specific coupling after removing the global power factor "
                     f"(RAW vs CAR vs PARTIAL)", fontsize=13)
        fig.tight_layout()
        fig.savefig(outd / f"specificity_{s}.png", dpi=130); plt.close(fig)
        print(f"  wrote figures/specificity_{s}.png")


if __name__ == "__main__":
    main()
