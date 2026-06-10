"""
coupling_group.py  (STEP A, group inference)
--------------------------------------------
Is there a CONSISTENT, spatially/spectrally specific EEG band-power -> network
coupling across the cohort, after removing the global broadband confound?

For each subject we compute a channel x band coupling map r[band,ch] between the
HRF-convolved log band power and each target (DMN/CEN/PDA), under three feature
versions: RAW, CAR (global spatial factor removed), PARTIAL (global+trend
regressed out). We then do GROUP inference across subjects:
  - mean coupling map,
  - one-sample t across subjects per cell,
  - FWER-controlled p via SIGN-FLIP max-statistic null (flip each subject's whole
    map by +-1, recompute max|t| over the grid; corrects for the 31x5 search).

If a consistent coupling exists (e.g. posterior alpha -> DMN), the group t-map has
a significant, interpretable peak. If the per-subject hits are scattered, the
group map is ~0 and nothing survives -> the residual couplings were noise.

Usage: python scripts/coupling_group.py --config config.yaml
Output: results/coupling_group.csv + results/figures/coupling_group_<version>.png
"""
import argparse, sys, warnings, csv
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bandpower import load_config, canonical_hrf, gather_subject, zscore  # noqa: E402
import mne  # noqa: E402
warnings.simplefilter("ignore"); mne.set_log_level("ERROR")


def resid(Y, nuis):
    X = np.column_stack([np.ones(len(nuis)), nuis])
    return Y - X @ np.linalg.lstsq(X, Y, rcond=None)[0]


def corr_map(F, y):
    """Pearson r of each column of F with y (both will be z-scored)."""
    Fz = (F - F.mean(0)) / (F.std(0) + 1e-12)
    yz = (y - y.mean()) / (y.std() + 1e-12)
    return (Fz * yz[:, None]).mean(0)


def subject_maps(runs, bands):
    """Return {version: {target: r[nb,nc]}} for one subject."""
    nb, nc = len(bands), len(runs[0]["chs"])
    raw_r, car_r, par_r = [], [], []
    tgt = {k: [] for k in ("DMN", "CEN", "PDA")}
    tgt_p = {k: [] for k in ("DMN", "CEN", "PDA")}
    for r in runs:
        zb = {b: zscore(r["bp"][b]) for b in bands}
        raw = np.hstack([zb[b] for b in bands])
        car = np.hstack([zb[b] - zb[b].mean(1, keepdims=True) for b in bands])
        g = raw.mean(1); t = np.arange(r["n_tr"])
        raw_r.append(raw); car_r.append(car)
        par_r.append(resid(raw, np.column_stack([t, g])))
        for k in tgt:
            yz = zscore(r["targets"][k])
            tgt[k].append(yz); tgt_p[k].append(resid(yz, np.column_stack([t, g])))
    F = {"RAW": np.vstack(raw_r), "CAR": np.vstack(car_r), "PARTIAL": np.vstack(par_r)}
    Y = {k: np.concatenate(tgt[k]) for k in tgt}
    Yp = {k: np.concatenate(tgt_p[k]) for k in tgt_p}
    out = {}
    for ver in F:
        out[ver] = {}
        for k in ("DMN", "CEN", "PDA"):
            yy = Yp[k] if ver == "PARTIAL" else Y[k]
            out[ver][k] = corr_map(F[ver], yy).reshape(nb, nc)
    return out


def signflip_maxt(R, nperm, rng):
    """R: [ns, nb, nc] per-subject r maps. Returns mean, t, p_fwer per cell + best."""
    ns = R.shape[0]
    mean = R.mean(0); sd = R.std(0, ddof=1)
    t = mean / (sd / np.sqrt(ns) + 1e-12)
    obs = np.abs(t); obs_max = obs.max()
    bidx = np.unravel_index(obs.argmax(), obs.shape)
    ge = np.zeros_like(t)
    cnt_max = 0
    for _ in range(nperm):
        s = rng.choice([-1, 1], size=ns)[:, None, None]
        Rp = R * s
        mp = Rp.mean(0); sdp = Rp.std(0, ddof=1)
        tp = np.abs(mp / (sdp / np.sqrt(ns) + 1e-12))
        if tp.max() >= obs_max:
            cnt_max += 1
        ge += (tp >= obs)
    p_best = (cnt_max + 1) / (nperm + 1)
    return mean, t, bidx, p_best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--nperm", type=int, default=5000)
    a = ap.parse_args(); cfg = load_config(a.config)
    bands = list(cfg["bands"]); nb = len(bands)
    hcfg = cfg["hrf"]; hrf = canonical_hrf(cfg["data"]["fmri"]["tr"], hcfg["length_s"],
                                           hcfg["delay"], hcfg["undershoot"])
    subs = [s for s in cfg["data"]["subjects"]["all"]
            if s not in set(cfg["data"]["subjects"].get("exclude", []))]
    rng = np.random.default_rng(0)

    maps = {v: {k: [] for k in ("DMN", "CEN", "PDA")} for v in ("RAW", "CAR", "PARTIAL")}
    chs = None
    for s in subs:
        runs = gather_subject(cfg, s, hrf)
        if len(runs) < 2:
            print(f"  skip {s} (<2 runs)"); continue
        chs = runs[0]["chs"]
        sm = subject_maps(runs, bands)
        for v in maps:
            for k in maps[v]:
                maps[v][k].append(sm[v][k])
        print(f"  {s} done")
    nc = len(chs)

    rows = []
    for ver in ("RAW", "CAR", "PARTIAL"):
        fig, ax = plt.subplots(2, 3, figsize=(16, 7))
        for ci, k in enumerate(("DMN", "CEN", "PDA")):
            R = np.stack(maps[ver][k])                     # [ns, nb, nc]
            mean, t, bidx, p_best = signflip_maxt(R, a.nperm, rng)
            bi, bj = bidx
            print(f"{ver:7s} {k}: group-best {chs[bj]}/{bands[bi]} "
                  f"mean_r={mean[bi,bj]:+.3f} t={t[bi,bj]:+.2f} p_fwer={p_best:.4f}")
            rows.append(dict(version=ver, target=k, best_ch=chs[bj], best_band=bands[bi],
                             mean_r=float(mean[bi, bj]), t=float(t[bi, bj]), p_fwer=p_best))
            im0 = ax[0, ci].imshow(mean, aspect="auto", cmap="RdBu_r", vmin=-.2, vmax=.2)
            ax[0, ci].set_title(f"{k}: group mean r", fontsize=9)
            im1 = ax[1, ci].imshow(t, aspect="auto", cmap="RdBu_r", vmin=-5, vmax=5)
            ax[1, ci].set_title(f"{k}: group t  (best {chs[bj]}/{bands[bi]} "
                                f"r={mean[bi,bj]:+.2f} p_fwer={p_best:.3f})", fontsize=8)
            for axi, im in ((ax[0, ci], im0), (ax[1, ci], im1)):
                axi.set_yticks(range(nb)); axi.set_yticklabels(bands, fontsize=7)
                axi.set_xticks(range(0, nc, 4))
                axi.set_xticklabels([chs[j] for j in range(0, nc, 4)], rotation=90, fontsize=5)
                fig.colorbar(im, ax=axi, fraction=.046, pad=.02)
        fig.suptitle(f"Group EEG band-power -> network coupling ({ver}, n={len(subs)}, "
                     f"sign-flip max-stat null)", fontsize=12)
        fig.tight_layout()
        outd = Path(cfg["project"]["base_dir"]) / "results" / "figures"
        outd.mkdir(parents=True, exist_ok=True)
        fig.savefig(outd / f"coupling_group_{ver}.png", dpi=110); plt.close(fig)

    outp = Path(cfg["project"]["base_dir"]) / "results" / "coupling_group.csv"
    with open(outp, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0].keys())); wr.writeheader(); wr.writerows(rows)
    print(f"saved: {outp} + figures/coupling_group_*.png")


if __name__ == "__main__":
    main()
