#!/usr/bin/env python3
"""
pertr_decode.py  —  does the per-TR f-SNR decode the target WITHIN the feedback block?
--------------------------------------------------------------------------------------
Phase C: add the instantaneous per-TR f-SNR features (temporal @1.2s, spectral @2.4/3.6s)
to the leak-free within-feedback decoder and compare against plain band power.

Targets (PDA/CEN/DMN) come from the band-power cache; per-TR f-SNR from results/pertr_fsnr/.
Matched per subject x run. Feedback block only (TR 30..end), target z-scored per run.
Evaluations: LORO (within subject) + LOSO, per target, vs circular-shift null.

Feature families:
  bandpower        31x5 HRF-convolved band power (Phase-B baseline)
  pertr_temporal   tsnr[31x5]  (within-1.2s envelope mean/std per band)
  pertr_spec2400   periodic/exponent/offset (31 each) + log bandpow (31x5)  [2.4s window]
  pertr_spec3600   same, 3.6s window
  bandpower+pertr  band power + temporal + spec3600 (does per-TR f-SNR ADD to band power?)
Construct (no fitting): frontal/posterior oscillatory-aperiodic f-SNR = periodic/offset, and
  temporal theta tsnr, correlated with each target in-block.
"""
from pathlib import Path
import numpy as np, glob, re, sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from within_fb_decode import loro, sflip, zs, BASELINE_TR, HRF_DROP, TARGETS, BANDS

DATA = Path(__file__).resolve().parents[1] / "data"
PERTR = Path(__file__).resolve().parents[1] / "results" / "pertr_fsnr"
QA = re.compile(r"dmnelf(999|1\d\d\d)")
FRONTAL = ["Fp1", "Fp2", "F3", "F4", "F7", "F8", "Fz", "FC1", "FC2", "FC5", "FC6"]
RNG = np.random.default_rng(0)


def pertr_feats(pr, kind):
    n = pr["n_tr"]
    if kind == "pertr_temporal":
        return pr["tsnr"].reshape(n, -1)                                  # [n,155]
    w = pr[kind.split("_")[1].replace("spec", "w")]                       # w2400 / w3600
    return np.column_stack([w["periodic"], w["exponent"], w["offset"],
                            w["bandpow"].reshape(n, -1)])                 # [n, 31*3+155]


def build(bp_npz, pertr_npz, kind, target):
    z = np.load(bp_npz, allow_pickle=True)
    pt = np.load(pertr_npz, allow_pickle=True)
    runmap = {int(r.replace("run", "")): pt[r].item() for r in pt["_runs"]}
    Xs, ys = [], []
    for rd in z["runs_data"]:
        run = int(rd["run"])
        if run not in runmap:
            continue
        n = rd["n_tr"]; sl = slice(BASELINE_TR + HRF_DROP, n)
        nch = rd["bp"]["theta"].shape[1]
        parts = []
        if kind in ("bandpower", "bandpower+pertr"):
            parts.append(np.column_stack([rd["bp"][b][:, c] for b in BANDS for c in range(nch)]))
        if kind in ("pertr_temporal", "bandpower+pertr"):
            parts.append(pertr_feats(runmap[run], "pertr_temporal"))
        if kind in ("pertr_spec2400", "pertr_spec3600"):
            parts.append(pertr_feats(runmap[run], kind))
        if kind == "bandpower+pertr":
            parts.append(pertr_feats(runmap[run], "pertr_spec3600"))
        X = np.column_stack(parts)[sl]
        y = zs(np.asarray(rd["targets"][target], float)[sl])
        ok = np.all(np.isfinite(X), 1) & np.isfinite(y)
        if ok.sum() > 20:
            Xs.append(X[ok]); ys.append(y[ok])
    return Xs, ys


def construct(bp_npz, pertr_npz, target):
    """Zero-fitting per-TR f-SNR constructs correlated with target in-block."""
    z = np.load(bp_npz, allow_pickle=True); pt = np.load(pertr_npz, allow_pickle=True)
    chs = [str(c) for c in z["ch_names"]]; fi = [i for i, c in enumerate(chs) if c in FRONTAL]
    runmap = {int(r.replace("run", "")): pt[r].item() for r in pt["_runs"]}
    oa, th = [], []
    for rd in z["runs_data"]:
        run = int(rd["run"])
        if run not in runmap:
            continue
        pr = runmap[run]; n = rd["n_tr"]; sl = slice(BASELINE_TR + HRF_DROP, n)
        y = zs(np.asarray(rd["targets"][target], float)[sl])
        w = pr["w3600"]; oaf = (w["periodic"] / (np.abs(w["offset"]) + 1e-6))[:, fi].mean(1)
        thf = pr["tsnr"][:, fi, 1].mean(1)      # theta temporal SNR, frontal
        for feat, acc in [(oaf, oa), (thf, th)]:
            f = zs(feat[sl]); m = np.isfinite(f) & np.isfinite(y)
            if m.sum() > 20:
                acc.append(np.corrcoef(f[m], y[m])[0, 1])
    return (np.nanmean(oa) if oa else np.nan), (np.nanmean(th) if th else np.nan)


def main():
    bps = {re.search(r"(dmnelf\w+)_bandpower", f).group(1): f
           for f in glob.glob(str(DATA / "*_bandpower.npz")) if not QA.search(f)}
    subs = sorted(s for s in bps if (PERTR / f"{s}_pertr.npz").exists())
    print(f"DMNELF: {len(subs)} subjects with both band-power + per-TR f-SNR\n")
    pairs = [(bps[s], str(PERTR / f"{s}_pertr.npz")) for s in subs]

    print("=== within-feedback LORO decoding (per target) ===")
    print(f"{'feature':16s} " + "  ".join(f"{t:>13s}" for t in TARGETS))
    for kind in ["bandpower", "pertr_temporal", "pertr_spec2400", "pertr_spec3600", "bandpower+pertr"]:
        cells = []
        for tg in TARGETS:
            r = np.array([loro(*build(b, p, kind, tg)) for b, p in pairs])
            o, pv, _ = sflip(r); cells.append(f"{o:+.3f}(p{pv:.2f})")
        print(f"{kind:16s} " + "  ".join(f"{c:>13s}" for c in cells))

    print("\n=== zero-fitting per-TR f-SNR constructs (frontal), in-block ===")
    for tg in TARGETS:
        oa = np.array([construct(b, p, tg)[0] for b, p in pairs])
        th = np.array([construct(b, p, tg)[1] for b, p in pairs])
        oo, op, _ = sflip(oa); to, tp, _ = sflip(th)
        print(f"  {tg}: osc/aperiodic r={oo:+.3f}(p{op:.2f})   theta-tSNR r={to:+.3f}(p{tp:.2f})")

    print("\n=== circular-shift null (best feature set, in-block LORO) ===")
    best = "bandpower+pertr"
    for tg in TARGETS:
        obs = np.nanmean([loro(*build(b, p, best, tg)) for b, p in pairs])
        nulls = []
        for _ in range(100):
            rs = []
            for b, p in pairs:
                Xs, ys = build(b, p, best, tg)
                ys = [np.roll(y, RNG.integers(5, len(y) - 5)) for y in ys]
                rs.append(loro(Xs, ys))
            nulls.append(np.nanmean(rs))
        nulls = np.array(nulls)
        print(f"  {tg}: obs={obs:+.3f}  null={nulls.mean():+.3f}±{nulls.std():.3f}  p={(nulls>=obs).mean():.3f}")


if __name__ == "__main__":
    main()
