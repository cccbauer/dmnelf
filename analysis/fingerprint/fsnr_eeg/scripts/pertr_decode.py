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
from within_fb_decode import loro, sflip, zs, BASELINE_TR, HRF_DROP, TARGETS, BANDS, ALPHAS
from sklearn.linear_model import RidgeCV

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


def loso_group(pairs, kind, target):
    """General/group decoder: leave-one-SUBJECT-out. Train on all others (concat runs),
    predict the held-out subject with NO per-subject calibration. Returns per-subject r."""
    data = [build(b, p, kind, target) for b, p in pairs]
    data = [(np.vstack(Xs), np.concatenate(ys)) for Xs, ys in data if Xs]
    subj_r = []
    for i in range(len(data)):
        Xte, yte = data[i]
        Xtr = np.vstack([data[j][0] for j in range(len(data)) if j != i])
        ytr = np.concatenate([data[j][1] for j in range(len(data)) if j != i])
        mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-12
        m = RidgeCV(alphas=ALPHAS).fit((Xtr - mu) / sd, ytr)
        p = m.predict((Xte - mu) / sd)
        if np.std(p) > 1e-9:
            subj_r.append(np.corrcoef(yte, p)[0, 1])
    return np.array(subj_r)


def get_pairs(cohort):
    bp_glob = str(DATA / "*_bandpower.npz") if cohort == "dmnelf" \
        else str(DATA / "rtbpd_nf1" / "*_bandpower.npz")
    bps = {re.search(rf"({cohort}\w+)_bandpower", f).group(1): f
           for f in glob.glob(bp_glob) if not (cohort == "dmnelf" and QA.search(f))}
    subs = sorted(s for s in bps if (PERTR / f"{s}_pertr.npz").exists())
    return [(bps[s], str(PERTR / f"{s}_pertr.npz")) for s in subs]


def cross_cohort(train_pairs, test_pairs, kind, target):
    """TRANSFER: one model fit on ALL train-cohort subjects, applied to each test subject
    with NO refitting. Returns per-test-subject r."""
    Xtr = np.vstack([x for b, p in train_pairs for x in build(b, p, kind, target)[0]])
    ytr = np.concatenate([y for b, p in train_pairs for y in build(b, p, kind, target)[1]])
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-12
    m = RidgeCV(alphas=ALPHAS).fit((Xtr - mu) / sd, ytr)
    r = []
    for b, p in test_pairs:
        Xs, ys = build(b, p, kind, target)
        if not Xs:
            continue
        Xte, yte = np.vstack(Xs), np.concatenate(ys)
        pr = m.predict((Xte - mu) / sd)
        if np.std(pr) > 1e-9:
            r.append(np.corrcoef(yte, pr)[0, 1])
    return np.array(r)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohort", default="dmnelf", choices=["dmnelf", "rtbpd"])
    ap.add_argument("--transfer", action="store_true",
                    help="train general decoder on ALL DMNELF, test on rtBPD (cross-cohort)")
    ap.add_argument("--null", type=int, default=0, help="circular-shift null iterations (0=skip)")
    a = ap.parse_args()

    if a.transfer:
        tr, te = get_pairs("dmnelf"), get_pairs("rtbpd")
        print(f"CROSS-COHORT transfer: train on {len(tr)} DMNELF (schizophrenia), "
              f"test on {len(te)} rtBPD (borderline traits) — NO refitting\n")
        print(f"{'feature':16s} " + "  ".join(f"{t:>13s}" for t in TARGETS))
        for kind in ["bandpower", "bandpower+pertr"]:
            cells = []
            for tg in TARGETS:
                r = cross_cohort(tr, te, kind, tg); o, pv, _ = sflip(r)
                cells.append(f"{o:+.3f}(p{pv:.2f})")
            print(f"{kind:16s} " + "  ".join(f"{c:>13s}" for c in cells))
        return

    subs_pairs = get_pairs(a.cohort)
    print(f"{a.cohort} ({'nf1' if a.cohort=='rtbpd' else 'discovery'}): "
          f"{len(subs_pairs)} subjects with both band-power + per-TR f-SNR\n")
    pairs = subs_pairs

    print("=== within-feedback LORO decoding (per target) ===")
    print(f"{'feature':16s} " + "  ".join(f"{t:>13s}" for t in TARGETS))
    for kind in ["bandpower", "pertr_temporal", "pertr_spec2400", "pertr_spec3600", "bandpower+pertr"]:
        cells = []
        for tg in TARGETS:
            r = np.array([loro(*build(b, p, kind, tg)) for b, p in pairs])
            o, pv, _ = sflip(r); cells.append(f"{o:+.3f}(p{pv:.2f})")
        print(f"{kind:16s} " + "  ".join(f"{c:>13s}" for c in cells))

    print("\n=== GENERAL/group decoder: leave-one-SUBJECT-out (no per-subject calibration) ===")
    print(f"{'feature':16s} " + "  ".join(f"{t:>13s}" for t in TARGETS))
    for kind in ["bandpower", "bandpower+pertr"]:
        cells = []
        for tg in TARGETS:
            r = loso_group(pairs, kind, tg); o, pv, _ = sflip(r)
            cells.append(f"{o:+.3f}(p{pv:.2f})")
        print(f"{kind:16s} " + "  ".join(f"{c:>13s}" for c in cells))

    print("\n=== zero-fitting per-TR f-SNR constructs (frontal), in-block ===")
    for tg in TARGETS:
        oa = np.array([construct(b, p, tg)[0] for b, p in pairs])
        th = np.array([construct(b, p, tg)[1] for b, p in pairs])
        oo, op, _ = sflip(oa); to, tp, _ = sflip(th)
        print(f"  {tg}: osc/aperiodic r={oo:+.3f}(p{op:.2f})   theta-tSNR r={to:+.3f}(p{tp:.2f})")

    if a.null <= 0:
        return
    print("\n=== circular-shift null (best feature set, in-block LORO) ===")
    best = "bandpower+pertr"
    for tg in TARGETS:
        obs = np.nanmean([loro(*build(b, p, best, tg)) for b, p in pairs])
        nulls = []
        for _ in range(a.null):
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
