#!/usr/bin/env python3
"""
montage_pz_ablation.py  (cluster, SLURM only)
---------------------------------------------
Does the CEN-DMN contrast (PDA) fail on the portable headset because the contrast is undecodable,
or because the headset has no centro-parietal midline coverage?

Motivation. In the research-cap single-electrode analysis, PDA is the BEST-transferring target
(LOSO n=19: PDA +0.157 at Pz, vs CEN +0.114 at CP6, DMN +0.107 at O1). The deployed EPOC-X
12-channel montage (F7 F3 FC5 T7 P7 O1 O2 P8 T8 FC6 F4 F8) contains no Pz/POz/CPz/Cz at all, and
on that montage PDA has no out-of-sample validity. efp_cen_clean.py:29 states the suspicion
directly: "EPOC X has NO centro-parietal midline (Pz/POz/Cz/P3/P4/CP1/CP2) where our CEN signal
peaks." Nobody has tested it, because Phase 2.5 evaluated cap31 for CEN/DMN only and dropped PDA.

Design. Hold the METHOD fixed (multivariate ridge on the [10 band x 11 delay] per-channel design,
exactly as train_pooled.py builds it) and vary ONLY the channel set:
    epoc12          12 ch  1320 feat   the deployed montage
    epoc12_pz       13 ch  1430 feat   + Pz  <- the single-electrode test
    epoc12_midline  15 ch  1650 feat   + Pz, POz, Cz
    cap31           31 ch  3410 feat   full research cap
Honest cross-subject LOSO over the 19 DMNELF subjects, with alpha chosen by nested subject-grouped
CV inside each training fold. This does NOT touch the locked external test set.

Efficiency. Naive LOSO would refit 19 x 4 x 3 ridges plus a nested alpha search. Instead we
accumulate per-subject Gram matrices (XtX, Xty) once; any train subset is then a SUM of those, and
leaving a subject out is a subtraction. One eigendecomposition per fold makes the whole alpha path
free. This is what keeps a nested-CV LOSO tractable at p=3410.
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np

FP = Path("/projects/swglab/data/DMNELF/analysis/fingerprint")
CACHE = FP / "19_fingerprint/efp_meirhasson/results/features_cache"
CENMEAN = FP / "efp_epoc" / "cen_mean_cache"
BASELINE_TR, HRF_DROP = 25, 5
ALPHAS = np.logspace(0, 8, 25)
TARGETS = ["CEN", "DMN", "PDA"]

EPOC12 = ["F7", "F3", "FC5", "T7", "P7", "O1", "O2", "P8", "T8", "FC6", "F4", "F8"]
CAP31 = ["Fp1", "Fp2", "F3", "F4", "C3", "C4", "P3", "P4", "O1", "O2", "F7", "F8", "T7", "T8",
         "P7", "P8", "Fz", "Cz", "Pz", "Oz", "FC1", "FC2", "CP1", "CP2", "FC5", "FC6", "CP5",
         "CP6", "TP9", "TP10", "POz"]
MONTAGES = {
    "epoc12":         EPOC12,
    "epoc12_pz":      EPOC12 + ["Pz"],
    "epoc12_midline": EPOC12 + ["Pz", "POz", "Cz"],
    "cap31":          CAP31,
}


def make_delay_design(bp_run, n_delays):
    n_bands, n_out = bp_run.shape
    n_valid = n_out - (n_delays - 1)
    if n_valid <= 0:
        return np.empty((0, n_bands * n_delays)), n_delays - 1
    X = np.empty((n_valid, n_bands * n_delays))
    for d in range(n_delays):
        X[:, d * n_bands:(d + 1) * n_bands] = bp_run[:, (n_delays - 1 - d):(n_out - d)].T
    return X, n_delays - 1


def zs(a):
    a = np.asarray(a, float)
    return (a - np.nanmean(a)) / (np.nanstd(a) + 1e-12)


def load_targets(sub):
    z = np.load(CENMEAN / f"cenmean_dmnelf_{sub}.npz", allow_pickle=True)
    out = {}
    for n in range(1, 9):
        if f"run{n}" in z.files and f"run{n}_dmn" in z.files:
            out[str(n)] = {"CEN": np.asarray(z[f"run{n}"], float),
                           "DMN": np.asarray(z[f"run{n}_dmn"], float)}
    return out


def subject_matrix(sub, channels, n_delays=11, target_source="cenmean"):
    """Return X (rows x p) and {target: y} for one subject, concatenated over runs."""
    z = np.load(CACHE / f"{sub}_efp.npz", allow_pickle=True)
    runs, ch_names = list(z["runs"]), [str(c) for c in z["ch_names"]]
    if any(c not in ch_names for c in channels):
        return None, None
    eidx = [ch_names.index(c) for c in channels]
    tv = load_targets(sub) if target_source == "cenmean" else None
    Xs, ys = [], {t: [] for t in TARGETS}
    for rd in runs:
        rd = rd.item() if hasattr(rd, "item") else rd
        r = str(rd["run"])
        if tv is not None and r not in tv:
            continue
        per_ch, off = [], None
        for ci in eidx:
            Xc, off = make_delay_design(rd["bp_tr"][ci], n_delays)
            per_ch.append((Xc - Xc.mean(0)) / (Xc.std(0) + 1e-12))
        if not per_ch or per_ch[0].shape[0] == 0:
            continue
        nvalid = per_ch[0].shape[0]
        t_idx = off + np.arange(nvalid)
        if tv is not None:
            cen = tv[r]["CEN"][off:off + nvalid]
            dmn = tv[r]["DMN"][off:off + nvalid]
            m = (t_idx >= BASELINE_TR + HRF_DROP) & np.isfinite(cen) & np.isfinite(dmn)
            if m.sum() < 20:
                continue
            Xs.append(np.column_stack([c[m] for c in per_ch]))
            a, c = cen[m], dmn[m]
            allv = np.concatenate([a, c]); mu, sd = allv.mean(), allv.std() + 1e-9
            ys["CEN"].append(zs(a)); ys["DMN"].append(zs(c))
            ys["PDA"].append(zs((a - mu) / sd - (c - mu) / sd))
        else:
            tgt = rd["tgt_tr"]                       # the research pipeline's own targets
            cols = {k: np.asarray(tgt[k], float)[off:off + nvalid] for k in TARGETS}
            m = (t_idx >= BASELINE_TR + HRF_DROP)
            for v in cols.values():
                m = m & np.isfinite(v)
            if m.sum() < 20:
                continue
            Xs.append(np.column_stack([c[m] for c in per_ch]))
            for k in TARGETS:
                ys[k].append(zs(cols[k][m]))
    if not Xs:
        return None, None
    return np.vstack(Xs), {t: np.concatenate(ys[t]) for t in TARGETS}


def ridge_from_gram(G, b, sx, sy, n, alphas, w=None, V=None):
    """Solve centred ridge for every alpha from accumulated sums.
    G=sum XtX, b=sum Xty, sx=sum X (per feature), sy=sum y, n=rows."""
    mx, my = sx / n, sy / n
    Gc = G - n * np.outer(mx, mx)
    bc = b - n * mx * my
    if w is None:
        w, V = np.linalg.eigh(Gc)
        w = np.maximum(w, 0.0)
    Vtb = V.T @ bc
    out = []
    for al in alphas:
        coef = V @ (Vtb / (w + al))
        out.append((coef, float(my - mx @ coef)))
    return out, w, V


def sign_flip_p(rs, n_perm=10000, seed=0):
    rs = np.asarray([r for r in rs if np.isfinite(r)], float)
    if rs.size < 3:
        return np.nan
    rng = np.random.default_rng(seed)
    obs = rs.mean()
    null = (rng.choice([-1.0, 1.0], size=(n_perm, rs.size)) * rs).mean(axis=1)
    return float((np.sum(np.abs(null) >= abs(obs)) + 1) / (n_perm + 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default=str(FP / "efp_pooled" / "cohort_split.json"))
    ap.add_argument("--inner", type=int, default=3)
    ap.add_argument("--out", required=True)
    ap.add_argument("--target-source", choices=["cenmean", "cache"], default="cenmean",
                    help="cenmean = rebuild targets from cen_mean_cache the way the DEPLOYED "
                         "export_model.py does; cache = use the research pipeline's own tgt_tr "
                         "targets stored in the feature cache (what the +0.157 LOSO benchmark used)")
    ap.add_argument("--single", action="store_true",
                    help="one model PER ELECTRODE (110 feat) instead of multivariate montages -- "
                         "isolates model complexity from electrode coverage")
    a = ap.parse_args()

    subs = json.load(open(a.split))["train"]["dmnelf"]
    global MONTAGES
    if a.single:
        MONTAGES = {ch: [ch] for ch in CAP31}
    print(f"LOSO over {len(subs)} DMNELF subjects; alpha by {a.inner}-fold nested grouped CV")
    print(f"target source: {a.target_source}"
          + ("  (deployed export_model.py definition)" if a.target_source == "cenmean"
             else "  (research pipeline tgt_tr -- comparable to the +0.157 LOSO benchmark)"))
    print(f"alpha grid {ALPHAS.min():g}..{ALPHAS.max():g} ({len(ALPHAS)} pts)\n")

    rows, summary = [], {}
    for mname, channels in MONTAGES.items():
        print(f"########## montage {mname}  ({len(channels)} ch, {len(channels)*110} feat)", flush=True)
        data = {}
        for s in subs:
            X, ys = subject_matrix(s, channels, target_source=a.target_source)
            if X is None:
                print(f"   {s}: skip"); continue
            data[s] = (X, ys)
        got = list(data)
        p = data[got[0]][0].shape[1]
        # per-subject accumulators
        acc = {s: {"XtX": X.T @ X, "sx": X.sum(0), "n": X.shape[0],
                   "Xty": {t: X.T @ ys[t] for t in TARGETS},
                   "sy": {t: float(ys[t].sum()) for t in TARGETS}}
               for s, (X, ys) in data.items()}
        print(f"   built Grams for {len(got)} subjects, p={p}", flush=True)

        for tgt in TARGETS:
            fold_r, chosen = [], []
            for k, held in enumerate(got):
                tr = [s for s in got if s != held]
                # ---- nested alpha selection on the training subjects only
                groups = [tr[i::a.inner] for i in range(a.inner)]
                inner_scores = np.zeros((len(ALPHAS), a.inner))
                for gi, va in enumerate(groups):
                    itr = [s for s in tr if s not in va]
                    G = sum(acc[s]["XtX"] for s in itr)
                    b = sum(acc[s]["Xty"][tgt] for s in itr)
                    sx = sum(acc[s]["sx"] for s in itr)
                    sy = sum(acc[s]["sy"][tgt] for s in itr)
                    n = sum(acc[s]["n"] for s in itr)
                    sols, _, _ = ridge_from_gram(G, b, sx, sy, n, ALPHAS)
                    Xv = np.vstack([data[s][0] for s in va])
                    yv = np.concatenate([data[s][1][tgt] for s in va])
                    for ai, (coef, icpt) in enumerate(sols):
                        pr = Xv @ coef + icpt
                        inner_scores[ai, gi] = 0.0 if np.std(pr) < 1e-12 else np.corrcoef(pr, yv)[0, 1]
                best_ai = int(np.nanargmax(np.nanmean(inner_scores, axis=1)))
                alpha = ALPHAS[best_ai]
                chosen.append(alpha)
                # ---- fit on all training subjects, score the held-out one
                G = sum(acc[s]["XtX"] for s in tr)
                b = sum(acc[s]["Xty"][tgt] for s in tr)
                sx = sum(acc[s]["sx"] for s in tr)
                sy = sum(acc[s]["sy"][tgt] for s in tr)
                n = sum(acc[s]["n"] for s in tr)
                sols, _, _ = ridge_from_gram(G, b, sx, sy, n, [alpha])
                coef, icpt = sols[0]
                Xh, yh = data[held][0], data[held][1][tgt]
                pr = Xh @ coef + icpt
                r = float(np.corrcoef(pr, yh)[0, 1]) if np.std(pr) > 1e-12 else np.nan
                fold_r.append(r)
                rows.append({"montage": mname, "n_ch": len(channels), "target": tgt,
                             "held_out": held, "alpha": float(alpha), "r": r})
            fr = np.array(fold_r, float)
            mean, sem = float(np.nanmean(fr)), float(np.nanstd(fr, ddof=1) / np.sqrt(np.sum(np.isfinite(fr))))
            pv = sign_flip_p(fr)
            summary[f"{mname}|{tgt}"] = {"mean_r": mean, "sem": sem, "p": pv,
                                         "n": int(np.sum(np.isfinite(fr))),
                                         "median_alpha": float(np.median(chosen))}
            print(f"   {tgt}: LOSO r = {mean:+.4f} ± {sem:.4f}  p = {pv:.4f}  "
                  f"(median alpha {np.median(chosen):g})", flush=True)
        print()

    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["montage", "n_ch", "target", "held_out", "alpha", "r"])
        w.writeheader(); w.writerows(rows)
    Path(str(out).replace(".csv", "_summary.json")).write_text(json.dumps(summary, indent=2))

    print("=" * 74)
    print(f"{'montage':16s} {'n_ch':>4s}  " + "  ".join(f"{t:>18s}" for t in TARGETS))
    print("-" * 74)
    for mname, ch in MONTAGES.items():
        cells = []
        for t in TARGETS:
            s = summary.get(f"{mname}|{t}")
            cells.append(f"{s['mean_r']:+.3f} p={s['p']:.3f}" if s else "n/a")
        print(f"{mname:16s} {len(ch):>4d}  " + "  ".join(f"{c:>18s}" for c in cells))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
