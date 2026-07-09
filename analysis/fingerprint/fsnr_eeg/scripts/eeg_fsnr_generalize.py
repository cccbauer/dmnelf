#!/usr/bin/env python3
"""
eeg_fsnr_generalize.py  —  Stream B generalization (LOSO + cross-cohort)
------------------------------------------------------------------------
The pure EEG f-SNR construct (band-power flavor) has NO fitted parameters, so its match to
PDA should transfer. We test:
  - DMNELF LOSO: leave-one-subject-out, pick the best construct (band x montage x kind) on
    the N-1 training subjects, evaluate on the held-out subject -> honest OOF matched r.
  - Cross-cohort rtBPD (nf1, nf2): apply the FIXED DMNELF-winning construct
    (frontal-theta f-SNR) to an independent cohort, matched to rtBPD PDA (signed, so the
    sign convention must transfer too). Zero fitting.
"""
from pathlib import Path
import numpy as np, glob, re
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "fsnr" / "scripts"))
from fsnr_proxy import running_fsnr
sys.path.insert(0, str(Path(__file__).resolve().parent))
from eeg_fsnr_bandpower import zs, subject_runs, FRONTAL, POST

PROJ = Path(__file__).resolve().parent.parent
DATA = PROJ / "data"; RES = PROJ / "results"

# candidate constructs: (band, montage, kind)
CANDS = [("theta", FRONTAL, "fsnr"), ("alpha", FRONTAL, "fsnr"), ("alpha", POST, "fsnr"),
         ("theta", FRONTAL, "raw"), ("alpha", POST, "raw")]
WINNER = ("theta", FRONTAL, "fsnr")     # DMNELF winner, fixed for cross-cohort


def cname(c):
    return f"{c[2]}:{c[0]}@{'front' if c[1] is FRONTAL else 'post'}"


def construct_match(runs, chs, cand):
    """Signed per-subject match r of one construct to PDA (mean over runs)."""
    band, chset, kind = cand
    pi = [i for i, c in enumerate(chs) if c in chset]
    if not pi:
        return np.nan
    rs = []
    for rd in runs:
        pda = zs(np.asarray(rd["targets"]["PDA"], float))
        if kind == "fsnr":
            e = zs(np.nanmean(np.column_stack([running_fsnr(rd["bp"][band][:, i])[1] for i in pi]), 1))
        else:
            e = zs(np.nanmean(np.column_stack([rd["bp"][band][:, i] for i in pi]), 1))
        m = np.isfinite(e) & np.isfinite(pda)
        if m.sum() > 20:
            rs.append(np.corrcoef(e[m], pda[m])[0, 1])
    return np.nanmean(rs) if rs else np.nan


def sflip(x, n=10000):
    x = np.asarray([v for v in x if np.isfinite(v)]); rng = np.random.default_rng(0)
    obs = x.mean(); null = (rng.choice([-1, 1], (n, len(x))) * np.abs(x)).mean(1)
    return obs, (np.sum(null >= obs) + 1) / (n + 1)


def cohort(folder, pattern="*_bandpower.npz"):
    out = {}
    for f in sorted(glob.glob(str(folder / pattern))):
        sub = re.search(r"(\w+?)_bandpower", Path(f).name).group(1)
        runs, chs = subject_runs(f)
        out[sub] = np.array([construct_match(runs, chs, c) for c in CANDS])
    return out


def main():
    dmn = cohort(DATA)                         # DMNELF (flat *_bandpower.npz)
    subs = list(dmn); M = np.array([dmn[s] for s in subs])   # [17 x ncand]
    print(f"DMNELF n={len(subs)}\n=== within-subject construct match to PDA (signed mean; p) ===")
    for j, c in enumerate(CANDS):
        o, p = sflip(M[:, j]); print(f"  {cname(c):18s} r={o:+.3f}  p={p:.4f}")

    # LOSO: pick best construct on N-1 (by mean), score held-out
    oof = []
    for i in range(len(subs)):
        others = np.delete(M, i, 0)
        jbest = np.nanargmax(np.nanmean(others, 0))
        oof.append(M[i, jbest])
    o, p = sflip(np.array(oof))
    print(f"\n=== DMNELF LOSO-selected construct (OOF) r={o:+.3f}  p={p:.4f} ===")

    # Cross-cohort rtBPD: FIXED winner construct
    jw = CANDS.index(WINNER)
    print(f"\n=== cross-cohort rtBPD — FIXED construct {cname(WINNER)} (zero fitting) ===")
    for tag, folder in [("nf1", DATA / "rtbpd_nf1"), ("nf2", DATA / "rtbpd_nf2")]:
        rt = cohort(folder)
        vals = np.array([v[jw] for v in rt.values()])
        o, p = sflip(vals)
        print(f"  {tag}: n={len(rt)}  r={o:+.3f}  p={p:.4f}  ({100*np.mean(vals>0):.0f}% subjects positive)")
        # also all candidates for context
        for j, c in enumerate(CANDS):
            vv = np.array([v[j] for v in rt.values()]); oo, pp = sflip(vv)
            print(f"      {cname(c):18s} r={oo:+.3f} p={pp:.4f}")


if __name__ == "__main__":
    main()
