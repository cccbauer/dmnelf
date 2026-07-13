#!/usr/bin/env python3
"""
within_fb_decode.py  —  does ANY EEG feature decode the target WITHIN the feedback block?
-----------------------------------------------------------------------------------------
The headline numbers (EFP 0.26, construct f-SNR 0.119) are full-run and inflated by the
rest->feedback state step. Here we decode PDA/CEN/DMN restricted to the FEEDBACK block
(state step removed), leak-free:

  window   : feedback = TR 30..end (25 rest + 5 HRF drop) ; also 'full' for head-to-head
  target   : z-scored PER RUN over the window (removes the run offset = the step)
  CV       : leave-one-run-out (LORO, within subject, primary) and leave-one-subject-out (LOSO)
  model    : multivariate RidgeCV over all 31ch x 5band features
  features : band power (cache) ; running/trailing f-SNR (the known state-level baseline)
  null     : circular-shift the per-run target (respects autocorrelation) -> chance group r

Discovery cohort = DMNELF. Reports OOF r per feature x target x window x CV, group sign-flip p,
and a circular-shift null for the feedback-block band-power decoder.
"""
from pathlib import Path
import numpy as np, glob, re, sys
from scipy import stats
from sklearn.linear_model import RidgeCV
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "fsnr" / "scripts"))
from fsnr_proxy import running_fsnr

DATA = Path(__file__).resolve().parents[1] / "data"
BANDS = ["delta", "theta", "alpha", "beta", "gamma"]
TARGETS = ["PDA", "CEN", "DMN"]
BASELINE_TR, HRF_DROP = 25, 5
QA = re.compile(r"dmnelf(999|1\d\d\d)")
ALPHAS = np.logspace(-2, 5, 15)
RNG = np.random.default_rng(0)


def zs(x): return (x - np.nanmean(x)) / (np.nanstd(x) + 1e-12)


def feats(rd, kind):
    nch = rd["bp"]["theta"].shape[1]
    if kind == "bandpower":
        return np.column_stack([rd["bp"][b][:, c] for b in BANDS for c in range(nch)])
    return np.column_stack([running_fsnr(rd["bp"][b][:, c])[1] for b in BANDS for c in range(nch)])


def subj_runs(f, kind, target, window):
    z = np.load(f, allow_pickle=True)
    Xs, ys = [], []
    for rd in z["runs_data"]:
        n = rd["n_tr"]; sl = slice(BASELINE_TR + HRF_DROP, n) if window == "fb" else slice(0, n)
        X = feats(rd, kind)[sl]; y = zs(np.asarray(rd["targets"][target], float)[sl])
        ok = np.all(np.isfinite(X), 1) & np.isfinite(y)
        if ok.sum() > 20:
            Xs.append(X[ok]); ys.append(y[ok])
    return Xs, ys


def loro(Xs, ys):
    if len(Xs) < 2: return np.nan
    obs, pred = [], []
    for i in range(len(Xs)):
        tr = [j for j in range(len(Xs)) if j != i]
        Xtr = np.vstack([Xs[j] for j in tr]); ytr = np.concatenate([ys[j] for j in tr])
        mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-12
        m = RidgeCV(alphas=ALPHAS).fit((Xtr - mu) / sd, ytr)
        pred.append(m.predict((Xs[i] - mu) / sd)); obs.append(ys[i])
    o, p = np.concatenate(obs), np.concatenate(pred)
    return float(np.corrcoef(o, p)[0, 1]) if np.std(p) > 1e-9 else np.nan


def loso(files, kind, target, window):
    S = {f: subj_runs(f, kind, target, window) for f in files}
    subj_r = []
    for held in files:
        Xte = np.vstack(S[held][0]) if S[held][0] else None
        if Xte is None: continue
        yte = np.concatenate(S[held][1])
        Xtr = np.vstack([x for f in files if f != held for x in S[f][0]])
        ytr = np.concatenate([y for f in files if f != held for y in S[f][1]])
        mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-12
        m = RidgeCV(alphas=ALPHAS).fit((Xtr - mu) / sd, ytr)
        p = m.predict((Xte - mu) / sd)
        if np.std(p) > 1e-9:
            subj_r.append(np.corrcoef(yte, p)[0, 1])
    return np.array(subj_r)


def sflip(a, n=10000):
    a = a[np.isfinite(a)]; obs = a.mean()
    null = (RNG.choice([-1, 1], (n, len(a))) * np.abs(a)).mean(1)
    return obs, float((np.abs(null) >= abs(obs)).mean()), len(a)


def main():
    files = sorted(f for f in glob.glob(str(DATA / "*_bandpower.npz")) if not QA.search(f))
    print(f"DMNELF discovery: {len(files)} subjects\n")
    print(f"{'feature':11s} {'target':4s} {'window':5s} {'LORO r':>8s} {'p':>6s} | {'LOSO r':>8s} {'p':>6s}")
    results = {}
    for kind in ["bandpower", "running_fsnr"]:
        for window in ["fb", "full"]:
            for tg in TARGETS:
                loro_r = np.array([loro(*subj_runs(f, kind, tg, window)) for f in files])
                lo, lp, _ = sflip(loro_r)
                ls = loso(files, kind, tg, window); so, sp, _ = sflip(ls)
                results[(kind, window, tg)] = loro_r
                print(f"{kind:11s} {tg:4s} {window:5s} {lo:+8.3f} {lp:6.3f} | {so:+8.3f} {sp:6.3f}")
        print()

    # circular-shift null for the feedback-block band-power decoder (the key claim)
    print("=== circular-shift null (bandpower, feedback block, LORO) ===")
    for tg in TARGETS:
        obs = np.nanmean(results[("bandpower", "fb", tg)])
        nulls = []
        for _ in range(100):
            rs = []
            for f in files:
                Xs, ys = subj_runs(f, "bandpower", tg, "fb")
                ys = [np.roll(y, RNG.integers(5, len(y) - 5)) for y in ys]
                rs.append(loro(Xs, ys))
            nulls.append(np.nanmean(rs))
        nulls = np.array(nulls)
        p = float((nulls >= obs).mean())
        print(f"  {tg}: obs={obs:+.3f}  null={nulls.mean():+.3f}±{nulls.std():.3f}  p={p:.3f}")


if __name__ == "__main__":
    main()
