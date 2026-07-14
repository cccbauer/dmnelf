#!/usr/bin/env python3
"""
cen_learned_decode.py  —  learned temporal EEG -> clean CEN-BOLD model
----------------------------------------------------------------------
Predict the CLEAN (confound-regressed) CEN mask-mean timecourse from EEG, adding the two
things the per-TR decoder lacked: (1) TEMPORAL CONTEXT (±k-TR lags — learn the EEG->BOLD
transfer, offline/non-causal OK), (2) richer features (band power + per-TR f-SNR).

Target: cenmean_{cohort}_{sub}.npz  run{N} = clean CEN (motion+WM/CSF+cosine regressed).
  Robustness: run{N}_gsr (global-signal also removed) + EEG-side motion regression [optional].
Eval: LORO (within-subject) + LOSO, feedback + full window, vs circular-shift null and the
~1.0 reliability ceiling. Models: lagged RidgeCV (baseline) and an MLP.

Usage: python cen_learned_decode.py --cohort {dmnelf,rtbpd} [--lags 3] [--gsr] [--mlp]
"""
from pathlib import Path
import numpy as np, glob, re, sys, argparse
sys.path.insert(0, str(Path(__file__).resolve().parent))
from within_fb_decode import zs, sflip, ALPHAS, BASELINE_TR, HRF_DROP, BANDS
from sklearn.linear_model import RidgeCV
from sklearn.neural_network import MLPRegressor

DATA = Path(__file__).resolve().parents[1] / "data"
PERTR = Path(__file__).resolve().parents[1] / "results" / "pertr_fsnr"
CEN = Path(__file__).resolve().parents[1] / "results" / "cen_ceiling"
QA = re.compile(r"dmnelf999")   # only 999 is phantom; 1001-1003 are REAL subjects
RNG = np.random.default_rng(0)


def bin_avg(x, b):
    """Non-overlapping bin-average (temporal downsampling) to a coarser timescale."""
    if b <= 1:
        return x
    m = (len(x) // b) * b
    return x[:m].reshape(-1, b).mean(1)


def lag_stack(X, k):
    """Stack ±k-TR shifted copies -> [T, F*(2k+1)] (bidirectional temporal context)."""
    T, F = X.shape; cols = []
    for L in range(-k, k + 1):
        s = np.zeros_like(X)
        if L < 0:
            s[:L] = X[-L:]
        elif L > 0:
            s[L:] = X[:-L]
        else:
            s = X.copy()
        cols.append(s)
    return np.column_stack(cols)


def eeg_feats(rd, pr, kind):
    nch = rd["bp"]["theta"].shape[1]; n = rd["n_tr"]; parts = []
    if "bp" in kind:
        parts.append(np.column_stack([rd["bp"][b][:, c] for b in BANDS for c in range(nch)]))
    if "pertr" in kind:
        parts.append(pr["tsnr"].reshape(n, -1))
        w = pr["w3600"]
        parts.append(np.column_stack([w["periodic"], w["exponent"], w["offset"],
                                      w["bandpow"].reshape(n, -1)]))
    return np.column_stack(parts)


def build(bp_npz, pertr_npz, cen_npz, kind, k, window, gsr):
    z = np.load(bp_npz, allow_pickle=True); pt = np.load(pertr_npz, allow_pickle=True)
    cm = np.load(cen_npz, allow_pickle=True)
    runmap = {int(r.replace("run", "")): pt[r].item() for r in pt["_runs"]}
    Xs, ys = [], []
    for rd in z["runs_data"]:
        run = int(rd["run"]); key = f"run{run}" + ("_gsr" if gsr else "")
        if run not in runmap or key not in cm.files:
            continue
        Xf = lag_stack(eeg_feats(rd, runmap[run], kind), k)
        yf = np.asarray(cm[key], float)
        m = min(len(Xf), len(yf))                      # align 1:1 (guard vs off-by-one)
        sl = slice(BASELINE_TR + HRF_DROP, m) if window == "fb" else slice(0, m)
        X = Xf[:m][sl]; y = zs(yf[:m][sl])
        ok = np.all(np.isfinite(X), 1) & np.isfinite(y)
        if ok.sum() > 20:
            Xs.append(X[ok]); ys.append(y[ok])
    return Xs, ys


def fit_predict(Xtr, ytr, Xte, mlp):
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-12
    Xtr, Xte = (Xtr - mu) / sd, (Xte - mu) / sd
    if mlp:
        m = MLPRegressor(hidden_layer_sizes=(64,), alpha=3.0, max_iter=400,
                         early_stopping=True, random_state=0).fit(Xtr, ytr)
    else:
        m = RidgeCV(alphas=ALPHAS).fit(Xtr, ytr)
    return m.predict(Xte)


def loro(Xs, ys, mlp=False, b=1):
    if len(Xs) < 2:
        return np.nan
    obs, pred = [], []
    for i in range(len(Xs)):
        tr = [j for j in range(len(Xs)) if j != i]
        p = fit_predict(np.vstack([Xs[j] for j in tr]), np.concatenate([ys[j] for j in tr]), Xs[i], mlp)
        pred.append(bin_avg(p, b)); obs.append(bin_avg(ys[i], b))   # per-run downsample
    o, p = np.concatenate(obs), np.concatenate(pred)
    return float(np.corrcoef(o, p)[0, 1]) if len(p) > 3 and np.std(p) > 1e-9 else np.nan


def loso(pairs, kind, k, window, gsr, mlp=False):
    data = [(np.vstack(Xs), np.concatenate(ys)) for Xs, ys in
            (build(*p, kind, k, window, gsr) for p in pairs) if Xs]
    r = []
    for i in range(len(data)):
        Xte, yte = data[i]
        Xtr = np.vstack([data[j][0] for j in range(len(data)) if j != i])
        ytr = np.concatenate([data[j][1] for j in range(len(data)) if j != i])
        p = fit_predict(Xtr, ytr, Xte, mlp)
        if np.std(p) > 1e-9:
            r.append(np.corrcoef(yte, p)[0, 1])
    return np.array(r)


def get_pairs(cohort):
    bg = str(DATA / "*_bandpower.npz") if cohort == "dmnelf" else str(DATA / "rtbpd_nf1" / "*_bandpower.npz")
    bps = {re.search(rf"({cohort}\w+)_bandpower", f).group(1): f
           for f in glob.glob(bg) if not (cohort == "dmnelf" and QA.search(f))}
    out = []
    for s, bp in sorted(bps.items()):
        p = PERTR / f"{s}_pertr.npz"; c = CEN / f"cenmean_{cohort}_{s}.npz"
        if p.exists() and c.exists():
            out.append((bp, str(p), str(c)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohort", default="dmnelf", choices=["dmnelf", "rtbpd"])
    ap.add_argument("--lags", type=int, default=3)
    ap.add_argument("--gsr", action="store_true")
    ap.add_argument("--mlp", action="store_true")
    ap.add_argument("--null", type=int, default=0)
    ap.add_argument("--sweep", action="store_true", help="downsample/timescale sweep (r vs bin size)")
    a = ap.parse_args()
    pairs = get_pairs(a.cohort)

    if a.sweep:
        print(f"{a.cohort}: {len(pairs)} subjects | DOWNSAMPLE sweep (bp+pertr, ±{a.lags}TR, "
              f"{'MLP' if a.mlp else 'ridge'}) | target={'GSR' if a.gsr else 'clean'} CEN")
        print("bin = TR averaged (1 TR = 1.2 s). LORO r per timescale:\n")
        print(f"{'window':6s} " + "  ".join(f"{b}TR/{b*1.2:.1f}s" for b in [1, 2, 3, 5, 10]))
        for window in ["full", "fb"]:
            cells = []
            for b in [1, 2, 3, 5, 10]:
                r = np.array([loro(*build(*p, "bp+pertr", a.lags, window, a.gsr), mlp=a.mlp, b=b)
                              for p in pairs])
                o, pv, _ = sflip(r)
                cells.append(f"{o:+.3f}{'*' if pv < 0.05 else ' '}")
            print(f"{window:6s} " + "  ".join(f"{c:>9s}" for c in cells))
        print("\nceiling ~1.0 at all timescales; watch fewer bins = noisier r estimate.")
        return
    tgt = "GSR-CEN (robustness)" if a.gsr else "clean CEN (standard)"
    print(f"{a.cohort}: {len(pairs)} subjects | target={tgt} | ±{a.lags}-TR context | "
          f"model={'MLP' if a.mlp else 'lagged-ridge'}\n")
    print(f"{'feature':10s} {'window':6s} {'LORO r':>9s} {'p':>6s} | {'LOSO r':>9s} {'p':>6s}")
    for kind in ["bp", "bp+pertr"]:
        for window in ["full", "fb"]:
            lr = np.array([loro(*build(*p, kind, a.lags, window, a.gsr), mlp=a.mlp) for p in pairs])
            lo, lp, _ = sflip(lr)
            ls = loso(pairs, kind, a.lags, window, a.gsr, mlp=a.mlp); so, sp, _ = sflip(ls)
            print(f"{kind:10s} {window:6s} {lo:+9.3f} {lp:6.3f} | {so:+9.3f} {sp:6.3f}")
    if a.null > 0:
        print("\n=== circular-shift null (bp+pertr, full, LORO) ===")
        obs = np.nanmean([loro(*build(*p, "bp+pertr", a.lags, "full", a.gsr)) for p in pairs])
        nulls = []
        for _ in range(a.null):
            rs = []
            for p in pairs:
                Xs, ys = build(*p, "bp+pertr", a.lags, "full", a.gsr)
                ys = [np.roll(y, RNG.integers(5, len(y) - 5)) for y in ys]
                rs.append(loro(Xs, ys))
            nulls.append(np.nanmean(rs))
        nulls = np.array(nulls)
        print(f"  obs={obs:+.3f}  null={nulls.mean():+.3f}±{nulls.std():.3f}  p={(nulls>=obs).mean():.3f}")


if __name__ == "__main__":
    main()
