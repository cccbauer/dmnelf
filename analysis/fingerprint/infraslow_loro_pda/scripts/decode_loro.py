"""
decode_loro.py
--------------
Within-subject, within-feedback PDA decoder.

For each participant:
  feature  : per-TR block-mean over 600 samples/channel of infraslow (0.01-40Hz)
             EEG -> [N_TR x 31], z-scored per channel.
  target   : PDA (CEN-DMN) per TR (from cyclic_features npz).
  model    : ElasticNet  alpha*[rho*||w||_1 + (1-rho)*0.5*||w||^2]  (ElasticNetCV
             picks alpha/rho by inner CV on the training runs).
  CV       : leave-one-run-out over the 4 feedback runs; held-out predictions are
             concatenated to a full predicted PDA timeseries.
  smoothing: post-hoc centered moving-average (window=11), applied PER RUN to the
             held-out predictions (aligns EEG-derived prediction to BOLD's slower
             frequency content).

Reports, per subject and cohort: Pearson r (raw and smoothed). A circular-shift
permutation p (shift within run, preserves autocorrelation) is added because
smoothing autocorrelated signals inflates the parametric p. Baseline (1-40Hz) is
run as a control.

Usage: python decode_loro.py --config config.yaml
"""
import argparse, warnings, csv
from pathlib import Path
import numpy as np, yaml, mne
from scipy.stats import pearsonr
from sklearn.linear_model import ElasticNetCV
from sklearn.preprocessing import StandardScaler
warnings.simplefilter("ignore"); mne.set_log_level("ERROR")


def load_config(p):
    cfg = yaml.safe_load(open(p)); d = cfg["data"]
    d["features_dir"] = (d["features_dir_cluster"] if Path("/projects/swglab").exists()
                         else d["features_dir_local"])
    return cfg


def block_mean(raw, spt, n):
    x = raw.get_data(picks="eeg"); x = x[:, :n*spt].reshape(x.shape[0], n, spt).mean(2).T
    return ((x - x.mean(0)) / (x.std(0) + 1e-8)).astype(np.float64)


def gather(cfg, subj, task, desc):
    """Return list of (pda, X) per run, in run order."""
    fdir = Path(cfg["data"]["features_dir"]) / f"sub-{subj}"
    eroot = Path(cfg["data"]["eeg_preproc_dir"]); ses = cfg["data"]["session"]
    spt = cfg["data"]["eeg"]["samples_per_tr"]; runs = []
    for npz in sorted(fdir.glob(f"sub-{subj}_task-{task}_run-*_features.npz")):
        d = np.load(npz, allow_pickle=True); pda = np.asarray(d["pda"], float)
        run = npz.name.split("run-")[1][0]
        fif = eroot/f"sub-{subj}"/ses/"eeg"/f"sub-{subj}_{ses}_task-{task}_run-{int(run):02d}_desc-{desc}_eeg.fif"
        if not fif.exists():
            continue
        raw = mne.io.read_raw_fif(str(fif), preload=True, verbose=False)
        runs.append((pda, block_mean(raw, spt, len(pda))))
    return runs


def moving_average(x, w):
    if w <= 1:
        return x
    k = np.ones(w) / w; pl = w // 2; pr = w - 1 - pl
    return np.convolve(np.pad(x, (pl, pr), mode="edge"), k, mode="valid")


def loro_predict(runs, mcfg):
    """Leave-one-run-out ElasticNet. Returns per-run (true, pred) in run order."""
    out = []
    for i in range(len(runs)):
        tr = [runs[j] for j in range(len(runs)) if j != i]
        Xtr = np.vstack([X for _, X in tr]); ytr = np.concatenate([y for y, _ in tr])
        yte, Xte = runs[i][0], runs[i][1]
        sc = StandardScaler().fit(Xtr)
        m = ElasticNetCV(l1_ratio=mcfg["l1_ratios"], n_alphas=mcfg["n_alphas"],
                         cv=mcfg["cv_inner"], max_iter=mcfg["max_iter"])
        m.fit(sc.transform(Xtr), ytr)
        out.append((yte, m.predict(sc.transform(Xte))))
    return out


def score(per_run, window, smooth_both=False):
    """Concatenate held-out runs; r raw and r with per-run smoothed predictions.

    smooth_both=False (default, honest): smooth the PREDICTION only.
    smooth_both=True (diagnostic): smooth BOTH prediction and true with the same
    window before correlating. This reproduces the old buggy cyclic-transcoder
    eval; the shared low-pass autocorrelation inflates r and is NOT a real gain.
    """
    true = np.concatenate([t for t, _ in per_run])
    pred = np.concatenate([p for _, p in per_run])
    pred_s = np.concatenate([moving_average(p, window) for _, p in per_run])
    if smooth_both:
        true_s = np.concatenate([moving_average(t, window) for t, _ in per_run])
    else:
        true_s = true
    r_raw = pearsonr(pred, true)[0] if pred.std() > 0 else np.nan
    r_smo = pearsonr(pred_s, true_s)[0] if pred_s.std() > 0 and true_s.std() > 0 else np.nan
    # circular-shift null on the smoothed prediction (shift the comparison true
    # signal within run; preserves autocorrelation). When smooth_both, the
    # comparison signal is the smoothed true, so the null shares its inflation.
    rng = np.random.default_rng(0); nperm = 2000; obs = abs(r_smo) if not np.isnan(r_smo) else 0
    ge = 0
    run_lens = [len(t) for t, _ in per_run]
    for _ in range(nperm):
        shifted = []
        off = 0
        for L in run_lens:
            seg = true_s[off:off+L]; k = int(rng.integers(5, max(6, L-5)))
            shifted.append(np.roll(seg, k)); off += L
        ts = np.concatenate(shifted)
        if abs(pearsonr(pred_s, ts)[0]) >= obs: ge += 1
    p = (ge + 1) / (nperm + 1)
    return r_raw, r_smo, p, len(true)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--out", default="results/decode_loro.csv")
    ap.add_argument("--smooth-both", action="store_true",
                    help="DIAGNOSTIC: smooth both pred AND true before correlating "
                         "(reproduces the old buggy eval; inflates r, not a real gain).")
    a = ap.parse_args(); cfg = load_config(a.config)
    subs = [s for s in cfg["data"]["subjects"]["all"]
            if s not in set(cfg["data"]["subjects"].get("exclude", []))]
    task = cfg["data"]["task"]; mcfg = cfg["model"]; w = cfg["smoothing"]["window"]
    di = cfg["data"]["eeg"]["desc_infraslow"]; db = cfg["data"]["eeg"]["desc"]

    sb = a.smooth_both
    print(f"LORO within-{task} decode  (ElasticNet, smooth w={w}"
          f"{', SMOOTH-BOTH diagnostic' if sb else ''})")
    print(f"{'subject':11s} | {'INFRASLOW r_raw  r_sm   p(circ)':32s} | {'BASELINE r_raw  r_sm':20s}")
    print("-" * 76)
    rows = []
    for s in subs:
        ri = gather(cfg, s, task, di); rb = gather(cfg, s, task, db)
        if len(ri) < 2:
            print(f"{s:11s} | <2 infraslow runs"); continue
        rr, rs, p, n = score(loro_predict(ri, mcfg), w, smooth_both=sb)
        if len(rb) >= 2:
            br, bs, bp, _ = score(loro_predict(rb, mcfg), w, smooth_both=sb)
        else:
            br = bs = np.nan
        print(f"{s:11s} | r_raw={rr:+.3f} r_sm={rs:+.3f} p={p:.4f}      "
              f"| r_raw={br:+.3f} r_sm={bs:+.3f}")
        rows.append(dict(subject=s, is_r_raw=rr, is_r_smooth=rs, is_p_circ=p,
                         base_r_raw=br, base_r_smooth=bs, n=n))
    if rows:
        isr = np.array([r['is_r_raw'] for r in rows]); iss = np.array([r['is_r_smooth'] for r in rows])
        bsr = np.array([r['base_r_raw'] for r in rows]); bss = np.array([r['base_r_smooth'] for r in rows])
        nsig = sum(1 for r in rows if r['is_p_circ'] < 0.05 and r['is_r_smooth'] > 0)
        print("-" * 76)
        print(f"INFRASLOW mean r_raw={np.nanmean(isr):+.3f}  r_smooth={np.nanmean(iss):+.3f}  "
              f"(smoothing delta={100*(np.nanmean(iss)-np.nanmean(isr))/abs(np.nanmean(isr)):+.0f}%)  "
              f"sig(circ p<.05,r>0)={nsig}/{len(rows)}")
        print(f"BASELINE  mean r_raw={np.nanmean(bsr):+.3f}  r_smooth={np.nanmean(bss):+.3f}")
        out_rel = a.out
        if sb and out_rel == "results/decode_loro.csv":
            out_rel = "results/decode_loro_smoothboth.csv"   # never overwrite the honest result
        outp = Path(cfg["project"]["base_dir"]) / out_rel
        outp.parent.mkdir(parents=True, exist_ok=True)
        with open(outp, "w", newline="") as f:
            wr = csv.DictWriter(f, fieldnames=list(rows[0].keys())); wr.writeheader(); wr.writerows(rows)
        print(f"saved: {outp}")


if __name__ == "__main__":
    main()
