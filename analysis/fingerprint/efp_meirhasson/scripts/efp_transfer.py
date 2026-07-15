#!/usr/bin/env python3
"""
efp_transfer.py  —  honest cross-cohort transfer: DMNELF general fingerprint -> rtBPD
-------------------------------------------------------------------------------------
Train ONE fixed EFP on ALL DMNELF (feedback block), freeze it, apply UNCHANGED to each
rtBPD nf1/nf2 subject (feedback block, NO refit). Answers: "can the DMNELF fingerprint be
used for rtBPD?". Per-run feature z-scoring divides out cohort-specific gain (the transfer
trick). Targets CEN/DMN/PDA; modes: best (single electrode selected on DMNELF by LOSO) and
all (multivariate). Reports per-rtBPD-subject r -> group mean + sign-flip p.

Runs on cluster (caches there). Output: efp_transfer.csv (session, target, mode, r per subject).
"""
import argparse, glob
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import zscore, pearsonr
from sklearn.linear_model import RidgeCV
from efp_features import load_config, load_subject_features, make_delay_design

BASELINE_TR, HRF_DROP = 25, 5
ALPHAS = np.logspace(-2, 5, 15)
B = "/projects/swglab/data/DMNELF/analysis/fingerprint/efp_meirhasson"
TRAIN_CACHE = f"{B}/results/features_cache"
TEST = {"nf1": f"{B}/results/features_cache_rtbpd", "nf2": f"{B}/results/features_cache_rtbpd_nf2"}


def subjects(cache):
    return sorted(Path(p).name.replace("_efp.npz", "") for p in glob.glob(f"{cache}/*_efp.npz"))


def subj_designs(cache, sub, target, n_delays):
    """Per-channel feedback-block design (per-run z-scored) + y, concatenated over runs."""
    runs, ch = load_subject_features(Path(cache), sub); nch = len(ch)
    per_ch = [[] for _ in range(nch)]; ys = []
    for rd in runs:
        Xs, off = [], None
        for ci in range(nch):
            X, off = make_delay_design(rd["bp_tr"][ci], n_delays)
            Xs.append((X - X.mean(0)) / (X.std(0) + 1e-12))
        nvalid = Xs[0].shape[0]; t_idx = off + np.arange(nvalid)
        y = np.asarray(rd["tgt_tr"][target], float)[off:off + nvalid]
        m = (t_idx >= BASELINE_TR + HRF_DROP) & np.isfinite(y)
        if m.sum() < 20:
            continue
        for ci in range(nch):
            per_ch[ci].append(Xs[ci][m])
        ys.append(zscore(y[m]))
    if not ys:
        return None
    return [np.vstack(per_ch[ci]) for ci in range(nch)], np.concatenate(ys), nch


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True)
    a = ap.parse_args(); out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    cfg = load_config(); tr = cfg["data"]["fmri"]["tr"]
    n_delays = int(round(cfg["efp"]["delay_window_s"] / tr)) + 1
    tr_subs = subjects(TRAIN_CACHE)
    rows = []
    for target in ["CEN", "DMN", "PDA"]:
        # --- assemble DMNELF training designs ---
        D = {s: subj_designs(TRAIN_CACHE, s, target, n_delays) for s in tr_subs}
        D = {s: v for s, v in D.items() if v is not None}
        nch = next(iter(D.values()))[2]
        # electrode selection on DMNELF (LOSO across subjects), best mean r
        best_ci, best = None, -np.inf
        for ci in range(nch):
            rs = []
            for held in D:
                Xtr = np.vstack([D[s][0][ci] for s in D if s != held])
                ytr = np.concatenate([D[s][1] for s in D if s != held])
                m = RidgeCV(alphas=ALPHAS).fit(Xtr, ytr)
                p = m.predict(D[held][0][ci])
                if np.std(p) > 1e-9:
                    rs.append(pearsonr(D[held][1], p)[0])
            if rs and np.mean(rs) > best:
                best, best_ci = np.mean(rs), ci
        # freeze models on ALL DMNELF
        m_best = RidgeCV(alphas=ALPHAS).fit(np.vstack([D[s][0][best_ci] for s in D]),
                                            np.concatenate([D[s][1] for s in D]))
        m_all = RidgeCV(alphas=ALPHAS).fit(np.vstack([np.column_stack(D[s][0]) for s in D]),
                                           np.concatenate([D[s][1] for s in D]))
        # --- apply UNCHANGED to rtBPD nf1 / nf2 ---
        for sess, cache in TEST.items():
            for s in subjects(cache):
                v = subj_designs(cache, s, target, n_delays)
                if v is None:
                    continue
                Xc, y, _ = v
                pb = m_best.predict(Xc[best_ci])
                pa = m_all.predict(np.column_stack(Xc))
                rows.append(dict(session=sess, target=target, mode="best",
                                 subject=s, r=pearsonr(y, pb)[0] if np.std(pb) > 1e-9 else np.nan))
                rows.append(dict(session=sess, target=target, mode="all",
                                 subject=s, r=pearsonr(y, pa)[0] if np.std(pa) > 1e-9 else np.nan))
        print(f"{target}: DMNELF best electrode idx={best_ci} (LOSO r={best:.3f})", flush=True)
    pd.DataFrame(rows).to_csv(out / "efp_transfer.csv", index=False)
    print("saved efp_transfer.csv", flush=True)


if __name__ == "__main__":
    main()
