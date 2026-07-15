#!/usr/bin/env python3
"""
efp_electrode_map.py  (cluster)  —  per-electrode EFP decodability, for topomaps
--------------------------------------------------------------------------------
For each target (CEN/DMN/PDA), compute how well EACH of the 31 electrodes alone decodes
it, DMNELF within-cohort LOSO (feedback block). Outputs a per-electrode r table so we can
render scalp topomaps of the fingerprint (focal vs distributed; where each network lives).
Also per-electrode cross-cohort transfer r to rtBPD nf1/nf2 (train each electrode on all
DMNELF, apply unchanged).

Output: efp_electrode_map.csv  (target, ch_idx, ch_name, dmnelf_loso_r, nf1_r, nf2_r)
"""
import argparse
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import pearsonr
from sklearn.linear_model import RidgeCV
from efp_features import load_config, load_subject_features
from efp_transfer import subjects, subj_designs, TRAIN_CACHE as TRAIN, TEST  # CLEAN targets

ALPHAS = np.logspace(-2, 5, 15)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True)
    a = ap.parse_args(); out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    cfg = load_config(); n_delays = int(round(cfg["efp"]["delay_window_s"] / cfg["data"]["fmri"]["tr"])) + 1
    tr_subs = subjects(TRAIN)
    rows = []
    for target in ["CEN", "DMN", "PDA"]:
        D = {s: subj_designs(TRAIN, s, target, n_delays) for s in tr_subs}
        D = {s: v for s, v in D.items() if v}
        nch = next(iter(D.values()))[2]
        _, ch = load_subject_features(Path(TRAIN), next(iter(D)))
        TE = {sess: {s: subj_designs(c, s, target, n_delays) for s in subjects(c)} for sess, c in TEST.items()}
        for ci in range(nch):
            # DMNELF LOSO
            rs = []
            for held in D:
                m = RidgeCV(alphas=ALPHAS).fit(np.vstack([D[s][0][ci] for s in D if s != held]),
                                               np.concatenate([D[s][1] for s in D if s != held]))
                p = m.predict(D[held][0][ci])
                if np.std(p) > 1e-9:
                    rs.append(pearsonr(D[held][1], p)[0])
            # freeze on all DMNELF, transfer
            mfull = RidgeCV(alphas=ALPHAS).fit(np.vstack([D[s][0][ci] for s in D]),
                                               np.concatenate([D[s][1] for s in D]))
            tr = {}
            for sess in TEST:
                rr = []
                for s, v in TE[sess].items():
                    if not v:
                        continue
                    pp = mfull.predict(v[0][ci])
                    if np.std(pp) > 1e-9:
                        rr.append(pearsonr(v[1], pp)[0])
                tr[sess] = float(np.mean(rr)) if rr else np.nan
            rows.append(dict(target=target, ch_idx=ci, ch_name=ch[ci],
                             dmnelf_loso_r=float(np.mean(rs)) if rs else np.nan,
                             nf1_r=tr["nf1"], nf2_r=tr["nf2"]))
        print(f"{target}: done", flush=True)
    pd.DataFrame(rows).to_csv(out / "efp_electrode_map.csv", index=False)
    print("saved efp_electrode_map.csv", flush=True)


if __name__ == "__main__":
    main()
