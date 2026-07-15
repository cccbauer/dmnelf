#!/usr/bin/env python3
"""
train.py  (cluster, GPU, env r33_fixed)  —  R-EEGNet training with modes + aug levels
-------------------------------------------------------------------------------------
--mode loso : cross-subject LOSO on DMNELF (+ frozen -> rtBPD transfer). The user's regime.
--mode loro : within-SUBJECT leave-one-run-out (the paper's easier subject-specific regime).
--aug full  : mixup + 0.1 noise, wd 1e-3, dropout 0.5   (original)
--aug light : no mixup, 0.02 noise, wd 1e-4, dropout 0.25 (ablation for weak signal)
Clean CEN/DMN targets, feedback block. vs EFP (~0.11 within, ~0.07 transfer).
"""
import argparse, glob, re
from pathlib import Path
import numpy as np
import torch, torch.nn as nn
from scipy.stats import pearsonr
from model import REEGNet

DEV = "cuda" if torch.cuda.is_available() else "cpu"


def load(win_dir, prefix, band):
    out = {}
    for f in sorted(glob.glob(f"{win_dir}/{prefix}*_windows{band}.npz")):
        sub = re.search(rf"({prefix}\w+?)_windows", Path(f).name).group(1)
        if re.search(r"dmnelf999", sub):
            continue
        z = np.load(f); X = z["X"].astype(np.float32)
        X = (X - X.mean(2, keepdims=True)) / (X.std(2, keepdims=True) + 1e-6)
        out[sub] = (X, np.stack([z["y_cen"], z["y_dmn"]], 1).astype(np.float32), z["run"])
    return out


def train_model(Xtr, Ytr, Xva, Yva, aug, epochs=60, bs=64, lr=1e-3, patience=12):
    p = 0.25 if aug == "light" else 0.5
    wd = 1e-4 if aug == "light" else 1e-3
    noise = 0.02 if aug == "light" else 0.1
    mixup = (aug == "full")
    m = REEGNet(p=p).to(DEV); opt = torch.optim.Adam(m.parameters(), lr=lr, weight_decay=wd)
    lossf = nn.MSELoss()
    Xtr = torch.tensor(Xtr, device=DEV); Ytr = torch.tensor(Ytr, device=DEV); Xva = torch.tensor(Xva, device=DEV)
    best, best_state, wait, n = -1e9, None, 0, len(Xtr)
    for ep in range(epochs):
        m.train(); perm = torch.randperm(n, device=DEV)
        for i in range(0, n, bs):
            idx = perm[i:i + bs]; xb, yb = Xtr[idx], Ytr[idx]
            xb = xb + noise * torch.randn_like(xb)
            if mixup and len(xb) > 1:
                lam = float(torch.rand(1)); j = torch.randperm(len(xb), device=DEV)
                xb = lam * xb + (1 - lam) * xb[j]; yb = lam * yb + (1 - lam) * yb[j]
            opt.zero_grad(); lossf(m(xb), yb).backward(); opt.step()
        m.eval()
        with torch.no_grad():
            pv = m(Xva).cpu().numpy()
        vr = np.nanmean([pearsonr(Yva[:, k], pv[:, k])[0] for k in range(2)])
        if vr > best:
            best, best_state, wait = vr, {k: v.clone() for k, v in m.state_dict().items()}, 0
        else:
            wait += 1
            if wait >= patience:
                break
    m.load_state_dict(best_state); m.eval()
    return m


def predict(m, X):
    with torch.no_grad():
        return m(torch.tensor(X, device=DEV)).cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--win-dir", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--band", default=""); ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--mode", default="loso", choices=["loso", "loro"])
    ap.add_argument("--aug", default="full", choices=["full", "light"])
    a = ap.parse_args(); out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(0); np.random.seed(0)
    D = load(a.win_dir, "dmnelf", a.band); subs = list(D)
    print(f"device={DEV} | DMNELF n={len(subs)} | mode={a.mode} aug={a.aug} band='{a.band or 'full'}'", flush=True)

    if a.mode == "loro":   # within-subject leave-one-run-out
        rows = []
        for s in subs:
            X, Y, run = D[s]; runs = sorted(set(run.tolist()))
            if len(runs) < 3:
                continue
            oc, oy = [], []
            for i, r in enumerate(runs):
                te = run == r; va = run == runs[(i + 1) % len(runs)]; trm = ~te & ~va
                if trm.sum() < 40 or te.sum() < 20:
                    continue
                m = train_model(X[trm], Y[trm], X[va], Y[va], a.aug, epochs=a.epochs)
                oc.append((Y[te], predict(m, X[te])))
            if not oc:
                continue
            yt = np.concatenate([o[0] for o in oc]); pt = np.concatenate([o[1] for o in oc])
            rc = pearsonr(yt[:, 0], pt[:, 0])[0]; rd = pearsonr(yt[:, 1], pt[:, 1])[0]
            rows.append((rc, rd)); print(f"  LORO {s}: CEN={rc:+.3f} DMN={rd:+.3f}", flush=True)
        rc = np.array([r[0] for r in rows]); rd = np.array([r[1] for r in rows])
        print(f"\nDMNELF within-subj LORO ({a.aug})  CEN={np.nanmean(rc):+.3f}  DMN={np.nanmean(rd):+.3f}", flush=True)
        return

    # mode == loso (cross-subject) + transfer
    rows = []
    for i, s in enumerate(subs):
        val = subs[(i + 1) % len(subs)]; tr = [u for u in subs if u not in (s, val)]
        m = train_model(np.concatenate([D[u][0] for u in tr]), np.concatenate([D[u][1] for u in tr]),
                        D[val][0], D[val][1], a.aug, epochs=a.epochs)
        p = predict(m, D[s][0])
        rows.append((pearsonr(D[s][1][:, 0], p[:, 0])[0], pearsonr(D[s][1][:, 1], p[:, 1])[0]))
    rc = np.array([r[0] for r in rows]); rd = np.array([r[1] for r in rows])
    print(f"\nDMNELF LOSO ({a.aug})  CEN={np.nanmean(rc):+.3f}  DMN={np.nanmean(rd):+.3f}", flush=True)
    va = subs[0]; tr = subs[1:]
    mfull = train_model(np.concatenate([D[u][0] for u in tr]), np.concatenate([D[u][1] for u in tr]),
                        D[va][0], D[va][1], a.aug, epochs=a.epochs)
    for coh, pfx, wd in [("rtbpd_nf1", "rtbpd", a.win_dir), ("rtbpd_nf2", "rtbpd", a.win_dir + "_nf2")]:
        T = load(wd, pfx, a.band); tr2 = [(pearsonr(Y[:, 0], predict(mfull, X)[:, 0])[0],
                                           pearsonr(Y[:, 1], predict(mfull, X)[:, 1])[0]) for X, Y, _ in T.values()]
        if tr2:
            print(f"transfer {coh} ({a.aug}): CEN={np.nanmean([t[0] for t in tr2]):+.3f} "
                  f"DMN={np.nanmean([t[1] for t in tr2]):+.3f} (n={len(tr2)})", flush=True)


if __name__ == "__main__":
    main()
