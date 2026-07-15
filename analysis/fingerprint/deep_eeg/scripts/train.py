#!/usr/bin/env python3
"""
train.py  (cluster, GPU, env r33_fixed)  —  R-EEGNet: DMNELF LOSO (Phase 2a gate) + frozen->rtBPD transfer
---------------------------------------------------------------------------------------------------------
Develop ENTIRELY on DMNELF: leave-one-subject-out (train on others minus 1 val for early stop, predict
held-out) -> honest within-DMNELF r (CEN/DMN). Then train a FINAL model on ALL DMNELF, freeze, apply
UNCHANGED to rtBPD nf1/nf2 (held-out transfer). Inputs standardized per-window/channel; augmentation
(gaussian noise + mixup). vs EFP baseline (CEN ~0.11 within, ~0.07 transfer).

Usage: python train.py --win-dir DIR --out DIR [--band ""|"_lt20"] [--epochs 60]
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
        z = np.load(f)
        X = z["X"].astype(np.float32)
        X = (X - X.mean(2, keepdims=True)) / (X.std(2, keepdims=True) + 1e-6)   # per-window/channel z
        out[sub] = (X, np.stack([z["y_cen"], z["y_dmn"]], 1).astype(np.float32))
    return out


def train_model(Xtr, Ytr, Xva, Yva, epochs=60, bs=64, lr=1e-3, patience=12):
    m = REEGNet().to(DEV); opt = torch.optim.Adam(m.parameters(), lr=lr, weight_decay=1e-3)
    lossf = nn.MSELoss(); Xtr = torch.tensor(Xtr, device=DEV); Ytr = torch.tensor(Ytr, device=DEV)
    Xva = torch.tensor(Xva, device=DEV); best, best_state, wait = -1e9, None, 0
    n = len(Xtr); g = torch.Generator(device=DEV)
    for ep in range(epochs):
        m.train(); perm = torch.randperm(n, generator=g, device=DEV)
        for i in range(0, n, bs):
            idx = perm[i:i + bs]; xb, yb = Xtr[idx], Ytr[idx]
            xb = xb + 0.1 * torch.randn_like(xb)                        # gaussian noise aug
            if len(xb) > 1:                                             # mixup
                lam = float(torch.rand(1)); j = torch.randperm(len(xb), device=DEV)
                xb = lam * xb + (1 - lam) * xb[j]; yb = lam * yb + (1 - lam) * yb[j]
            opt.zero_grad(); loss = lossf(m(xb), yb); loss.backward(); opt.step()
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
    a = ap.parse_args(); out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(0); np.random.seed(0)
    D = load(a.win_dir, "dmnelf", a.band); subs = list(D)
    print(f"device={DEV} | DMNELF n={len(subs)} | band='{a.band or 'full'}'", flush=True)

    # ---- Phase 2a: DMNELF LOSO ----
    rows = []
    for i, s in enumerate(subs):
        val = subs[(i + 1) % len(subs)]
        tr = [u for u in subs if u not in (s, val)]
        Xtr = np.concatenate([D[u][0] for u in tr]); Ytr = np.concatenate([D[u][1] for u in tr])
        m = train_model(Xtr, Ytr, D[val][0], D[val][1], epochs=a.epochs)
        p = predict(m, D[s][0])
        rc = pearsonr(D[s][1][:, 0], p[:, 0])[0]; rd = pearsonr(D[s][1][:, 1], p[:, 1])[0]
        rows.append((s, rc, rd)); print(f"  LOSO {s}: CEN={rc:+.3f} DMN={rd:+.3f}", flush=True)
    rc = np.array([r[1] for r in rows]); rd = np.array([r[2] for r in rows])
    print(f"\nDMNELF LOSO  CEN={np.nanmean(rc):+.3f}  DMN={np.nanmean(rd):+.3f}  (EFP ~0.11/0.10)", flush=True)
    np.savez(out / f"dmnelf_loso{a.band}.npz", subs=subs, rc=rc, rd=rd)

    # ---- Phase 3: FINAL model on ALL DMNELF -> freeze -> rtBPD transfer ----
    va = subs[0]; tr = subs[1:]
    mfull = train_model(np.concatenate([D[u][0] for u in tr]), np.concatenate([D[u][1] for u in tr]),
                        D[va][0], D[va][1], epochs=a.epochs)
    for coh, pfx in [("rtbpd_nf1", "rtbpd"), ("rtbpd_nf2", "rtbpd")]:
        wd = a.win_dir if coh == "rtbpd_nf1" else a.win_dir + "_nf2"
        T = load(wd, pfx, a.band)
        trows = []
        for s, (X, Y) in T.items():
            p = predict(mfull, X)
            trows.append((pearsonr(Y[:, 0], p[:, 0])[0], pearsonr(Y[:, 1], p[:, 1])[0]))
        if trows:
            tc = np.array([t[0] for t in trows]); td = np.array([t[1] for t in trows])
            print(f"transfer {coh}: CEN={np.nanmean(tc):+.3f} DMN={np.nanmean(td):+.3f} (n={len(trows)})", flush=True)
            np.savez(out / f"transfer_{coh}{a.band}.npz", tc=tc, td=td)


if __name__ == "__main__":
    main()
