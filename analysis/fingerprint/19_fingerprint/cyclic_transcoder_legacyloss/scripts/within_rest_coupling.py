"""Within-rest infraslow->PDA coupling: per-channel x lag cross-correlation.
Assumption-light test (no decoder): does any EEG channel correlate with rest PDA
at any EEG-leads-BOLD lag? Compares baseline (1-40Hz) vs infraslow (0.01-40Hz)."""
import warnings, numpy as np, yaml, mne
from pathlib import Path
from scipy.stats import pearsonr
warnings.simplefilter("ignore"); mne.set_log_level("ERROR")

SUBJ="dmnelf007"; CT=Path("/projects/swglab/data/DMNELF/analysis/fingerprint/cyclic_transcoder")
cfg=yaml.safe_load(open(CT/"config.yaml"))
fdir=Path(cfg["data"]["features_dir_cluster"])/f"sub-{SUBJ}"
eroot=Path(cfg["data"]["eeg_preproc_dir"]); ses=cfg["data"]["session"]; spt=cfg["data"]["eeg"]["samples_per_tr"]

def blockmean(raw,n):
    x=raw.get_data(picks="eeg"); x=x[:,:n*spt].reshape(x.shape[0],n,spt).mean(2).T
    return (x-x.mean(0))/ (x.std(0)+1e-8)

def isfif(task,run): return eroot/f"sub-{SUBJ}"/ses/"eeg"/f"sub-{SUBJ}_{ses}_task-{task}_run-{run}_desc-preproc500HzISp01_eeg.fif"

PDA=[]; XB=[]; XI=[]
for npz in sorted(fdir.glob(f"sub-{SUBJ}_task-rest_run-*_features.npz")):
    d=np.load(npz,allow_pickle=True); pda=np.asarray(d["pda"],float); n=len(pda)
    run=npz.name.split("run-")[1][0]
    raw=mne.io.read_raw_fif(str(isfif("rest",f"{int(run):02d}")),preload=True,verbose=False)
    PDA.append(pda); XB.append(np.asarray(d["eeg_block"],float)); XI.append(blockmean(raw,n))
chs=raw.ch_names[:XB[0].shape[1]]

def best_coupling(Xruns):
    # per lag: max |r| over channels, computed per-run then averaged (z) — but
    # simplest: concat runs (lag applied within run to avoid cross-run leakage)
    out={}
    for lag in range(0,18):
        rs=np.zeros(len(chs))
        for c in range(len(chs)):
            xs=[]; ys=[]
            for X,pda in zip(Xruns,PDA):
                if lag>0: xs.append(X[:-lag,c]); ys.append(pda[lag:])
                else:     xs.append(X[:,c]);     ys.append(pda)
            x=np.concatenate(xs); y=np.concatenate(ys)
            rs[c]=pearsonr(x,y)[0]
        out[lag]=rs
    return out

N=sum(len(p) for p in PDA); thr=2/np.sqrt(N)
print(f"{SUBJ} within-rest coupling  (n_TR={N}, |r|>{thr:.3f} ~ p<.05 single test)")
for name,Xr in [("baseline",XB),("infraslow",XI)]:
    out=best_coupling(Xr)
    # best lag = lag with largest max|r|
    best_lag=max(out,key=lambda L: np.max(np.abs(out[L])))
    rs=out[best_lag]; ci=int(np.argmax(np.abs(rs)))
    print(f"\n[{name}] best lag={best_lag}TR ({best_lag*1.2:.1f}s)  "
          f"best ch={chs[ci]} r={rs[ci]:+.3f}")
    # how many channels exceed threshold at best lag
    nsig=int(np.sum(np.abs(rs)>thr))
    print(f"   channels |r|>{thr:.3f} at best lag: {nsig}/{len(chs)}")
    print("   max|r| by lag: "+"  ".join(f"L{L}:{np.max(np.abs(out[L])):.3f}" for L in range(0,18)))
