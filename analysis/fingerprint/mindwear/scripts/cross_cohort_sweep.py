"""Cross-cohort validation: does the 2-TR online analysis window beat 1-TR against BOLD,
across all DMNELF subjects x feedback runs? Mirrors the EFP pipeline's EEG loading
(desc-preproc500Hz -> resample 250 Hz) and the online decoder (running-z + frozen ridge)."""
import sys, glob, warnings
from pathlib import Path
import numpy as np
from scipy.stats import pearsonr
warnings.filterwarnings("ignore")

FP = Path("/Users/cccbauer/Documents/GitHub/dmnelf/analysis/fingerprint")
NF = FP / "mindwear"
sys.path.insert(0, str(NF)); sys.path.insert(0, str(FP / "efp_meirhasson" / "scripts"))
from stockwell import stockwell_power
import mne

EEG_DIR = Path.home() / "Documents/GitHub/dmnelf/data/DMNELF/derivatives/eeg_preprocessed"
SUBS = ["dmnelf001","dmnelf004","dmnelf005","dmnelf006","dmnelf007","dmnelf008","dmnelf009",
        "dmnelf010","dmnelf011","dmnelf012","dmnelf013","dmnelf014","dmnelf015","dmnelf016",
        "dmnelf1001","dmnelf1002","dmnelf1003"]
RUNS = [1,2,3,4]
BASELINE_TR, HRF_DROP = 25, 5
WINDOWS = [1, 2, 3]

m = np.load(NF/"model"/"efp_epoc_model.npz", allow_pickle=True)
chans=list(m["channels"]); n_bands=int(m["n_bands"]); n_delays=int(m["n_delays"]); tr=float(m["tr"])
fmin=int(m["fmin"]); fmax=int(m["fmax"]); band_edges=np.asarray(m["band_edges_hz"])
cen_coef=np.asarray(m["cen_coef"],float); cen_b=float(m["cen_intercept"])
dmn_coef=np.asarray(m["dmn_coef"],float); dmn_b=float(m["dmn_intercept"])
cen_fm=np.asarray(m["cen_feat_mean"],float); cen_fs=np.asarray(m["cen_feat_std"],float)
dmn_fm=np.asarray(m["dmn_feat_mean"],float); dmn_fs=np.asarray(m["dmn_feat_std"],float)
SF=250.0; N_TR=int(round(SF*tr))

def load_eeg(sub, run):
    f=EEG_DIR/f"sub-{sub}/ses-dmnelf/eeg/sub-{sub}_ses-dmnelf_task-feedback_run-{run:02d}_desc-preproc500Hz_eeg.fif"
    if not f.exists(): return None
    raw=mne.io.read_raw_fif(str(f),preload=True,verbose="ERROR")
    raw.pick(mne.pick_types(raw.info,eeg=True,exclude=[]))
    if abs(raw.info["sfreq"]-SF)>1e-3: raw.resample(SF,verbose="ERROR")
    if any(c not in raw.ch_names for c in chans): return None
    idx=[raw.ch_names.index(c) for c in chans]
    return raw.get_data()[idx]*1e6   # [12, n_samp] uV, in model-channel order

def bp_win(win):
    win=win-win.mean(0,keepdims=True)
    bp=np.empty((win.shape[0],n_bands))
    for ci in range(win.shape[0]):
        fr,po=stockwell_power(win[ci],SF,fmin,fmax)
        for bi,(lo,hi) in enumerate(band_edges):
            mm=(fr>=lo)&(fr<=hi); bp[ci,bi]=po[mm].mean() if mm.any() else 0.0
    return bp

def bp_series(data, L):
    nvol=data.shape[1]//N_TR
    return np.array([bp_win(data[:, max(0,(t+1)*N_TR-L*N_TR):(t+1)*N_TR]) for t in range(nvol)])

def decode(bps):
    nf=cen_coef.size; mean=np.zeros(nf); var=np.ones(nf); a0=1-0.5**(1/60)
    ring=[]; cen=[]; dmn=[]; cnt=0
    for t in range(len(bps)):
        ring.append(bps[t]);  ring[:] = ring[-n_delays:]
        if len(ring)<n_delays: continue
        design=np.concatenate([ring[-1-d][ci] for ci in range(len(chans)) for d in range(n_delays)])
        cnt+=1; a=max(a0,1/cnt); d=design-mean; mean=mean+a*d; var=(1-a)*(var+a*d*d)
        z=(design-mean)/(np.sqrt(var)+1e-9)
        if cnt<nf//100: cen.append(np.nan); dmn.append(np.nan); continue
        cen.append(float(((z-cen_fm)/cen_fs)@cen_coef+cen_b))
        dmn.append(float(((z-dmn_fm)/dmn_fs)@dmn_coef+dmn_b))
    return np.array(cen), np.array(dmn)

def corr(online, target):
    tr0=n_delays-1; idx=np.arange(len(online))+tr0
    mm=(idx>=BASELINE_TR+HRF_DROP)&(idx<len(target))&np.isfinite(online)
    if mm.sum()<15: return np.nan
    return pearsonr(online[mm], target[idx[mm]])[0]

rows={w:{"CEN":[],"DMN":[],"PDA":[]} for w in WINDOWS}
per_run=[]
for sub in SUBS:
    z=np.load(FP/"fsnr_eeg/results/cen_ceiling"/f"cenmean_dmnelf_{sub}.npz")
    for run in RUNS:
        if f"run{run}" not in z or f"run{run}_dmn" not in z: continue
        data=load_eeg(sub,run)
        if data is None: continue
        oc=np.asarray(z[f"run{run}"],float); od=np.asarray(z[f"run{run}_dmn"],float)
        obs={"CEN":oc,"DMN":od,"PDA":oc-od}
        rec={"sub":sub,"run":run}
        for w in WINDOWS:
            cen,dmn=decode(bp_series(data,w)); pda=cen-dmn
            for t,o in [("CEN",cen),("DMN",dmn),("PDA",pda)]:
                r=corr(o,obs[t]); rows[w][t].append(r); rec[f"{t}{w}"]=r
        per_run.append(rec)
        print(f"  {sub} run{run}: "+"  ".join(
            f"w{w} PDA={rec.get(f'PDA{w}',float('nan')):+.2f}" for w in WINDOWS))

print(f"\n=== SUMMARY over {len(per_run)} subject-runs (DMNELF) ===")
print(f"{'window':>7} | {'CEN':>14} {'DMN':>14} {'PDA':>14}")
for w in WINDOWS:
    def stat(t):
        a=np.array(rows[w][t],float); a=a[np.isfinite(a)]
        return f"{a.mean():+.3f}±{a.std()/np.sqrt(len(a)):.3f}"
    print(f"{w:>5}TR | {stat('CEN'):>14} {stat('DMN'):>14} {stat('PDA'):>14}")

# paired 2-TR vs 1-TR
for t in ["CEN","DMN","PDA"]:
    a1=np.array([r.get(f"{t}1",np.nan) for r in per_run],float)
    a2=np.array([r.get(f"{t}2",np.nan) for r in per_run],float)
    g=np.isfinite(a1)&np.isfinite(a2); d=a2[g]-a1[g]
    from scipy.stats import wilcoxon
    try: _,p=wilcoxon(a2[g],a1[g])
    except Exception: p=np.nan
    print(f"  {t}: 2TR-1TR mean Δr={d.mean():+.3f}  improved {int((d>0).sum())}/{g.sum()}  p={p:.4f}")
np.savez(NF/"model"/"cross_cohort_window_sweep.npz",
         per_run=per_run, windows=WINDOWS, allow_pickle=True)
print("\nsaved -> model/cross_cohort_window_sweep.npz")
