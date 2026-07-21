"""EEG-only decoder vs BOLD, one feedback run per subject, BOTH cohorts.
DMNELF: CEN/DMN/PDA (local BOLD has _dmn).  rtBPD: CEN only (local BOLD lacks _dmn)."""
import sys, warnings
from pathlib import Path
import numpy as np
from scipy.stats import pearsonr
warnings.filterwarnings("ignore")

FP = Path("/Users/cccbauer/Documents/GitHub/dmnelf/analysis/fingerprint"); NF = FP/"mindwear"
DATA = Path("/Users/cccbauer/Documents/GitHub/dmnelf/data")
sys.path.insert(0, str(NF)); sys.path.insert(0, str(FP/"efp_meirhasson"/"scripts"))
from stockwell import stockwell_power
import mne
CEN_DIR = FP/"fsnr_eeg/results/cen_ceiling"; BASELINE_TR,HRF_DROP=25,5

m=np.load(NF/"model"/"efp_epoc_model.npz",allow_pickle=True)
chans=list(m["channels"]); n_bands=int(m["n_bands"]); n_delays=int(m["n_delays"]); tr=float(m["tr"])
fmin=int(m["fmin"]); fmax=int(m["fmax"]); band_edges=np.asarray(m["band_edges_hz"])
cc=np.asarray(m["cen_coef"],float); cb=float(m["cen_intercept"]); dc=np.asarray(m["dmn_coef"],float); db=float(m["dmn_intercept"])
cfm=np.asarray(m["cen_feat_mean"],float); cfs=np.asarray(m["cen_feat_std"],float)
dfm=np.asarray(m["dmn_feat_mean"],float); dfs=np.asarray(m["dmn_feat_std"],float)
SF=250.0; N_TR=int(round(SF*tr)); NF_=cc.size

# cohort -> (eeg session dir label, subjects from local BOLD)
def subs_for(prefix):
    return sorted(p.name.split(f"cenmean_{prefix}_")[1].split(".npz")[0]
                  for p in CEN_DIR.glob(f"cenmean_{prefix}_*.npz"))
COH = {
    "dmnelf": dict(ses="ses-dmnelf", subs=subs_for("dmnelf"), has_dmn=True),
    "rtbpd":  dict(ses="ses-nf",     subs=subs_for("rtbpd"),  has_dmn=False),
}

def eeg_path(cohort, sub, ses):
    root = DATA/("DMNELF" if cohort=="dmnelf" else "rtBPD")/"derivatives/eeg_preprocessed"
    return root/f"sub-{sub}/{ses}/eeg/sub-{sub}_{ses}_task-feedback_run-01_desc-preproc500Hz_eeg.fif"

def load(f):
    if not f.exists(): return None
    raw=mne.io.read_raw_fif(str(f),preload=True,verbose="ERROR"); raw.pick(mne.pick_types(raw.info,eeg=True,exclude=[]))
    if abs(raw.info["sfreq"]-SF)>1e-3: raw.resample(SF,verbose="ERROR")
    if any(c not in raw.ch_names for c in chans): return None
    return raw.get_data()[[raw.ch_names.index(c) for c in chans]]*1e6

def decode(data):
    nvol=data.shape[1]//N_TR
    bps=[]
    for t in range(nvol):
        w=data[:,t*N_TR:(t+1)*N_TR]; w=w-w.mean(0,keepdims=True); bp=np.empty((len(chans),n_bands))
        for ci in range(len(chans)):
            fr,po=stockwell_power(w[ci],SF,fmin,fmax)
            for bi,(lo,hi) in enumerate(band_edges):
                mm=(fr>=lo)&(fr<=hi); bp[ci,bi]=po[mm].mean() if mm.any() else 0.0
        bps.append(bp)
    mean=np.zeros(NF_); var=np.ones(NF_); a0=1-0.5**(1/60); ring=[]; cen=[]; dmn=[]; cnt=0
    for t in range(nvol):
        ring.append(bps[t]); ring[:]=ring[-n_delays:]
        if len(ring)<n_delays: continue
        des=np.concatenate([ring[-1-d][ci] for ci in range(len(chans)) for d in range(n_delays)])
        cnt+=1; a=max(a0,1/cnt); d=des-mean; mean=mean+a*d; var=(1-a)*(var+a*d*d)
        z=(des-mean)/(np.sqrt(var)+1e-9)
        if cnt<NF_//100: cen.append(np.nan); dmn.append(np.nan); continue
        cen.append(((z-cfm)/cfs)@cc+cb); dmn.append(((z-dfm)/dfs)@dc+db)
    return np.array(cen),np.array(dmn),np.arange(len(cen))+(n_delays-1)

def corr(o,idx,t):
    mm=(idx>=BASELINE_TR+HRF_DROP)&(idx<len(t))&np.isfinite(o)
    return pearsonr(o[mm],t[idx[mm]])[0] if mm.sum()>=15 else np.nan

for cohort,cfg in COH.items():
    print(f"\n================  {cohort.upper()}  (run-01, EEG-only vs BOLD)  ================")
    agg={"CEN":[],"DMN":[],"PDA":[]}
    for sub in cfg["subs"]:
        f=eeg_path(cohort,sub,cfg["ses"])
        data=load(f)
        if data is None:
            print(f"  {sub:>11}  (no EEG / channels)"); continue
        z=np.load(CEN_DIR/f"cenmean_{cohort}_{sub}.npz")
        if "run1" not in z: print(f"  {sub:>11}  (no BOLD run1)"); continue
        oc=np.asarray(z["run1"],float)
        cen,dmn,idx=decode(data)
        rc=corr(cen,idx,oc); agg["CEN"].append(rc)
        if cfg["has_dmn"] and "run1_dmn" in z:
            od=np.asarray(z["run1_dmn"],float); op=oc-od
            rd=corr(dmn,idx,od); rp=corr(cen-dmn,idx,op); agg["DMN"].append(rd); agg["PDA"].append(rp)
            print(f"  {sub:>11}  CEN={rc:+.2f}  DMN={rd:+.2f}  PDA={rp:+.2f}")
        else:
            print(f"  {sub:>11}  CEN={rc:+.2f}   (DMN/PDA: no BOLD DMN target)")
    print(f"  {'-'*40}")
    for t in ["CEN","DMN","PDA"]:
        a=np.array(agg[t],float); a=a[np.isfinite(a)]
        if len(a): print(f"  mean {t}: {a.mean():+.3f} ± {a.std()/np.sqrt(len(a)):.3f}  (n={len(a)}, range {a.min():+.2f}..{a.max():+.2f})")
