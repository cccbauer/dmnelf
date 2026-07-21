"""Test the EEG-only decoder + bad-channel QC against our paired fMRI/EEG data.
(1) Per-subject clean decoder performance (which subjects the EEG-only system tracks).
(2) Robustness: simulate 2 dead EPOC sensors, decode WITHOUT QC vs WITH QC (CAR-over-good +
    neutral fill, exactly as RTFeatureExtractor now does), and see if QC recovers the loss."""
import sys, warnings
from pathlib import Path
import numpy as np
from scipy.stats import pearsonr
warnings.filterwarnings("ignore")

FP = Path("/Users/cccbauer/Documents/GitHub/dmnelf/analysis/fingerprint"); NF = FP/"mindwear"
sys.path.insert(0, str(NF)); sys.path.insert(0, str(FP/"efp_meirhasson"/"scripts"))
from stockwell import stockwell_power
import mne
EEG_DIR = Path.home()/"Documents/GitHub/dmnelf/data/DMNELF/derivatives/eeg_preprocessed"
SUBS = ["dmnelf001","dmnelf004","dmnelf005","dmnelf006","dmnelf007","dmnelf008","dmnelf009",
        "dmnelf010","dmnelf011","dmnelf012","dmnelf013","dmnelf014","dmnelf015","dmnelf1001",
        "dmnelf1002","dmnelf1003"]
RUNS=[1,2,3,4]; BASELINE_TR,HRF_DROP=25,5
m=np.load(NF/"model"/"efp_epoc_model.npz",allow_pickle=True)
chans=list(m["channels"]); n_bands=int(m["n_bands"]); n_delays=int(m["n_delays"]); tr=float(m["tr"])
fmin=int(m["fmin"]); fmax=int(m["fmax"]); band_edges=np.asarray(m["band_edges_hz"])
cc=np.asarray(m["cen_coef"],float); cb=float(m["cen_intercept"]); dc=np.asarray(m["dmn_coef"],float); db=float(m["dmn_intercept"])
cfm=np.asarray(m["cen_feat_mean"],float); cfs=np.asarray(m["cen_feat_std"],float)
dfm=np.asarray(m["dmn_feat_mean"],float); dfs=np.asarray(m["dmn_feat_std"],float)
SF=250.0; N_TR=int(round(SF*tr)); NF_=cc.size
BAD_SIM=["T7","T8"]; bad_idx=[chans.index(c) for c in BAD_SIM]

def load(sub,run):
    f=EEG_DIR/f"sub-{sub}/ses-dmnelf/eeg/sub-{sub}_ses-dmnelf_task-feedback_run-{run:02d}_desc-preproc500Hz_eeg.fif"
    if not f.exists(): return None
    raw=mne.io.read_raw_fif(str(f),preload=True,verbose="ERROR"); raw.pick(mne.pick_types(raw.info,eeg=True,exclude=[]))
    if abs(raw.info["sfreq"]-SF)>1e-3: raw.resample(SF,verbose="ERROR")
    if any(c not in raw.ch_names for c in chans): return None
    return raw.get_data()[[raw.ch_names.index(c) for c in chans]]*1e6

def bp_win(win, bad_mask):
    good=~bad_mask; win=win-win[good].mean(0,keepdims=True); bp=np.empty((win.shape[0],n_bands))
    for ci in range(win.shape[0]):
        if bad_mask[ci]: continue
        fr,po=stockwell_power(win[ci],SF,fmin,fmax)
        for bi,(lo,hi) in enumerate(band_edges):
            mm=(fr>=lo)&(fr<=hi); bp[ci,bi]=po[mm].mean() if mm.any() else 0.0
    if bad_mask.any(): bp[bad_mask]=bp[good].mean(0)
    return bp

def decode(data, bad_mask):
    nvol=data.shape[1]//N_TR; bps=[bp_win(data[:,t*N_TR:(t+1)*N_TR],bad_mask) for t in range(nvol)]
    mean=np.zeros(NF_); var=np.ones(NF_); a0=1-0.5**(1/60); ring=[]; cen=[]; dmn=[]; cnt=0
    for t in range(nvol):
        ring.append(bps[t]); ring[:]=ring[-n_delays:]
        if len(ring)<n_delays: continue
        design=np.concatenate([ring[-1-d][ci] for ci in range(len(chans)) for d in range(n_delays)])
        cnt+=1; a=max(a0,1/cnt); d=design-mean; mean=mean+a*d; var=(1-a)*(var+a*d*d)
        z=(design-mean)/(np.sqrt(var)+1e-9)
        if cnt<NF_//100: cen.append(np.nan); dmn.append(np.nan); continue
        cen.append(((z-cfm)/cfs)@cc+cb); dmn.append(((z-dfm)/dfs)@dc+db)
    idx=np.arange(len(cen))+(n_delays-1)
    return np.array(cen),np.array(dmn),idx

def corr(o,idx,target):
    mm=(idx>=BASELINE_TR+HRF_DROP)&(idx<len(target))&np.isfinite(o)
    return pearsonr(o[mm],target[idx[mm]])[0] if mm.sum()>=15 else np.nan

no_bad=np.zeros(len(chans),bool); with_bad=np.zeros(len(chans),bool); with_bad[bad_idx]=True
persub={}; clean_all={"CEN":[],"DMN":[],"PDA":[]}; rob=[]
print(f"per-subject EEG-only decoder (clean) + dead-sensor robustness (killed {BAD_SIM})\n")
print(f"{'subject':>11} {'CEN':>6} {'DMN':>6} {'PDA':>6} | {'PDA noQC':>8} {'PDA QC':>7}")
for sub in SUBS:
    z=np.load(FP/"fsnr_eeg/results/cen_ceiling"/f"cenmean_dmnelf_{sub}.npz")
    rc=[]; rd=[]; rp=[]; rp_noqc=[]; rp_qc=[]
    for run in RUNS:
        if f"run{run}" not in z: continue
        data=load(sub,run)
        if data is None: continue
        oc=np.asarray(z[f"run{run}"],float); od=np.asarray(z[f"run{run}_dmn"],float); op=oc-od
        cen,dmn,idx=decode(data,no_bad); rc.append(corr(cen,idx,oc)); rd.append(corr(dmn,idx,od)); rp.append(corr(cen-dmn,idx,op))
        # simulate 2 dead sensors: flatten to ~0.1 uV noise (deterministic per run via index)
        dead=data.copy(); rng=np.arange(data.shape[1])
        for bi in bad_idx: dead[bi]=0.1*np.sin(0.01*rng+bi)      # tiny flat-ish signal
        cn,dn,idn=decode(dead,no_bad); rp_noqc.append(corr(cn-dn,idn,op))     # no QC (channel corrupts CAR)
        cq,dq,idq=decode(dead,with_bad); rp_qc.append(corr(cq-dq,idq,op))     # QC: excluded + neutral fill
    if not rp: continue
    mc,md,mp=np.nanmean(rc),np.nanmean(rd),np.nanmean(rp)
    mnoqc,mqc=np.nanmean(rp_noqc),np.nanmean(rp_qc)
    for t,v in [("CEN",rc),("DMN",rd),("PDA",rp)]: clean_all[t]+=list(v)
    rob.append((mp,mnoqc,mqc))
    print(f"{sub:>11} {mc:>+6.2f} {md:>+6.2f} {mp:>+6.2f} | {mnoqc:>+8.2f} {mqc:>+7.2f}")

print(f"\n=== cohort (clean, {len(clean_all['PDA'])} runs) ===")
for t in ["CEN","DMN","PDA"]:
    a=np.array(clean_all[t]); a=a[np.isfinite(a)]
    print(f"  {t}: mean r = {a.mean():+.3f} ± {a.std()/np.sqrt(len(a)):.3f}   (range {a.min():+.2f}..{a.max():+.2f})")
R=np.array(rob)
print(f"\n=== dead-sensor robustness (2 channels killed), PDA, per-subject means ===")
print(f"  clean      : {R[:,0].mean():+.3f}")
print(f"  killed,noQC: {R[:,1].mean():+.3f}   (loss {R[:,1].mean()-R[:,0].mean():+.3f})")
print(f"  killed, QC : {R[:,2].mean():+.3f}   (loss {R[:,2].mean()-R[:,0].mean():+.3f})")
print(f"  QC recovers {100*(R[:,2].mean()-R[:,1].mean())/max(1e-9,R[:,0].mean()-R[:,1].mean()):.0f}% of the dead-sensor loss")
