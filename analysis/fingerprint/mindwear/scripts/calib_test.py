"""EEG-only calibration test: does per-subject normalization from held-out feedback runs beat the
online running-z? Frozen group ridge throughout (no weight refit — EEG-only, no fMRI).

For each subject/run we compare, vs observed BOLD:
  running-z      : adaptive online self-normalization (what the app uses now)
  calib-1run     : fix per-feature mean/std from ONE other run of the same subject
  calib-allother : fix per-feature mean/std from ALL other runs of the same subject
  calib-xsubject : fix mean/std from a DIFFERENT subject's runs (control: is it subject-specific?)
"""
import sys, warnings
from pathlib import Path
import numpy as np
from scipy.stats import pearsonr, wilcoxon
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
RUNS=[1,2,3,4]; BASELINE_TR,HRF_DROP=25,5

m=np.load(NF/"model"/"efp_epoc_model.npz",allow_pickle=True)
chans=list(m["channels"]); n_bands=int(m["n_bands"]); n_delays=int(m["n_delays"]); tr=float(m["tr"])
fmin=int(m["fmin"]); fmax=int(m["fmax"]); band_edges=np.asarray(m["band_edges_hz"])
cc=np.asarray(m["cen_coef"],float); cb=float(m["cen_intercept"])
dc=np.asarray(m["dmn_coef"],float); db=float(m["dmn_intercept"])
cfm=np.asarray(m["cen_feat_mean"],float); cfs=np.asarray(m["cen_feat_std"],float)
dfm=np.asarray(m["dmn_feat_mean"],float); dfs=np.asarray(m["dmn_feat_std"],float)
SF=250.0; N_TR=int(round(SF*tr)); NF_=cc.size

def load_eeg(sub,run):
    f=EEG_DIR/f"sub-{sub}/ses-dmnelf/eeg/sub-{sub}_ses-dmnelf_task-feedback_run-{run:02d}_desc-preproc500Hz_eeg.fif"
    if not f.exists(): return None
    raw=mne.io.read_raw_fif(str(f),preload=True,verbose="ERROR")
    raw.pick(mne.pick_types(raw.info,eeg=True,exclude=[]))
    if abs(raw.info["sfreq"]-SF)>1e-3: raw.resample(SF,verbose="ERROR")
    if any(c not in raw.ch_names for c in chans): return None
    return raw.get_data()[[raw.ch_names.index(c) for c in chans]]*1e6

def bp_win(win):
    win=win-win.mean(0,keepdims=True); bp=np.empty((win.shape[0],n_bands))
    for ci in range(win.shape[0]):
        fr,po=stockwell_power(win[ci],SF,fmin,fmax)
        for bi,(lo,hi) in enumerate(band_edges):
            mm=(fr>=lo)&(fr<=hi); bp[ci,bi]=po[mm].mean() if mm.any() else 0.0
    return bp

def designs(data):
    """[n_emitted, NF_] design matrix (window=1), and run-TR index per row."""
    nvol=data.shape[1]//N_TR
    bps=[bp_win(data[:,t*N_TR:(t+1)*N_TR]) for t in range(nvol)]
    X=[]; idx=[]
    for t in range(nvol):
        if t<n_delays-1: continue
        row=np.concatenate([bps[t-d][ci] for ci in range(len(chans)) for d in range(n_delays)])
        X.append(row); idx.append(t)          # run-TR = t
    return np.array(X), np.array(idx)

def predict(X, mean, std):
    z=(X-mean)/(std+1e-9)
    cen=((z-cfm)/cfs)@cc+cb; dmn=((z-dfm)/dfs)@dc+db
    return cen, dmn

def running_z(X):
    mean=np.zeros(NF_); var=np.ones(NF_); a0=1-0.5**(1/60); cen=[]; dmn=[]
    for k in range(len(X)):
        cnt=k+1; a=max(a0,1/cnt); d=X[k]-mean; mean=mean+a*d; var=(1-a)*(var+a*d*d)
        z=(X[k]-mean)/(np.sqrt(var)+1e-9)
        cen.append(((z-cfm)/cfs)@cc+cb); dmn.append(((z-dfm)/dfs)@dc+db)
    return np.array(cen), np.array(dmn)

def corr(o, idx, target):
    mm=(idx>=BASELINE_TR+HRF_DROP)&(idx<len(target))&np.isfinite(o)
    if mm.sum()<15: return np.nan
    return pearsonr(o[mm], target[idx[mm]])[0]

# cache designs + targets per subject-run
print("computing designs (window=1) for all subject-runs…")
DAT={}
for sub in SUBS:
    z=np.load(FP/"fsnr_eeg/results/cen_ceiling"/f"cenmean_dmnelf_{sub}.npz")
    for run in RUNS:
        if f"run{run}" not in z: continue
        data=load_eeg(sub,run)
        if data is None: continue
        X,idx=designs(data)
        oc=np.asarray(z[f"run{run}"],float); od=np.asarray(z[f"run{run}_dmn"],float)
        DAT[(sub,run)]={"X":X,"idx":idx,"CEN":oc,"DMN":od,"PDA":oc-od}
    print(f"  {sub}: {sum(1 for r in RUNS if (sub,r) in DAT)} runs")

methods=["running-z","calib-1run","calib-allother","calib-xsubject"]
res={mth:{"CEN":[],"DMN":[],"PDA":[]} for mth in methods}
paired={"CEN":[],"DMN":[],"PDA":[]}   # (running-z, calib-allother) per test run

subs_present=sorted({s for (s,r) in DAT})
for sub in subs_present:
    runs=[r for r in RUNS if (sub,r) in DAT]
    others_sub=[s for s in subs_present if s!=sub]
    for t in runs:
        D=DAT[(sub,t)]; X=D["X"]; idx=D["idx"]
        # running-z
        rc,rd=running_z(X); rp=rc-rd
        # calib from all other runs of same subject
        other=[DAT[(sub,o)]["X"] for o in runs if o!=t]
        Xo=np.vstack(other) if other else X
        m_all,s_all=Xo.mean(0),Xo.std(0)
        ac,ad=predict(X,m_all,s_all); ap=ac-ad
        # calib from ONE other run (first available)
        one=[o for o in runs if o!=t]
        if one:
            X1=DAT[(sub,one[0])]["X"]; ac1,ad1=predict(X,X1.mean(0),X1.std(0)); ap1=ac1-ad1
        else:
            ac1,ad1,ap1=ac,ad,ap
        # cross-subject calib (control): a different subject's run1 if available
        xs=next((os for os in others_sub if (os,1) in DAT), None)
        if xs:
            Xx=DAT[(xs,1)]["X"]; xc,xd=predict(X,Xx.mean(0),Xx.std(0)); xp=xc-xd
        else:
            xc,xd,xp=ac,ad,ap
        for tt,(o_rz,o_a1,o_all,o_xs) in {"CEN":(rc,ac1,ac,xc),"DMN":(rd,ad1,ad,xd),"PDA":(rp,ap1,ap,xp)}.items():
            res["running-z"][tt].append(corr(o_rz,idx,D[tt]))
            res["calib-1run"][tt].append(corr(o_a1,idx,D[tt]))
            res["calib-allother"][tt].append(corr(o_all,idx,D[tt]))
            res["calib-xsubject"][tt].append(corr(o_xs,idx,D[tt]))
            paired[tt].append((corr(o_rz,idx,D[tt]), corr(o_all,idx,D[tt])))

n=len(res["running-z"]["PDA"])
print(f"\n=== EEG-only calibration, {n} held-out subject-runs (frozen group ridge) ===")
print(f"{'method':>16} | {'CEN':>7} {'DMN':>7} {'PDA':>7}")
for mth in methods:
    def mean(tt):
        a=np.array(res[mth][tt],float); a=a[np.isfinite(a)]; return a.mean()
    print(f"{mth:>16} | {mean('CEN'):>+7.3f} {mean('DMN'):>+7.3f} {mean('PDA'):>+7.3f}")
print("\ncalib-allother vs running-z (paired):")
for tt in ["CEN","DMN","PDA"]:
    P=np.array(paired[tt],float); g=np.isfinite(P).all(1); a=P[g]
    d=a[:,1]-a[:,0]
    try: _,p=wilcoxon(a[:,1],a[:,0])
    except Exception: p=np.nan
    print(f"  {tt}: Δr={d.mean():+.3f}  improved {int((d>0).sum())}/{len(d)}  p={p:.4f}")
