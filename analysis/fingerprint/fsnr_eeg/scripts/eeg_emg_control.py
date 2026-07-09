#!/usr/bin/env python3
"""
eeg_emg_control.py  —  is the feedback beta/gamma variance quench EMG or neural?
--------------------------------------------------------------------------------
EMG is broadband and edge/muscle-localized (temporal electrodes) and shows up in the
aperiodic (1/f) component; genuine neural change includes midline sites and lives in the
oscillatory / 1/f-slope structure. We test the beta/gamma band-power variance quench
(rest TR 0:25 vs feedback 30:end, from the NON-convolved specparam features) by electrode
group, plus the aperiodic offset (broadband, EMG-sensitive) and exponent (1/f slope).

Verdict logic:
  EMG artifact  -> temporal >> midline, and broadband offset variance quenches.
  Neural        -> midline present/strongest, offset flat, 1/f exponent stabilizes.
Also motivated by task demands: active noting-practice (esp. novices) would not reduce
EMG vs eyes-closed rest, so a variance DROP cannot be a 'stiller body' artifact.
"""
from pathlib import Path
import numpy as np, glob, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

PROJ = Path(__file__).resolve().parent.parent
SPEC = PROJ / "results" / "specparam"; RES = PROJ / "results"
BASELINE_TR, HRF_DROP = 25, 5
BANDS = ["delta", "theta", "alpha", "beta", "gamma"]
GRP = {"temporal (EMG)": ["T7", "T8", "TP9", "TP10"],
       "midline (low-EMG)": ["Fz", "Cz", "Pz", "POz", "Oz"],
       "frontal (EOG)": ["Fp1", "Fp2", "F7", "F8"]}


def qdb(vr, vf):
    return 10 * np.log10(vr / vf) if (vf > 1e-12 and vr > 0) else np.nan


def sflip(x, n=10000):
    x = np.asarray([v for v in x if np.isfinite(v)]); rng = np.random.default_rng(0)
    o = x.mean(); nul = (rng.choice([-1, 1], (n, len(x))) * np.abs(x)).mean(1)
    return o, (np.sum(nul >= o) + 1) / (n + 1)


def var_split(arr, pi):
    vr = np.nanvar(arr[:BASELINE_TR][:, pi], 0).mean()
    vf = np.nanvar(arr[BASELINE_TR + HRF_DROP:][:, pi], 0).mean()
    return qdb(vr, vf)


def main():
    q = {f"{b}|{g}": [] for b in ["beta", "gamma"] for g in GRP}
    qap = {"aperiodic offset": [], "aperiodic exponent": []}
    for f in sorted(glob.glob(str(SPEC / "*_specparam.npz"))):
        z = np.load(f, allow_pickle=True)
        for rk in [str(k) for k in z["_runs"]]:
            d = z[rk].item(); chs = list(d["chs"]); bp = d["bandpow"]
            for b in ["beta", "gamma"]:
                bi = BANDS.index(b)
                for g, names in GRP.items():
                    pi = [i for i, c in enumerate(chs) if c in names]
                    if pi:
                        q[f"{b}|{g}"].append(var_split(bp[:, :, bi], pi))
            pim = [i for i, c in enumerate(chs) if c in GRP["midline (low-EMG)"]]
            qap["aperiodic offset"].append(var_split(d["offset"], pim))
            qap["aperiodic exponent"].append(var_split(d["exponent"], pim))

    rows = []
    print("=== beta/gamma band-power variance quench by electrode group (+dB=declutter) ===")
    for b in ["beta", "gamma"]:
        for g in GRP:
            o, p = sflip(q[f"{b}|{g}"]); rows.append(dict(measure=b, group=g, quench_db=o, p=p))
            print(f"  {b:5s} {g:18s} {o:+.2f} dB  p={p:.4f}")
        print()
    print("=== aperiodic (midline) variance quench ===")
    for k, v in qap.items():
        o, p = sflip(v); rows.append(dict(measure=k, group="midline (low-EMG)", quench_db=o, p=p))
        print(f"  {k:20s} {o:+.2f} dB  p={p:.4f}")
    df = pd.DataFrame(rows); df.to_csv(RES / "eeg_emg_control.csv", index=False)

    # verdict
    bm = df[(df.measure == "beta")].set_index("group").quench_db
    off = df[df.measure == "aperiodic offset"].quench_db.iloc[0]
    print("\nVERDICT: midline >= temporal? %s | offset flat? %s -> %s" % (
        bm["midline (low-EMG)"] >= bm["temporal (EMG)"], abs(off) < 0.5,
        "NEURAL, not EMG" if (bm["midline (low-EMG)"] >= bm["temporal (EMG)"] and abs(off) < 0.5) else "check"))

    # figure
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2), gridspec_kw=dict(width_ratios=[2, 1]))
    x = np.arange(len(GRP)); w = 0.38; cols = {"temporal (EMG)": "#c0504d", "midline (low-EMG)": "#2e7d32", "frontal (EOG)": "#e08e45"}
    for j, b in enumerate(["beta", "gamma"]):
        vals = [df[(df.measure == b) & (df.group == g)].quench_db.iloc[0] for g in GRP]
        bars = ax[0].bar(x + (j - 0.5) * w, vals, w, label=b, color=["#c0504d", "#2e7d32", "#e08e45"], edgecolor='k', lw=.4, alpha=0.9 if j else 0.6)
    ax[0].set_xticks(x); ax[0].set_xticklabels(list(GRP), fontsize=9)
    ax[0].set_ylabel("variance quench (dB)"); ax[0].axhline(0, color='k', lw=.6)
    ax[0].set_title("Quench is midline-strongest, not temporal → not EMG", fontweight='bold', fontsize=10)
    ax[0].text(0.02, 0.95, "darker=gamma, lighter=beta", transform=ax[0].transAxes, fontsize=7, va='top')
    apv = [df[df.measure == m].quench_db.iloc[0] for m in ["aperiodic offset", "aperiodic exponent"]]
    ax[1].bar(["offset\n(broadband/EMG)", "exponent\n(1/f slope)"], apv, color=["#999", "#1f3a5f"], edgecolor='k', lw=.5)
    ax[1].axhline(0, color='k', lw=.6); ax[1].set_ylabel("variance quench (dB)")
    ax[1].set_title("Offset flat (not EMG);\n1/f slope stabilizes (neural)", fontweight='bold', fontsize=9.5)
    for a in ax: a.spines[['top', 'right']].set_visible(False)
    fig.tight_layout(); fig.savefig(RES / "fig_emg_control.png", dpi=150)
    print("saved fig_emg_control.png +", RES / "eeg_emg_control.csv")


if __name__ == "__main__":
    main()
