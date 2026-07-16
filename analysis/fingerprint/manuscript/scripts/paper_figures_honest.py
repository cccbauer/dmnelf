#!/usr/bin/env python3
"""
paper_figures_honest.py  —  data-driven manuscript figures (honest EFP spine)
-----------------------------------------------------------------------------
Builds Fig 4 (replication + calibration), Fig 5 (EPOC deployment), Fig 6 (rigor + clinical)
from committed CSVs. Signal panels for Figs 1-3 (preprocessing, EFP method, timeseries) need a
cluster EEG pull and are built separately. All numbers: clean targets, LORO, feedback block.
Output: manuscript/figures/fig4_*.png ...
"""
from pathlib import Path
import numpy as np, pandas as pd, glob
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

FP = Path(__file__).resolve().parent.parent.parent           # analysis/fingerprint
EFP = FP / "efp_meirhasson" / "results"
FS = FP / "fsnr_eeg" / "results"
OUT = FP / "manuscript" / "figures"; OUT.mkdir(parents=True, exist_ok=True)
RNG = np.random.default_rng(0)
C = {"CEN": "#2c7fb8", "DMN": "#d95f0e", "PDA": "#31a354"}
COH = [("DMNELF (SZ)", "cen_clean"), ("rtBPD nf1", "cen_clean_rt_nf1_clean"),
       ("rtBPD nf2", "cen_clean_rt_nf2_clean")]
plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.dpi": 150})


def load_clean(sub_dir):
    fs = glob.glob(str(EFP / sub_dir / "efp_cen_clean_*.csv"))
    return pd.concat([pd.read_csv(f) for f in fs], ignore_index=True) if fs else pd.DataFrame()


def grp(df, tgt, ttype, mode):
    s = df[(df.target == tgt) & (df.ttype == ttype) & (df["mode"] == mode)].r.values
    s = s[np.isfinite(s)]
    if len(s) < 3:
        return np.nan, np.nan, np.nan
    obs = s.mean(); null = (RNG.choice([-1, 1], (10000, len(s))) * np.abs(s)).mean(1)
    p = (np.sum(np.abs(null) >= abs(obs)) + 1) / 10001
    return obs, s.std(ddof=1) / np.sqrt(len(s)), p


def star(p):
    return "***" if p < .001 else "**" if p < .01 else "*" if p < .05 else ""


# ---------------- Figure 4: replication + calibration ----------------
def fig4():
    dfs = {name: load_clean(d) for name, d in COH}
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(11, 4.2), gridspec_kw={"width_ratios": [1.4, 1]})
    tgts = ["CEN", "DMN", "PDA"]; x = np.arange(len(COH)); w = 0.26
    for i, t in enumerate(tgts):
        vals, err, ps = [], [], []
        for name, _ in COH:
            m, se, p = grp(dfs[name], t, "clean", "all"); vals.append(m); err.append(se); ps.append(p)
        bars = axA.bar(x + (i - 1) * w, vals, w, yerr=err, capsize=3, color=C[t], label=t, alpha=.9)
        for xi, (v, p) in enumerate(zip(vals, ps)):
            axA.text(x[xi] + (i - 1) * w, v + err[xi] + .004, star(p), ha="center", va="bottom", fontsize=9)
    axA.set_xticks(x); axA.set_xticklabels([c[0] for c in COH]); axA.axhline(0, color="k", lw=.6)
    axA.set_ylabel("within-subject decoding  r"); axA.legend(frameon=False, ncol=3, loc="upper center")
    axA.set_title("A  Within-subject decoding replicates across cohorts", loc="left", fontsize=11, weight="bold")

    cal = pd.read_csv(EFP / "efp_calibrate_mv.csv")
    order = ["transfer", "dmnelf+cal1", "within_loro"]; lab = ["0-shot\ntransfer", "+1-run\ncalibration", "within-\nsubject"]
    for sess, mk, col in [("nf1", "o", "#2c7fb8"), ("nf2", "s", "#762a83")]:
        ys = [cal[(cal.session == sess) & (cal.target == "CEN") & (cal.scheme == s)].r.mean() for s in order]
        axB.plot(range(3), ys, mk + "-", color=col, label=f"CEN {sess}", lw=2, ms=7)
    axB.set_xticks(range(3)); axB.set_xticklabels(lab); axB.axhline(0, color="k", lw=.6)
    axB.set_ylabel("cross-cohort  r"); axB.legend(frameon=False)
    axB.set_title("B  Calibration recovers transfer", loc="left", fontsize=11, weight="bold")
    fig.tight_layout(); fig.savefig(OUT / "fig4_replication_calibration.png", bbox_inches="tight")
    print("wrote fig4")


# ---------------- Figure 5: EPOC deployment ----------------
def fig5():
    d = load_clean("cen_clean")   # DMNELF has all + epoc modes
    fig, ax = plt.subplots(figsize=(6, 4.2))
    tgts = ["CEN", "DMN", "PDA"]; x = np.arange(len(tgts)); w = 0.36
    full = [grp(d, t, "clean", "all")[0] for t in tgts]
    epoc = [grp(d, t, "clean", "epoc")[0] for t in tgts]
    ax.bar(x - w / 2, full, w, label="full 31-ch cap", color="#636363", alpha=.9)
    ax.bar(x + w / 2, epoc, w, label="EPOC-X 12-ch", color="#41ab5d", alpha=.9)
    for xi, (f, e) in enumerate(zip(full, epoc)):
        ax.text(xi + w / 2, e + .003, f"{100*e/f:.0f}%", ha="center", va="bottom", fontsize=9)
    ax.set_xticks(x); ax.set_xticklabels(tgts); ax.set_ylabel("decoding  r"); ax.axhline(0, color="k", lw=.6)
    ax.legend(frameon=False); ax.set_title("Consumer-headset feasibility (EPOC-X retains ~92% of CEN)",
                                           fontsize=11, weight="bold")
    fig.tight_layout(); fig.savefig(OUT / "fig5_epoc_deployment.png", bbox_inches="tight")
    print("wrote fig5")


# ---------------- Figure 6: rigor + clinical ----------------
def fig6():
    d = load_clean("cen_clean")
    fig, axs = plt.subplots(1, 3, figsize=(14, 4.2))
    # A: confound-cleaning (orig vs clean), DMNELF
    tgts = ["CEN", "DMN", "PDA"]; x = np.arange(len(tgts)); w = 0.36
    orig = [grp(d, t, "orig", "all")[0] for t in tgts]; clean = [grp(d, t, "clean", "all")[0] for t in tgts]
    axs[0].bar(x - w / 2, orig, w, label="naive (motion-retained)", color="#cccccc")
    axs[0].bar(x + w / 2, clean, w, label="confound-regressed", color="#2c7fb8")
    axs[0].set_xticks(x); axs[0].set_xticklabels(tgts); axs[0].set_ylabel("decoding  r")
    axs[0].legend(frameon=False, fontsize=9); axs[0].set_title("A  Motion inflates naive coupling ~2–3×",
                                                               loc="left", fontsize=11, weight="bold")
    # B: controls fail (CEN readout by method) — committed honest values
    methods = ["EFP\n(linear)", "deep\n(R-EEGNet)", "frontal\ntheta/FTA", "f-SNR"]
    cen_by = [grp(d, "CEN", "clean", "all")[0], 0.004, 0.005, 0.012]   # deep/theta/fsnr: committed nulls
    cols = ["#2c7fb8", "#bbbbbb", "#bbbbbb", "#bbbbbb"]
    axs[1].bar(range(4), cen_by, color=cols)
    axs[1].axhline(0, color="k", lw=.6); axs[1].set_xticks(range(4)); axs[1].set_xticklabels(methods, fontsize=9)
    axs[1].set_ylabel("CEN decoding  r"); axs[1].set_title("B  Only the linear decoder works",
                                                           loc="left", fontsize=11, weight="bold")
    # C: clinical anchor — calm <-> PDA by cohort (from sliders_both)
    sl = pd.read_csv(FS / "sliders_both.csv")
    rows = []
    for lab, mask in [("DMNELF", (sl.cohort == "DMNELF")), ("rtBPD nf1", (sl.cohort == "rtBPD") & (sl.session == "nf1")),
                      ("rtBPD nf2", (sl.cohort == "rtBPD") & (sl.session == "nf2"))]:
        g = sl[mask].dropna(subset=["slider_calm", "rt_pda_mean"])
        if len(g) > 5:
            r, p = pearsonr(g.slider_calm, g.rt_pda_mean); rows.append((lab, r, len(g)))
    axs[2].bar([r[0] for r in rows], [r[1] for r in rows], color="#31a354", alpha=.9)
    for i, (lab, r, n) in enumerate(rows):
        axs[2].text(i, r + .005, f"r={r:.2f}\nn={n}", ha="center", va="bottom", fontsize=8)
    axs[2].axhline(0, color="k", lw=.6); axs[2].set_ylabel("calm ↔ PDA regulation  r")
    axs[2].set_ylim(top=max(r[1] for r in rows) * 1.25)
    axs[2].set_title("C  Regulating the target ↔ feeling calmer", loc="left", fontsize=11, weight="bold")
    fig.tight_layout(); fig.savefig(OUT / "fig6_rigor_clinical.png", bbox_inches="tight")
    print("wrote fig6")


if __name__ == "__main__":
    fig4(); fig5(); fig6()
    print("figures in", OUT)
