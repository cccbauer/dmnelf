#!/usr/bin/env python3
"""
fig123_signals.py  (local, env eeg_preproc)  —  signal figures 1-3
------------------------------------------------------------------
Fig 1: EEG preprocessing chain (raw gradient -> gradient-corrected -> filtered -> post-BCG ->
       post-ICA -> final) on real signal, + rejected-ICA topographies + best-vs-worst final.
Fig 2: EFP method (Stockwell spectrogram -> 10 bands -> [band x delay] design -> fingerprint
       weights -> prediction), Meir-Hasson style.
Fig 3: DMNELF fingerprint — predicted vs observed BOLD timeseries, BEST vs WORST subject.
Assets in manuscript/figures/assets/. Output manuscript/figures/fig{1,2,3}_*.png.
"""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mne
mne.set_log_level("ERROR")

FIG = Path(__file__).resolve().parent.parent / "figures"
A = FIG / "assets"
BEST, WORST = "dmnelf1002", "dmnelf009"
CH = "Pz"
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.dpi": 150})


def montage_pos(ch_names):
    info = mne.create_info(list(ch_names), 500, "eeg")
    info.set_montage(mne.channels.make_standard_montage("standard_1020"), on_missing="ignore")
    return info


# ───────────────────────── Figure 1: preprocessing chain ─────────────────────────
def fig1():
    z = np.load(A / f"preproc_sub-{BEST}.npz", allow_pickle=True)
    zk = set(z.files)
    stages = []
    g = A / f"gradient_{'dmnelf001'}.npz"
    if g.exists():
        gz = np.load(g, allow_pickle=True)
        stages.append(("0 · raw (gradient artifact)", gz["t"], gz["x"] * 1e3, "mV", "#b2182b"))
    S = [("1 · gradient-corrected (BVA, 1 kHz)", "s1_gradient_corrected"),
         ("2 · band-pass 1–40 Hz", "s2_filtered"),
         ("3 · BCG (heartbeat) removed", "s3_post_bcg"),
         ("4 · ICA (eye/muscle) removed", "s4_post_ica"),
         ("5 · final: interp + common-avg ref", "s5_final")]
    for lab, key in S:
        tk, xk = f"{key}_t", f"{key}_{CH}"
        if xk in zk:
            stages.append((lab, z[tk], z[xk] * 1e6, "µV", "#2166ac"))

    n = len(stages)
    fig = plt.figure(figsize=(13, 2.0 + 1.05 * n))
    gs = fig.add_gridspec(n, 3, width_ratios=[3.2, 1, 1.5], hspace=0.55, wspace=0.3)
    # R-peaks in the window (for stage 3 heartbeat annotation)
    rp = None
    if "rpeaks" in zk and "ecg_sf" in zk:
        sf = float(z["ecg_sf"]); rp = np.asarray(z["rpeaks"]) / sf - 60.0
        rp = rp[(rp >= 0) & (rp <= 6)]
    for i, (lab, t, x, unit, col) in enumerate(stages):
        ax = fig.add_subplot(gs[i, 0])
        ax.plot(t, x, color=col, lw=0.6)
        ax.set_ylabel(unit, fontsize=8); ax.set_title(lab, loc="left", fontsize=9.5, weight="bold")
        ax.margins(x=0)
        if rp is not None and ("BCG" in lab or "band-pass" in lab):
            for r in rp:
                ax.axvline(r, color="#d6604d", lw=0.6, ls=":", alpha=.7)
        if i < n - 1:
            ax.set_xticklabels([])
        else:
            ax.set_xlabel("time (s)")
    # PSD before/after (stage 1 vs final)
    axp = fig.add_subplot(gs[:2, 1])
    for key, c, l in [("s1_gradient_corrected", "#999999", "input"), ("s5_final", "#2166ac", "final")]:
        fk, pk = f"{key}_{CH}_psdf", f"{key}_{CH}_psd"
        if pk in zk:
            axp.plot(z[fk], z[pk], color=c, lw=1.2, label=l)
    axp.set_title("PSD", fontsize=9, weight="bold"); axp.set_xlabel("Hz"); axp.set_ylabel("dB")
    axp.legend(frameon=False, fontsize=8)
    # ECG + R-peaks
    if "ecg" in zk and z["ecg"] is not None and np.ndim(z["ecg"]) > 0:
        axe = fig.add_subplot(gs[3:5, 1]); sf = float(z["ecg_sf"])
        seg = z["ecg"][int(60 * sf):int(66 * sf)]; te = np.arange(len(seg)) / sf
        axe.plot(te, seg, color="#762a83", lw=0.7)
        if rp is not None:
            for r in rp:
                axe.axvline(r, color="#d6604d", lw=0.6, ls=":")
        axe.set_title("ECG + R-peaks", fontsize=9, weight="bold"); axe.set_xlabel("s"); axe.set_yticks([])
    # ICA rejected component topomaps
    if "ica_topos" in zk and z["ica_topos"].size:
        topos = z["ica_topos"]; chn = z["ica_ch_names"]; labs = z["ica_rej_labels"]
        info = montage_pos(chn); k = min(n, len(topos))
        for j in range(k):
            axt = fig.add_subplot(gs[j, 2])
            mne.viz.plot_topomap(topos[j], info, axes=axt, show=False, contours=0, cmap="RdBu_r")
            axt.set_title(f"IC rej: {labs[j]}", fontsize=7); axt.axis("off")
    fig.suptitle(f"Simultaneous EEG–fMRI preprocessing chain  (subject {BEST}, channel {CH})",
                 fontsize=12, weight="bold", y=0.995)
    fig.savefig(FIG / "fig1_preprocessing.png", bbox_inches="tight")
    plt.close(fig); print("wrote fig1")


# ───────────────────────── Figure 2: EFP method ─────────────────────────
def fig2():
    z = np.load(A / f"efp_{BEST}.npz", allow_pickle=True)
    band_hz = z["band_hz"]; blab = [f"{lo}-{hi}" for lo, hi in band_hz]
    fig = plt.figure(figsize=(15, 4.4))
    gs = fig.add_gridspec(1, 5, width_ratios=[1.5, 1.2, 1, 1, 1.5], wspace=0.45)
    # A spectrogram
    axA = fig.add_subplot(gs[0])
    axA.imshow(z["spec_power"], aspect="auto", origin="lower", cmap="magma",
               extent=[z["spec_t"][0], z["spec_t"][-1], z["spec_freqs"][0], z["spec_freqs"][-1]])
    axA.set_title(f"A  Stockwell spectrogram\n(electrode {z['best_ch']})", loc="left", fontsize=9.5, weight="bold")
    axA.set_xlabel("time (s)"); axA.set_ylabel("Hz")
    # B 10-band power
    axB = fig.add_subplot(gs[1])
    axB.imshow(z["band_power"], aspect="auto", origin="lower", cmap="viridis",
               extent=[0, z["band_power"].shape[1], 0, 10])
    axB.set_yticks(np.arange(10) + .5); axB.set_yticklabels(blab, fontsize=6)
    axB.set_title("B  10 equal-energy\nbands (per TR)", loc="left", fontsize=9.5, weight="bold")
    axB.set_xlabel("TR")
    # C design example [delay x band]
    axC = fig.add_subplot(gs[2])
    axC.imshow(z["design_example"], aspect="auto", origin="lower", cmap="viridis")
    axC.set_title("C  [band × delay]\ndesign (one TR)", loc="left", fontsize=9.5, weight="bold")
    axC.set_xlabel("band"); axC.set_ylabel("delay (TR)")
    # D fingerprint weights
    axD = fig.add_subplot(gs[3])
    fp = z["fingerprint"]; vm = np.abs(fp).max()
    im = axD.imshow(fp, aspect="auto", origin="lower", cmap="RdBu_r", vmin=-vm, vmax=vm)
    axD.set_title("D  learned ridge\nweights = fingerprint", loc="left", fontsize=9.5, weight="bold")
    axD.set_xlabel("band"); axD.set_ylabel("delay (TR)")
    fig.colorbar(im, ax=axD, fraction=.046)
    # E prediction teaser
    axE = fig.add_subplot(gs[4])
    o, p = z["obs_CEN"][:120], z["pred_CEN"][:120]
    axE.plot(o, color="#333333", lw=1.2, label="observed CEN")
    axE.plot(p, color="#2166ac", lw=1.2, label="predicted")
    axE.set_title(f"E  prediction\n(CEN r={float(z['r_CEN']):.2f})", loc="left", fontsize=9.5, weight="bold")
    axE.set_xlabel("TR"); axE.legend(frameon=False, fontsize=8)
    fig.suptitle("The EEG fingerprint (EFP) decoding method", fontsize=12, weight="bold", y=1.06)
    fig.savefig(FIG / "fig2_efp_method.png", bbox_inches="tight")
    plt.close(fig); print("wrote fig2")


# ───────────────────────── Figure 3: best vs worst timeseries ─────────────────────────
def fig3():
    zb = np.load(A / f"efp_{BEST}.npz", allow_pickle=True)
    zw = np.load(A / f"efp_{WORST}.npz", allow_pickle=True)
    fig = plt.figure(figsize=(14, 6))
    gs = fig.add_gridspec(2, 2, width_ratios=[2.6, 1], hspace=0.4, wspace=0.3)
    for i, (z, role) in enumerate([(zb, "best"), (zw, "worst")]):
        ax = fig.add_subplot(gs[i, 0])
        o, p = z["obs_CEN"][:150], z["pred_CEN"][:150]
        ax.plot(o, color="#333333", lw=1.3, label="observed CEN (BOLD)")
        ax.plot(p, color="#2166ac" if i == 0 else "#b2182b", lw=1.3, label="EEG-predicted")
        ax.set_title(f"{role.upper()} — {z['subject']}  (CEN r = {float(z['r_CEN']):+.2f})",
                     loc="left", fontsize=11, weight="bold")
        ax.set_ylabel("z-scored activation"); ax.margins(x=0); ax.legend(frameon=False, fontsize=9, ncol=2)
        if i == 1:
            ax.set_xlabel("feedback TR")
    # right: scalp topography of CEN decodability (reuse committed clean topomap)
    axt = fig.add_subplot(gs[:, 1])
    topo = FIG.parent.parent / "efp_meirhasson" / "results" / "efp_topomap.png"
    if topo.exists():
        axt.imshow(plt.imread(topo)); axt.axis("off")
        axt.set_title("CEN decodability\n(clean, centro-parietal)", fontsize=10, weight="bold")
    else:
        axt.axis("off")
    fig.suptitle("The DMNELF fingerprint: EEG tracks the CEN network — best vs worst subject",
                 fontsize=12.5, weight="bold")
    fig.savefig(FIG / "fig3_fingerprint_timeseries.png", bbox_inches="tight")
    plt.close(fig); print("wrote fig3")


if __name__ == "__main__":
    fig1(); fig2(); fig3()
    print("done ->", FIG)
