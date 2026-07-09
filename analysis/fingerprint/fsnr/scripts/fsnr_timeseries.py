#!/usr/bin/env python3
"""
fsnr_timeseries.py
------------------
Intuitive timeseries explorer for the fMRI f-SNR variables (DMN/CEN/PDA/global) during
neurofeedback. Two artifacts:
  1. results/fsnr_timeseries_runs.pdf  — one feedback run per page (sorted by beta_PDA),
     shape-overlay panel + rolling-variance/quench panel, annotated with per-run metrics.
  2. results/fig_fsnr_timeseries_group.png — onset-aligned group mean +/- SEM and the
     average rolling-variance quench trajectory, with dedicated rest runs as flat control.
Read-only on data; z-score only for the shape overlay (native scale for variance panels).
"""
from pathlib import Path
import numpy as np, pandas as pd, re
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from fsnr_fmri import task_regressor, BASELINE_TR, HRF_DROP, DMN_I, CEN_I, TR
from fsnr_proxy import running_fsnr, W as PW

PROJ = Path(__file__).resolve().parent.parent
DATA = PROJ / "data"; RES = PROJ / "results"
GS = dict(np.load(DATA / "global_signal.npz", allow_pickle=True))
WIN = 15            # rolling window (TR) ~ 18 s
COL = {"DMN": "#c0504d", "CEN": "#1f77b4", "PDA": "#7030a0", "global": "#555"}


def z(x):
    return (x - np.nanmean(x)) / (np.nanstd(x) + 1e-12)


def roll_var(x, w=WIN):
    s = pd.Series(x)
    return s.rolling(w, center=True, min_periods=w // 2).var().values


def roll_corr(a, b, w=WIN):
    return pd.Series(a).rolling(w, center=True, min_periods=w // 2).corr(pd.Series(b)).values


def load(task):
    out = []
    for f in sorted(DATA.glob(f"sub-*_task-{task}_run-*_features.npz")):
        m = re.match(rf"sub-(\w+?)_task-{task}_run-(\d+)_features", f.stem)
        zf = np.load(f, allow_pickle=True)
        fm = np.asarray(zf["fmri_features"], float)
        out.append(dict(sub=m.group(1), run=int(m.group(2)), n=fm.shape[0],
                        DMN=fm[:, DMN_I], CEN=fm[:, CEN_I], PDA=np.asarray(zf["pda"], float),
                        gs=GS.get(f"{m.group(1)}|{task}|{int(m.group(2))}")))
    return out


def page(pdfp, r, met):
    n = r["n"]; t = np.arange(n) * TR
    fig, ax = plt.subplots(3, 1, figsize=(11, 8.4), sharex=True,
                           gridspec_kw=dict(height_ratios=[1.15, 1, 0.9]))
    # --- top: z-scored shape overlay ---
    for nm in ["DMN", "CEN", "PDA"]:
        ax[0].plot(t, z(r[nm]), color=COL[nm], lw=1.3, label=nm)
    if r["gs"] is not None and len(r["gs"]) == n:
        ax[0].plot(t, z(r["gs"]), color=COL["global"], lw=1.0, alpha=.6, label="global")
    tr = task_regressor(n)
    ax[0].plot(t, z(tr) * .8, color="k", lw=1.2, ls="--", alpha=.6, label="task (HRF)")
    ax[0].axvspan(0, BASELINE_TR * TR, color="grey", alpha=.12, label="rest baseline")
    ax[0].axvspan(BASELINE_TR * TR, (BASELINE_TR + HRF_DROP) * TR, color="orange", alpha=.10)
    ax[0].axhline(0, color="k", lw=.5, alpha=.4)
    ax[0].set_ylabel("z-scored signal"); ax[0].legend(ncol=6, fontsize=8, loc="upper right")
    m = met.get((r["sub"], r["run"]), {})
    ax[0].set_title(f"{r['sub']}  run-{r['run']}    "
                    f"β_PDA={m.get('betaPDA',np.nan):+.2f}  fSNR_PDA={m.get('fsnrPDA',np.nan):+.1f}dB   "
                    f"qDMN={m.get('qDMN',np.nan):+.1f}dB  qGLOBAL={m.get('qGLOBAL',np.nan):+.1f}dB",
                    fontsize=10, fontweight="bold")
    # --- bottom: rolling variance (native) + rolling DMN-CEN corr on twin axis ---
    for nm in ["DMN", "CEN"]:
        ax[1].plot(t, roll_var(r[nm]), color=COL[nm], lw=1.4, label=f"var {nm}")
    if r["gs"] is not None and len(r["gs"]) == n:
        gv = roll_var(z(r["gs"]))     # z-scored global so it shares the variance axis scale
        ax[1].plot(t, gv, color=COL["global"], lw=1.0, alpha=.6, label="var global (z)")
    ax[1].axvspan(0, BASELINE_TR * TR, color="grey", alpha=.12)
    ax[1].set_ylabel("rolling variance")
    ax2 = ax[1].twinx()
    ax2.plot(t, roll_corr(r["DMN"], r["CEN"]), color="#2e7d32", lw=1.0, alpha=.5)
    ax2.axhline(0, color="#2e7d32", lw=.5, ls=":", alpha=.5)
    ax2.set_ylabel("DMN–CEN corr", color="#2e7d32"); ax2.set_ylim(-1, 1)
    ax[1].legend(ncol=3, fontsize=8, loc="upper right")
    # --- 3rd panel: causal running f-SNR (dB) — the actual f-SNR over time ---
    for nm in ["PDA", "CEN", "DMN"]:
        _, db, _, _ = running_fsnr(r[nm])
        ax[2].plot(t, db, color=COL[nm], lw=1.4, label=f"f-SNR {nm}")
    ax[2].axvspan(0, BASELINE_TR * TR, color="grey", alpha=.12)
    ax[2].axhline(0, color="k", lw=.5, alpha=.4)
    ax[2].set_ylabel(f"running f-SNR (dB)\n[causal {PW}-TR window]"); ax[2].set_xlabel("time (s)")
    ax[2].legend(ncol=2, fontsize=8, loc="upper right")
    for a in [ax[0], ax[1], ax[2]]:
        a.spines[["top"]].set_visible(False)
    fig.tight_layout(); pdfp.savefig(fig); plt.close(fig)


def group_page(fb, rest):
    N = 125
    def stack(runs, key, n=N):
        M = [r[key][:n] for r in runs if r[key] is not None and len(r[key]) >= n]
        return np.array(M)
    fig, ax = plt.subplots(1, 4, figsize=(19, 4.4))
    t = np.arange(N) * TR
    # (a) onset-aligned group mean +/- SEM of networks (z per run first)
    for nm in ["DMN", "CEN", "PDA"]:
        M = np.array([z(r[nm])[:N] for r in fb if r["n"] >= N])
        mu, se = M.mean(0), M.std(0) / np.sqrt(len(M))
        ax[0].plot(t, mu, color=COL[nm], lw=1.6, label=nm)
        ax[0].fill_between(t, mu - se, mu + se, color=COL[nm], alpha=.2)
    ax[0].axvspan(0, BASELINE_TR * TR, color="grey", alpha=.12)
    ax[0].axhline(0, color="k", lw=.5); ax[0].set_xlabel("time (s)"); ax[0].set_ylabel("z (per-run) mean±SEM")
    ax[0].set_title("Group-mean network trajectory", fontweight="bold", fontsize=10.5); ax[0].legend(fontsize=8)
    # (b) average rolling variance (feedback) DMN/CEN/global native
    for nm in ["DMN", "CEN"]:
        RV = np.array([roll_var(r[nm])[:N] for r in fb if r["n"] >= N])
        ax[1].plot(t, np.nanmean(RV, 0), color=COL[nm], lw=1.8, label=f"{nm} (feedback)")
    RVg = np.array([roll_var(z(r["gs"]))[:N] for r in fb if r["gs"] is not None and len(r["gs"]) >= N])
    ax[1].plot(t, np.nanmean(RVg, 0), color=COL["global"], lw=1.4, alpha=.7, label="global-z (feedback)")
    ax[1].axvspan(0, BASELINE_TR * TR, color="grey", alpha=.12)
    ax[1].set_xlabel("time (s)"); ax[1].set_ylabel("mean rolling variance")
    ax[1].set_title("Quench trajectory: variance drops after baseline", fontweight="bold", fontsize=10.5)
    ax[1].legend(fontsize=8)
    # (c) rest-run control: rolling variance is flat (DMN)
    RVf = np.array([roll_var(r["DMN"])[:N] for r in fb if r["n"] >= N])
    RVr = np.array([roll_var(r["DMN"])[:N] for r in rest if r["n"] >= N])
    ax[2].plot(t, np.nanmean(RVf, 0), color=COL["DMN"], lw=1.8, label="DMN feedback")
    ax[2].plot(t, np.nanmean(RVr, 0), color="#e08e45", lw=1.8, label="DMN rest (control)")
    ax[2].axvspan(0, BASELINE_TR * TR, color="grey", alpha=.12)
    ax[2].set_xlabel("time (s)"); ax[2].set_ylabel("mean rolling variance")
    ax[2].set_title("Feedback quenches; rest stays flat", fontweight="bold", fontsize=10.5); ax[2].legend(fontsize=8)
    # (d) group running f-SNR(t) in dB — the actual f-SNR trajectory
    for nm in ["PDA", "CEN", "DMN"]:
        DB = np.array([running_fsnr(r[nm])[1][:N] for r in fb if r["n"] >= N])
        mu, se = np.nanmean(DB, 0), np.nanstd(DB, 0) / np.sqrt(len(DB))
        ax[3].plot(t, mu, color=COL[nm], lw=1.8, label=f"f-SNR {nm}")
        ax[3].fill_between(t, mu - se, mu + se, color=COL[nm], alpha=.2)
    ax[3].axvspan(0, BASELINE_TR * TR, color="grey", alpha=.12)
    ax[3].axhline(0, color="k", lw=.5); ax[3].set_xlabel("time (s)"); ax[3].set_ylabel("running f-SNR (dB)")
    ax[3].set_title("f-SNR(t) rises during feedback", fontweight="bold", fontsize=10.5); ax[3].legend(fontsize=8)
    for a in ax: a.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(RES / "fig_fsnr_timeseries_group.png", dpi=150); plt.close(fig)


def main():
    fb = load("feedback"); rest = load("rest")
    met = {}
    csv = RES / "fsnr_tighten.csv"
    if csv.exists():
        d = pd.read_csv(csv)
        for _, r in d.iterrows():
            met[(r["subject"], int(r["run"]))] = r.to_dict()
    fb_sorted = sorted(fb, key=lambda r: -met.get((r["sub"], r["run"]), {}).get("betaPDA", 0))
    with PdfPages(RES / "fsnr_timeseries_runs.pdf") as pdfp:
        for r in fb_sorted:
            page(pdfp, r, met)
    group_page(fb, rest)
    print(f"saved {RES/'fsnr_timeseries_runs.pdf'} ({len(fb)} pages)")
    print(f"saved {RES/'fig_fsnr_timeseries_group.png'}")


if __name__ == "__main__":
    main()
