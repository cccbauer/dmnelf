#!/usr/bin/env python3
"""
manuscript_stats.py
-------------------
Regenerate the Results tables and the per-subject scatter figure for MANUSCRIPT.md
directly from the pipeline outputs, so the manuscript stays in sync after re-runs.

- Table 1 (within-subject r, mean +/- SD [95% CI]) and the LOSO table are injected
  into MANUSCRIPT.md between <!-- BEGIN:tableN --> / <!-- END:tableN --> markers.
- A per-subject scatter of EFP r by target is written to results/full/.

Reads:  results/full/efp_persubject_all.csv, efp_group_loso.csv
Runs locally (no cluster data needed):
    python manuscript_stats.py
"""
from pathlib import Path
import numpy as np, pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJ = Path(__file__).resolve().parent.parent
RES = PROJ / "results" / "full"
MD = PROJ / "MANUSCRIPT.md"
ORDER = ["CEN", "PDA", "GSR_CEN", "DMN", "GSR_PDA", "VIS", "GSR_DMN"]


def msd_ci(r):
    r = np.asarray(r, float)
    m, sd, n = r.mean(), r.std(ddof=1), len(r)
    lo, hi = stats.t.interval(0.95, n - 1, loc=m, scale=stats.sem(r))
    return m, sd, lo, hi, n


def table1(df, res="tr"):
    d = df[df.resolution == res]
    rows = ["| Target | EFP  (mean ± SD) [95% CI] | HRF (mean ± SD) | T/A (mean ± SD) | n |",
            "|---|---|---|---|---|"]
    for t in ORDER:
        cell, means, n = {}, {}, 0
        for meth in ("EFP", "HRF", "TA"):
            r = d[(d.target == t) & (d.method == meth)].mean_r.values
            if len(r) == 0:
                cell[meth] = "—"; continue
            m, sd, lo, hi, n = msd_ci(r)
            means[meth] = m
            cell[meth] = (f"{m:.3f} ± {sd:.3f} [{lo:.3f}, {hi:.3f}]" if meth == "EFP"
                          else f"{m:.3f} ± {sd:.3f}")
        winner = max(means, key=means.get) if means else None   # bold best predictor per row
        if winner:
            cell[winner] = f"**{cell[winner]}**"
        rows.append(f"| {t} | {cell['EFP']} | {cell['HRF']} | {cell['TA']} | {n} |")
    return "\n".join(rows)


def loso_table(lo, res="tr"):
    d = lo[lo.resolution == res].set_index("target").reindex(ORDER).dropna(how="all")
    rows = ["| Target | Electrode | LOSO r | p (sign-flip) |", "|---|---|---|---|"]
    for t, row in d.iterrows():
        p = row["sign_flip_p"]
        star = " *" if p < 0.05 else ""
        rows.append(f"| {t} | {row['common_ch']} | {row['loso_mean_r']:.3f}{star} | {p:.3f} |")
    return "\n".join(rows)


def crosscohort_table(res="tr"):
    """Combine the nf1 (and nf2 if present) cross-cohort summaries into one table."""
    import numpy as np
    arms = [("nf1", RES.parent / f"cross_cohort_efp_summary_{res}.csv"),
            ("nf2", RES.parent / f"cross_cohort_efp_summary_{res}_nf2.csv")]
    frames = {}
    for name, p in arms:
        if p.exists():
            frames[name] = pd.read_csv(p).set_index("target")
    if not frames:
        return "_(cross-cohort results pending)_"
    hdr = "| Target | Electrode |"
    sub = "|---|---|"
    for name in frames:
        hdr += f" {name} r | {name} p |"; sub += "---|---|"
    rows = [hdr, sub]
    for t in ORDER:
        any_arm = next((f for f in frames.values() if t in f.index), None)
        if any_arm is None:
            continue
        el = any_arm.loc[t, "electrode"]
        line = f"| {t} | {el} |"
        for name, f in frames.items():
            if t in f.index:
                r = f.loc[t, "mean_r"]; p = f.loc[t, "sign_flip_p"]
                star = " *" if p < 0.05 else ""
                line += f" {r:+.3f}{star} | {p:.3f} |"
            else:
                line += " — | — |"
        rows.append(line)
    ntxt = ", ".join(f"{name} n={f['n_test'].iloc[0]}" for name, f in frames.items())
    return "\n".join(rows) + f"\n\n*Transfer electrode = DMNELF LOSO modal channel; {ntxt}. `*` p<0.05.*"


def scatter(df, res="tr", out=RES / "paper_fig_persubject_scatter_tr.png"):
    d = df[(df.resolution == res) & (df.method == "EFP")]
    fig, ax = plt.subplots(figsize=(9, 4.6))
    rng = np.random.default_rng(0)
    for i, t in enumerate(ORDER):
        r = d[d.target == t].mean_r.values
        if not len(r):
            continue
        x = i + (rng.random(len(r)) - 0.5) * 0.28
        ax.scatter(x, r, s=26, color="#2E86C1", alpha=0.7, zorder=3, edgecolor="white", linewidth=0.4)
        m, sd, lo, hi, n = msd_ci(r)
        ax.plot([i - 0.28, i + 0.28], [m, m], color="#111", lw=2.2, zorder=4)
        ax.add_patch(plt.Rectangle((i - 0.2, lo), 0.4, hi - lo, color="#111", alpha=0.12, zorder=2))
    ax.axhline(0, color="grey", lw=0.8, ls="--")
    ax.set_xticks(range(len(ORDER))); ax.set_xticklabels(ORDER, rotation=20)
    ax.set_ylabel("within-subject CV r (EFP)")
    ax.set_title("Per-subject EFP decoding by target (TR)  —  bar = group mean, band = 95% CI",
                 fontsize=11, fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(out, dpi=150); plt.close(fig)
    print("saved", out)


def inject(md_text, key, content):
    a, b = f"<!-- BEGIN:{key} -->", f"<!-- END:{key} -->"
    block = f"{a}\n{content}\n{b}"
    if a in md_text and b in md_text:
        pre = md_text.split(a)[0]; post = md_text.split(b)[1]
        return pre + block + post
    return md_text  # markers absent -> leave unchanged


def main():
    df = pd.read_csv(RES / "efp_persubject_all.csv")
    lo = pd.read_csv(RES / "efp_group_loso.csv")
    scatter(df)
    if MD.exists():
        txt = MD.read_text()
        txt = inject(txt, "table1", table1(df))
        txt = inject(txt, "loso", loso_table(lo))
        txt = inject(txt, "crosscohort", crosscohort_table())
        MD.write_text(txt)
        print("updated", MD)


if __name__ == "__main__":
    main()
