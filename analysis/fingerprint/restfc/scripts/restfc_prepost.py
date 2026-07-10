#!/usr/bin/env python3
"""
restfc_prepost.py  —  pre vs post resting-state connectivity change (mbNF)
--------------------------------------------------------------------------
Reads the per-subject restfc CSVs (results/restfc_{cohort}_{sub}.csv), maps rest
runs to pre/post, and tests the within-session pre->post change per group:
  DMNELF (schizophrenia): rest run-01=pre, run-02=post
  rtBPD  (elevated BPD):  runs 1,2=pre (mean), runs 3,4=post (mean); ses-nf1 & ses-nf2 separately
Groups: DMNELF, rtBPD nf1, rtBPD nf2. Paired t-test on post-pre per measure.
Then relates the per-subject connectivity change to the self-reported CALM change
(sliders_both.csv, first->last feedback run within session).
"""
from pathlib import Path
import numpy as np, pandas as pd, glob, re
from scipy import stats

RES = Path(__file__).resolve().parent.parent / "results"
SLIDERS = Path(__file__).resolve().parents[2] / "fsnr_eeg" / "results" / "sliders_both.csv"
# PRIMARY: DiFuMo component connectivity (robust to global-signal / first-run inflation).
# DIAGNOSTIC: personalized voxelwise pairwise-r measures are dominated by shared global signal
#   and a first-run (run-1) arousal/motion artifact — sign-flips when run-1 is dropped from the
#   pre average — so they are NOT interpreted as neurofeedback effects (see run1_sensitivity()).
PRIMARY = ["within_dmn_difumo", "within_cen_difumo", "dmn_cen_difumo",
           "mpfc_pcc", "acc_pcc", "acc_mpfc"]
DIAG = ["within_dmn_pers", "within_cen_pers", "dmn_cen_pers"]
MEAS = PRIMARY + DIAG
QA = re.compile(r"dmnelf(999|1\d\d\d)$")   # exclude phantom/QA ids


def load():
    files = sorted(glob.glob(str(RES / "restfc_dmnelf_*.csv")) + glob.glob(str(RES / "restfc_rtbpd_*.csv")))
    frames = [pd.read_csv(f) for f in files]
    d = pd.concat(frames, ignore_index=True)
    d = d[~d.subject.map(lambda s: bool(QA.match(s)))].copy()
    # DMNELF has 2 rest runs (run1=pre, run2=post); rtBPD has 4 (runs 1,2=pre, 3,4=post)
    thresh = np.where(d.cohort == "dmnelf", 1, 2)
    d["phase"] = np.where(d.run <= thresh, "pre", "post")
    return d


def prepost_table(d):
    """Per subject×session: mean pre and mean post for each measure."""
    g = d.groupby(["cohort", "subject", "session", "phase"])[MEAS].mean().reset_index()
    pre = g[g.phase == "pre"].drop(columns="phase")
    post = g[g.phase == "post"].drop(columns="phase")
    m = pre.merge(post, on=["cohort", "subject", "session"], suffixes=("_pre", "_post"))
    for meas in MEAS:
        m[f"{meas}_delta"] = m[f"{meas}_post"] - m[f"{meas}_pre"]
    return m


def report_group(name, sub):
    print(f"\n===== {name}  (n={len(sub)} subjects) =====")
    print(f"  {'measure':20s} {'pre':>8s} {'post':>8s} {'Δ':>8s} {'t':>7s} {'p':>7s} {'d':>6s}")
    rows = []
    for meas in MEAS:
        a = sub[f"{meas}_pre"].values; b = sub[f"{meas}_post"].values
        ok = np.isfinite(a) & np.isfinite(b); a, b = a[ok], b[ok]
        if len(a) < 3:
            continue
        t, p = stats.ttest_rel(b, a); dz = np.mean(b - a) / (np.std(b - a, ddof=1) + 1e-12)
        star = "*" if p < 0.05 else ("." if p < 0.1 else " ")
        print(f"  {meas:20s} {a.mean():8.3f} {b.mean():8.3f} {b.mean()-a.mean():+8.3f} "
              f"{t:7.2f} {p:7.3f}{star} {dz:+6.2f}")
        rows.append(dict(group=name, measure=meas, n=len(a), pre=a.mean(), post=b.mean(),
                         delta=b.mean()-a.mean(), t=t, p=p, dz=dz))
    return rows


def calm_change():
    """Per subject×session slider_calm change (last - first feedback run)."""
    if not SLIDERS.exists():
        return None
    s = pd.read_csv(SLIDERS)
    s = s.dropna(subset=["slider_calm"])
    out = []
    key = "session" if "session" in s.columns else None
    grp = ["cohort", "subject"] + ([key] if key else [])
    for gk, gg in s.groupby(grp):
        gg = gg.sort_values("run")
        if gg.slider_calm.notna().sum() >= 2:
            rec = dict(zip(grp, gk if isinstance(gk, tuple) else (gk,)))
            rec["calm_delta"] = gg.slider_calm.iloc[-1] - gg.slider_calm.iloc[0]
            out.append(rec)
    return pd.DataFrame(out)


def run1_sensitivity(d):
    """Show the personalized voxelwise measures are a first-run artifact: dropping run-1 from the
    pre average reverses the sign. Only meaningful where >=3 pre runs exist (rtBPD)."""
    print("\n===== run-1 artifact check (personalized voxelwise measures, rtBPD) =====")
    print("  dropping the anomalous first rest run from 'pre' should reverse a genuine artifact")
    for ses in ["ses-nf1", "ses-nf2"]:
        s = d[(d.cohort == "rtbpd") & (d.session == ses)]
        for meas in DIAG:
            outs = []
            for lab, pre_runs in [("pre=1,2", [1, 2]), ("pre=2", [2])]:
                pre = s[s.run.isin(pre_runs)].groupby("subject")[meas].mean()
                post = s[s.run.isin([3, 4])].groupby("subject")[meas].mean()
                mm = pd.concat([pre.rename("pre"), post.rename("post")], axis=1).dropna()
                if len(mm) >= 3:
                    dz = (mm.post - mm.pre).mean() / ((mm.post - mm.pre).std(ddof=1) + 1e-12)
                    outs.append(f"{lab}: d={dz:+.2f}")
            print(f"  {ses} {meas:18s} " + " | ".join(outs))


def main():
    d = load()
    print(f"loaded {d.subject.nunique()} subjects, {len(d)} rest runs")
    m = prepost_table(d)
    m.to_csv(RES / "restfc_prepost_subject.csv", index=False)

    groups = {
        "DMNELF (schizophrenia)": m[m.cohort == "dmnelf"],
        "rtBPD nf1": m[(m.cohort == "rtbpd") & (m.session == "ses-nf1")],
        "rtBPD nf2": m[(m.cohort == "rtbpd") & (m.session == "ses-nf2")],
    }
    allrows = []
    for name, sub in groups.items():
        allrows += report_group(name, sub)
    pd.DataFrame(allrows).to_csv(RES / "restfc_prepost_stats.csv", index=False)
    run1_sensitivity(d)

    # ---- connectivity change vs calm change ----
    cc = calm_change()
    if cc is not None:
        # harmonize to restfc keys: cohort lower-case; session -> "ses-<x>"
        cc["cohort"] = cc["cohort"].str.lower()
        cc["session"] = cc["session"].astype(str).apply(lambda s: s if s.startswith("ses-") else "ses-" + s)
        j = m.merge(cc, on=["cohort", "subject", "session"], how="inner")
        print(f"\n===== connectivity Δ  vs  calm Δ  (n={len(j)} sessions) =====")
        print(f"  {'measure':20s} {'r':>7s} {'p':>7s}")
        crows = []
        for meas in MEAS:
            x = j[f"{meas}_delta"].values; y = j["calm_delta"].values
            ok = np.isfinite(x) & np.isfinite(y)
            if ok.sum() < 5 or np.std(x[ok]) == 0 or np.std(y[ok]) == 0:
                continue
            r = np.corrcoef(x[ok], y[ok])[0, 1]; n = ok.sum()
            t = r * np.sqrt(n - 2) / np.sqrt(1 - r ** 2); p = 2 * stats.t.sf(abs(t), n - 2)
            star = "*" if p < 0.05 else ("." if p < 0.1 else " ")
            print(f"  {meas:20s} {r:+7.2f} {p:7.3f}{star}")
            crows.append(dict(measure=meas, r=r, p=p, n=int(n)))
        pd.DataFrame(crows).to_csv(RES / "restfc_calm_link.csv", index=False)

    print("\nsaved restfc_prepost_subject.csv, restfc_prepost_stats.csv, restfc_calm_link.csv")


if __name__ == "__main__":
    main()
