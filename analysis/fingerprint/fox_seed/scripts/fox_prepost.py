#!/usr/bin/env python3
"""
fox_prepost.py  (local)  —  pre vs post change in Fox MPFC-seed connectivity
----------------------------------------------------------------------------
Does a mbNF session shift the Fox-2005 MPFC->PCC coupling (and MPFC->DLPFC
anticorrelation)? Reads fox_roi_timeseries.csv (per rest run seed-to-seed Fisher-z),
maps runs to pre/post, paired t-test per group, plus a run-1 robustness check and a
link to the self-reported CALM change.
  DMNELF: rest run-01=pre, run-02=post; rtBPD: runs 1,2=pre, 3,4=post (mean)
"""
from pathlib import Path
import numpy as np, pandas as pd, re
from scipy import stats

RES = Path(__file__).resolve().parent.parent / "results"
SLIDERS = Path(__file__).resolve().parents[2] / "fsnr_eeg" / "results" / "sliders_both.csv"
MEAS = ["z_pcc", "z_ldlpfc", "z_rdlpfc"]
QA = re.compile(r"dmnelf(999|1\d\d\d)$")


def load():
    import glob
    d = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(str(RES / "fox_roi_*.csv")))],
                  ignore_index=True)
    d = d[~d.subject.map(lambda s: bool(QA.match(s)))].copy()
    thresh = np.where(d.cohort == "dmnelf", 1, 2)
    d["phase"] = np.where(d.run <= thresh, "pre", "post")
    return d


def prepost(d):
    g = d.groupby(["cohort", "subject", "session", "phase"])[MEAS].mean().reset_index()
    pre = g[g.phase == "pre"].drop(columns="phase"); post = g[g.phase == "post"].drop(columns="phase")
    m = pre.merge(post, on=["cohort", "subject", "session"], suffixes=("_pre", "_post"))
    for meas in MEAS:
        m[f"{meas}_delta"] = m[f"{meas}_post"] - m[f"{meas}_pre"]
    return m


def report(name, sub):
    print(f"\n===== {name}  (n={len(sub)}) =====")
    print(f"  {'measure':10s} {'pre':>8s} {'post':>8s} {'Δ':>8s} {'t':>7s} {'p':>7s} {'dz':>6s}")
    for meas in MEAS:
        a = sub[f"{meas}_pre"].values; b = sub[f"{meas}_post"].values
        ok = np.isfinite(a) & np.isfinite(b); a, b = a[ok], b[ok]
        if len(a) < 3:
            continue
        t, p = stats.ttest_rel(b, a); dz = np.mean(b - a) / (np.std(b - a, ddof=1) + 1e-12)
        star = "*" if p < 0.05 else ("." if p < 0.1 else " ")
        print(f"  {meas:10s} {a.mean():8.3f} {b.mean():8.3f} {b.mean()-a.mean():+8.3f} {t:7.2f} {p:7.3f}{star} {dz:+6.2f}")


def main():
    d = load()
    print(f"loaded {d.subject.nunique()} subjects, {len(d)} rest runs")
    print("baseline sanity (all rest runs): z_pcc mean=%.3f  z_ldlpfc=%.3f  z_rdlpfc=%.3f"
          % (d.z_pcc.mean(), d.z_ldlpfc.mean(), d.z_rdlpfc.mean()))
    m = prepost(d); m.to_csv(RES / "fox_prepost_subject.csv", index=False)
    groups = {"DMNELF (schizophrenia)": m[m.cohort == "dmnelf"],
              "rtBPD nf1": m[(m.cohort == "rtbpd") & (m.session == "ses-nf1")],
              "rtBPD nf2": m[(m.cohort == "rtbpd") & (m.session == "ses-nf2")]}
    for name, sub in groups.items():
        report(name, sub)

    # run-1 robustness (rtBPD): drop first rest run from 'pre'
    print("\n===== run-1 robustness (rtBPD z_pcc): pre=1,2 vs pre=2 only =====")
    for ses in ["ses-nf1", "ses-nf2"]:
        s = d[(d.cohort == "rtbpd") & (d.session == ses)]
        outs = []
        for lab, pr in [("pre=1,2", [1, 2]), ("pre=2", [2])]:
            pre = s[s.run.isin(pr)].groupby("subject").z_pcc.mean()
            post = s[s.run.isin([3, 4])].groupby("subject").z_pcc.mean()
            mm = pd.concat([pre.rename("pre"), post.rename("post")], axis=1).dropna()
            if len(mm) >= 3:
                dz = (mm.post - mm.pre).mean() / ((mm.post - mm.pre).std(ddof=1) + 1e-12)
                outs.append(f"{lab}: d={dz:+.2f}")
        print(f"  {ses}: " + " | ".join(outs))

    # link to calm change
    if SLIDERS.exists():
        sl = pd.read_csv(SLIDERS).dropna(subset=["slider_calm"])
        cc = []
        for gk, gg in sl.groupby(["cohort", "subject", "session"]):
            gg = gg.sort_values("run")
            if gg.slider_calm.notna().sum() >= 2:
                cc.append(dict(cohort=gk[0].lower(), subject=gk[1],
                               session=gk[2] if str(gk[2]).startswith("ses-") else "ses-" + str(gk[2]),
                               calm_delta=gg.slider_calm.iloc[-1] - gg.slider_calm.iloc[0]))
        cc = pd.DataFrame(cc)
        j = m.merge(cc, on=["cohort", "subject", "session"], how="inner")
        print(f"\n===== MPFC-connectivity Δ  vs  calm Δ  (n={len(j)}) =====")
        for meas in MEAS:
            x = j[f"{meas}_delta"].values; y = j["calm_delta"].values
            ok = np.isfinite(x) & np.isfinite(y)
            if ok.sum() >= 5 and np.std(x[ok]) > 0 and np.std(y[ok]) > 0:
                r = np.corrcoef(x[ok], y[ok])[0, 1]; n = ok.sum()
                p = 2 * stats.t.sf(abs(r * np.sqrt(n - 2) / np.sqrt(1 - r ** 2)), n - 2)
                print(f"  {meas:10s} r={r:+.2f} p={p:.3f} (n={int(n)})")
    print("\nsaved fox_prepost_subject.csv")


if __name__ == "__main__":
    main()
