# 03_stats_pre_vs_post_rtbpd.py
# Run locally: python 03_stats_pre_vs_post_rtbpd.py
# Deploys cluster script, submits SLURM job, monitors.
#
# What it does on the cluster:
#   PRIMARY      — paired Welch/Student t-test (ttest_rel, within-subject
#                  design: each of the 15 subjects contributes one pre and
#                  one post value per parameter x microstate) + paired
#                  Cohen's d, across the 4 temporal parameters x 7
#                  microstates (28 tests) + BH-FDR correction
#                  (statsmodels.stats.multitest.multipletests, fdr_bh).
#   CONFIRMATORY — MixedLM per (parameter, microstate): value ~ C(condition),
#                  groups=subject (random intercept only -- each subject
#                  contributes exactly 2 rows [pre, post] per model, so a
#                  random slope for condition is not identifiable; a
#                  random-intercept model is the standard/simple choice
#                  here and is essentially a mixed-model analogue of the
#                  paired t-test).
#   FIGURES      — (a) 7-panel topomap of the fitted microstate templates
#                  (b) 4-panel pre-vs-post bar chart (one per parameter)
#                      with BH-FDR significance stars.
#
# Outputs ({CLUSTER_BASE}/results/):
#   stats_pre_vs_post.csv   parameter x microstate paired-test + LMM table
#   templates_topomap.png
#   pre_vs_post_params.png

import py_compile
import time
from pathlib import Path
from utils_cluster import run_ssh, scp_to, make_cluster_dirs
from config_rtbpd_ms import (
    CLUSTER_BASE, SLURM_ACCOUNT, PYTHON,
    SFREQ_TAG, N_MICROSTATES, SCRIPTS_DIR,
)

# ── 1. Build cluster-side script ───────────────────────────
lines = [
    '#!/usr/bin/env python3',
    '"""',
    '03_stats_pre_vs_post_rtbpd_cluster.py',
    'Paired pre-vs-post stats (t-test + MixedLM confirmatory) on the',
    'temporal_params.csv produced by 02_temporal_params_rtbpd.py, plus',
    'template topomaps and a pre/post summary figure.',
    '"""',
    'import sys',
    'sys.stdout.reconfigure(line_buffering=True)',
    'import numpy as np',
    'import json',
    'from pathlib import Path',
    'import warnings',
    'warnings.filterwarnings("ignore")',
    'import matplotlib',
    'matplotlib.use("Agg")',
    'import matplotlib.pyplot as plt',
    'from scipy import stats as spstats',
    'import pandas as pd',
    'from statsmodels.stats.multitest import multipletests',
    '',
    'try:',
    '    import statsmodels.formula.api as smf',
    '    HAS_LMM = True',
    '    print("statsmodels available -- LMM enabled")',
    'except ImportError:',
    '    HAS_LMM = False',
    '    print("WARNING: statsmodels not available -- LMM skipped")',
    '',
    'try:',
    '    import mne',
    '    HAS_MNE = True',
    'except ImportError:',
    '    HAS_MNE = False',
    '    print("WARNING: mne not available -- topomap figure skipped")',
    '',
    '# -- Paths ----------------------------------------------------',
    'CLUSTER_BASE = Path("' + CLUSTER_BASE + '")',
    'MS_DIR       = CLUSTER_BASE / "microstates"',
    'RESULTS_DIR  = CLUSTER_BASE / "results"',
    'RESULTS_DIR.mkdir(parents=True, exist_ok=True)',
    'SFREQ_TAG    = "' + SFREQ_TAG + '"',
    'N_MICROSTATES = ' + repr(N_MICROSTATES),
    'PARAMS   = ["duration_ms", "occurrence_hz", "coverage_pct", "gev_pct"]',
    'ALPHA    = 0.05',
    '',
    '# -- Load microstate labels + channels + templates -------------',
    'assign_path = MS_DIR / ("assignments_" + SFREQ_TAG + ".json")',
    'if assign_path.exists():',
    '    with open(str(assign_path)) as f:',
    '        _asgn = json.load(f)',
    '    MS_LABELS = [_asgn.get(str(i), "MS" + chr(65 + i))',
    '                 for i in range(N_MICROSTATES)]',
    'else:',
    '    MS_LABELS = ["MS" + chr(65 + i) for i in range(N_MICROSTATES)]',
    'print("Labels:", MS_LABELS)',
    '',
    'templates = np.load(str(MS_DIR / ("templates_" + SFREQ_TAG + ".npy")))',
    'with open(str(MS_DIR / ("channels_" + SFREQ_TAG + ".json"))) as f:',
    '    REF_CHANNELS = json.load(f)',
    '',
    '# -- Helpers ----------------------------------------------------',
    'def cohens_d_paired(pre, post):',
    '    """Paired Cohen\'s d = mean(diff) / std(diff), diff = post - pre."""',
    '    diff = np.asarray(post) - np.asarray(pre)',
    '    if len(diff) < 2 or diff.std(ddof=1) == 0:',
    '        return float("nan")',
    '    return float(diff.mean() / diff.std(ddof=1))',
    '',
    'def sig_stars(p):',
    '    if p is None or (isinstance(p, float) and np.isnan(p)):',
    '        return ""',
    '    return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""',
    '',
    '# ================================================================',
    'print()',
    'print("=" * 60)',
    'print("Loading temporal_params.csv")',
    'print("=" * 60)',
    '',
    'df = pd.read_csv(str(RESULTS_DIR / "temporal_params.csv"))',
    'for p in PARAMS:',
    '    df[p] = pd.to_numeric(df[p], errors="coerce")',
    'print("Rows loaded: " + str(len(df)))',
    'print("Subjects:    " + str(df["subject"].nunique()))',
    '',
    '# Average the 2 runs within each condition (pre = runs 01+02,',
    '# post = runs 03+04) per subject x microstate x parameter.',
    'cond_means = (df.groupby(["subject", "condition", "microstate"])[PARAMS]',
    '                .mean().reset_index())',
    '',
    '# ================================================================',
    'print()',
    'print("=" * 60)',
    'print("PRIMARY: paired t-test (ttest_rel) + paired Cohen\'s d")',
    'print("=" * 60)',
    '',
    'stat_rows  = []',
    'raw_pvals  = []',
    '',
    'for param in PARAMS:',
    '    for ms in MS_LABELS:',
    '        wide = cond_means[cond_means["microstate"] == ms].pivot(',
    '            index="subject", columns="condition", values=param)',
    '        wide = wide.dropna(subset=["pre", "post"])',
    '        n = len(wide)',
    '        if n < 3:',
    '            stat_rows.append(dict(',
    '                parameter=param, microstate=ms,',
    '                mean_pre=None, mean_post=None, mean_diff=None,',
    '                t=None, p=None, p_fdr=None, cohens_d=None, n=n,',
    '            ))',
    '            raw_pvals.append(1.0)',
    '            continue',
    '        pre, post = wide["pre"].values, wide["post"].values',
    '        t, p = spstats.ttest_rel(post, pre)',
    '        d = cohens_d_paired(pre, post)',
    '        stat_rows.append(dict(',
    '            parameter=param, microstate=ms,',
    '            mean_pre=round(float(pre.mean()), 5),',
    '            mean_post=round(float(post.mean()), 5),',
    '            mean_diff=round(float(post.mean() - pre.mean()), 5),',
    '            t=round(float(t), 4), p=round(float(p), 6),',
    '            p_fdr=None, cohens_d=round(d, 4), n=n,',
    '        ))',
    '        raw_pvals.append(float(p))',
    '',
    '_, p_fdr, _, _ = multipletests(raw_pvals, alpha=ALPHA, method="fdr_bh")',
    'for row, pf in zip(stat_rows, p_fdr):',
    '    row["p_fdr"] = round(float(pf), 6)',
    '    row["sig_fdr"] = bool(pf < ALPHA)',
    '    stars = sig_stars(pf) if row["p"] is not None else ""',
    '    if row["p"] is not None:',
    '        print("  [ttest] " + row["parameter"] + " " + row["microstate"]',
    '              + "  diff=" + str(row["mean_diff"])',
    '              + "  t=" + str(row["t"])',
    '              + "  p_fdr=" + str(row["p_fdr"]) + "  " + stars)',
    '',
    '# ================================================================',
    'print()',
    'print("=" * 60)',
    'print("CONFIRMATORY: MixedLM  value ~ C(condition), groups=subject")',
    'print("=" * 60)',
    '',
    'if not HAS_LMM:',
    '    for row in stat_rows:',
    '        row.update(lmm_beta=None, lmm_se=None, lmm_z=None,',
    '                   lmm_p=None, lmm_p_fdr=None)',
    'else:',
    '    lmm_raw_p = []',
    '    lmm_cache = []',
    '    for param in PARAMS:',
    '        for ms in MS_LABELS:',
    '            dms = cond_means[cond_means["microstate"] == ms][',
    '                ["subject", "condition", param]].dropna().copy()',
    '            dms = dms.rename(columns={param: "value"})',
    '            dms["condition"] = pd.Categorical(',
    '                dms["condition"], categories=["pre", "post"])',
    '            if dms["subject"].nunique() < 3 or len(dms) < 6:',
    '                lmm_raw_p.append(1.0)',
    '                lmm_cache.append(None)',
    '                continue',
    '            try:',
    '                md = smf.mixedlm("value ~ C(condition)", dms,',
    '                                  groups=dms["subject"])',
    '                res = md.fit(reml=True, method="lbfgs", maxiter=300)',
    '                key = [k for k in res.params.index if "condition" in k][0]',
    '                beta = float(res.params[key])',
    '                se   = float(res.bse[key])',
    '                z    = float(res.tvalues[key])',
    '                pval = float(res.pvalues[key])',
    '                lmm_raw_p.append(pval if not np.isnan(pval) else 1.0)',
    '                lmm_cache.append((beta, se, z, pval))',
    '                print("  [lmm] " + param + " " + ms',
    '                      + "  beta=" + str(round(beta, 4))',
    '                      + "  z=" + str(round(z, 3))',
    '                      + "  p=" + str(round(pval, 4)))',
    '            except Exception as e:',
    '                print("  [lmm] " + param + " " + ms + "  FAILED: " + str(e))',
    '                lmm_raw_p.append(1.0)',
    '                lmm_cache.append(None)',
    '',
    '    _, lmm_p_fdr, _, _ = multipletests(lmm_raw_p, alpha=ALPHA, method="fdr_bh")',
    '    for row, res, pf in zip(stat_rows, lmm_cache, lmm_p_fdr):',
    '        if res is None:',
    '            row.update(lmm_beta=None, lmm_se=None, lmm_z=None,',
    '                       lmm_p=None, lmm_p_fdr=None)',
    '        else:',
    '            beta, se, z, pval = res',
    '            row.update(lmm_beta=round(beta, 5), lmm_se=round(se, 5),',
    '                       lmm_z=round(z, 4), lmm_p=round(pval, 6),',
    '                       lmm_p_fdr=round(float(pf), 6))',
    '',
    '# ================================================================',
    'print()',
    'print("Saving stats CSV...")',
    'stats_df = pd.DataFrame(stat_rows)',
    'stats_csv = RESULTS_DIR / "stats_pre_vs_post.csv"',
    'stats_df.to_csv(str(stats_csv), index=False)',
    'print("  " + str(stats_csv))',
    '',
    '# ================================================================',
    'print()',
    'print("Generating template topomap figure...")',
    'if HAS_MNE:',
    '    try:',
    '        montage = mne.channels.make_standard_montage("standard_1020")',
    '        info = mne.create_info(REF_CHANNELS, sfreq=' + str(500) + ',',
    '                                 ch_types="eeg")',
    '        info.set_montage(montage, on_missing="ignore")',
    '        fig, axes = plt.subplots(1, N_MICROSTATES,',
    '                                   figsize=(3 * N_MICROSTATES, 3.4))',
    '        for k in range(N_MICROSTATES):',
    '            ax = axes[k]',
    '            mne.viz.plot_topomap(templates[k], info, axes=ax,',
    '                                   show=False, cmap="RdBu_r")',
    '            ax.set_title(MS_LABELS[k], fontsize=11, fontweight="bold")',
    '        fig.suptitle("rtBPD nf1 rest microstate templates (k=7, pre+post pooled)",',
    '                      fontsize=12, fontweight="bold")',
    '        plt.tight_layout()',
    '        topo_path = RESULTS_DIR / "templates_topomap.png"',
    '        plt.savefig(str(topo_path), dpi=150, bbox_inches="tight")',
    '        plt.close(fig)',
    '        print("  " + str(topo_path))',
    '    except Exception as e:',
    '        print("  Topomap figure FAILED: " + str(e))',
    'else:',
    '    print("  Skipped (mne unavailable)")',
    '',
    '# ================================================================',
    'print()',
    'print("Generating pre-vs-post parameter figure...")',
    '',
    'PARAM_TITLES = {',
    '    "duration_ms":    "Mean duration (ms)",',
    '    "occurrence_hz":  "Occurrence rate (segments/s)",',
    '    "coverage_pct":   "Coverage (%)",',
    '    "gev_pct":        "GEV (%)",',
    '}',
    'x = np.arange(N_MICROSTATES)',
    'w_bar = 0.38',
    'C_PRE  = "#5080D0"',
    'C_POST = "#E07020"',
    '',
    'fig, axes = plt.subplots(2, 2, figsize=(14, 10))',
    'fig.suptitle("rtBPD nf1: microstate temporal parameters, pre vs post\\n"',
    '              "(paired t-test, Benjamini-Hochberg FDR corrected, n="',
    '              + str(cond_means["subject"].nunique()) + " subjects)",',
    '              fontsize=12, fontweight="bold")',
    '',
    'for ax, param in zip(axes.ravel(), PARAMS):',
    '    means_pre, sems_pre, means_post, sems_post, pfs = [], [], [], [], []',
    '    for ms in MS_LABELS:',
    '        wide = cond_means[cond_means["microstate"] == ms].pivot(',
    '            index="subject", columns="condition", values=param)',
    '        wide = wide.dropna(subset=["pre", "post"])',
    '        pre, post = wide["pre"].values, wide["post"].values',
    '        n = len(pre)',
    '        means_pre.append(pre.mean() if n else np.nan)',
    '        means_post.append(post.mean() if n else np.nan)',
    '        sems_pre.append(pre.std(ddof=1) / np.sqrt(n) if n > 1 else 0.0)',
    '        sems_post.append(post.std(ddof=1) / np.sqrt(n) if n > 1 else 0.0)',
    '        row = [r for r in stat_rows',
    '               if r["parameter"] == param and r["microstate"] == ms][0]',
    '        pfs.append(row["p_fdr"] if row["p_fdr"] is not None else 1.0)',
    '    means_pre  = np.array(means_pre);  sems_pre  = np.array(sems_pre)',
    '    means_post = np.array(means_post); sems_post = np.array(sems_post)',
    '',
    '    ax.bar(x - w_bar / 2, means_pre,  w_bar, color=C_PRE,  alpha=0.85,',
    '           label="pre",  yerr=sems_pre,  capsize=4)',
    '    ax.bar(x + w_bar / 2, means_post, w_bar, color=C_POST, alpha=0.85,',
    '           label="post", yerr=sems_post, capsize=4)',
    '',
    '    ymax = np.nanmax(np.concatenate([means_pre + sems_pre,',
    '                                       means_post + sems_post])) * 1.15',
    '    ymax = ymax if ymax > 0 else 1.0',
    '    for i in range(N_MICROSTATES):',
    '        s = sig_stars(pfs[i])',
    '        if s:',
    '            yb = max(means_pre[i] + sems_pre[i], means_post[i] + sems_post[i])',
    '            ax.text(x[i], yb + ymax * 0.03, s, ha="center", va="bottom",',
    '                     fontsize=13, fontweight="bold")',
    '',
    '    ax.set_xticks(x)',
    '    ax.set_xticklabels(MS_LABELS, fontsize=10)',
    '    ax.set_title(PARAM_TITLES[param], fontsize=11, fontweight="bold")',
    '    ax.legend(fontsize=9)',
    '    ax.tick_params(axis="y", labelsize=9)',
    '',
    'plt.tight_layout()',
    'bar_path = RESULTS_DIR / "pre_vs_post_params.png"',
    'plt.savefig(str(bar_path), dpi=150, bbox_inches="tight")',
    'plt.close(fig)',
    'print("  " + str(bar_path))',
    '',
    'print()',
    'print("=" * 60)',
    'print("DONE")',
    'print("=" * 60)',
]

# ── 2. Save cluster script locally ─────────────────────────
script_name = "03_stats_pre_vs_post_rtbpd_cluster.py"
script_path = SCRIPTS_DIR / script_name
script_path.parent.mkdir(parents=True, exist_ok=True)

with open(script_path, "w") as f:
    f.write("\n".join(lines))

# ── 3. Syntax check ────────────────────────────────────────
print("Checking syntax...")
try:
    py_compile.compile(str(script_path), doraise=True)
    print("Syntax OK: " + script_name)
except py_compile.PyCompileError as e:
    print("SYNTAX ERROR: " + str(e))
    raise

# ── 4. Deploy ──────────────────────────────────────────────
print("\nDeploying...")
make_cluster_dirs()
remote_script = CLUSTER_BASE + "/scripts/" + script_name
scp_to(script_path, remote_script, verbose=False)
print("Deployed: " + script_name)

# ── 5. Submit SLURM job ────────────────────────────────────
job_name = "rtbpd_ms_stats"
sbatch_lines = [
    "#!/bin/bash",
    "#SBATCH --job-name=" + job_name,
    "#SBATCH --output=" + CLUSTER_BASE + "/logs/" + job_name + "_%j.out",
    "#SBATCH --error="  + CLUSTER_BASE + "/logs/" + job_name + "_%j.err",
    "#SBATCH --partition=short",
    "#SBATCH --time=00:45:00",
    "#SBATCH --cpus-per-task=2",
    "#SBATCH --mem=16G",
    "#SBATCH --account=" + SLURM_ACCOUNT,
    "",
    PYTHON + " " + CLUSTER_BASE + "/scripts/" + script_name,
]

sbatch_name = "03_stats_pre_vs_post_rtbpd.sh"
sbatch_path = SCRIPTS_DIR / sbatch_name
with open(sbatch_path, "w") as f:
    f.write("\n".join(sbatch_lines))

remote_sbatch = CLUSTER_BASE + "/scripts/" + sbatch_name
scp_to(sbatch_path, remote_sbatch, verbose=False)

print("\nSubmitting SLURM job...")
result = run_ssh("sbatch " + remote_sbatch)
job_id = ""
for line in result.stdout.strip().split("\n"):
    if "Submitted" in line:
        job_id = line.strip().split()[-1]
        print("Job ID: " + job_id)

# ── 6. Monitor ─────────────────────────────────────────────
if job_id:
    print("\nMonitoring job " + job_id + "  (Ctrl+C to stop)")
    print("-" * 55)
    try:
        while True:
            r = run_ssh(
                "squeue -j " + job_id
                + " --format=%.8i_%.8T_%.10M 2>/dev/null",
                verbose=False
            )
            status = r.stdout.strip()
            if status and "JOBID" not in status.split("\n")[-1]:
                print(status)
            else:
                print("Job finished — checking log...")
                log = run_ssh(
                    "tail -60 " + CLUSTER_BASE
                    + "/logs/" + job_name + "_" + job_id
                    + ".out 2>/dev/null",
                    verbose=False
                )
                print(log.stdout)
                break
            time.sleep(20)
    except KeyboardInterrupt:
        print("\nStopped watching.")
        print("  tail -f " + CLUSTER_BASE
              + "/logs/" + job_name + "_" + job_id + ".out")
