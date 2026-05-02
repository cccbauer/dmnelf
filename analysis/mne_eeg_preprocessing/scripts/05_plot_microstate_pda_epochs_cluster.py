#!/usr/bin/env python3
"""
05_plot_microstate_pda_epochs_cluster.py
Visualize EEG microstate T-hat projections during PDA+ vs PDA- fMRI epochs.
"""
import sys
sys.stdout.reconfigure(line_buffering=True)
import numpy as np
import json
from pathlib import Path
import warnings
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import mne
mne.set_log_level("WARNING")

# ── Paths ─────────────────────────────────────────────────
CLUSTER_BASE    = Path("/projects/swglab/data/DMNELF/analysis/MNE/jupyter/microstate_pda_v3")
EEG_ROOT        = Path("/projects/swglab/data/DMNELF/derivatives/eeg_preprocessed")
MICROSTATES_DIR = CLUSTER_BASE / "microstates"
FEATURES_DIR    = CLUSTER_BASE / "features"
TARGETS_DIR     = CLUSTER_BASE / "targets"
PLOTS_ROOT      = Path("/projects/swglab/data/DMNELF/derivatives/microstate_pda_plots")
PLOTS_ROOT.mkdir(parents=True, exist_ok=True)

# ── Config ────────────────────────────────────────────────
OVERWRITE     = True
SUBJECTS      = ['sub-dmnelf001', 'sub-dmnelf004', 'sub-dmnelf005', 'sub-dmnelf006', 'sub-dmnelf007', 'sub-dmnelf008', 'sub-dmnelf009', 'sub-dmnelf010', 'sub-dmnelf011', 'sub-dmnelf1001', 'sub-dmnelf1002', 'sub-dmnelf1003']
MISSING_RUNS  = {'sub-dmnelf1002': [('rest', '02')], 'sub-dmnelf1003': [('feedback', '04')]}
SFREQ_TAG     = "250Hz"
TASKS_ORDERED = ["rest", "shortrest", "feedback"]
TASK_RUNS = {
    "rest":      ["01", "02"],
    "shortrest": ["01"],
    "feedback":  ["01", "02", "03", "04"],
}
COLOR_POS = "#FF1A00"
COLOR_NEG = "#0077FF"

# ── Load templates + assignments ──────────────────────────
print("=" * 55)
print("Loading templates")
print("=" * 55)

templates_path   = MICROSTATES_DIR / "templates_250Hz.npy"
assignments_path = MICROSTATES_DIR / "assignments_250Hz.json"

if not templates_path.exists():
    print("MISSING: " + str(templates_path))
    sys.exit(1)

templates = np.load(str(templates_path))  # (7, n_ch)
with open(str(assignments_path)) as f:
    assignments = json.load(f)
labels = [assignments.get(str(i), "MS" + str(i)) for i in range(7)]
print("Templates shape: " + str(templates.shape))
print("Labels: " + str(labels))

# ── Load MNE info matching templates channel count ────────
# templates.shape[1] is authoritative — find a FIF with the same n_ch
# Always re-apply standard_1020 montage so positions are guaranteed.
def _load_raw_channels(subject, tag):
    fif_path = (EEG_ROOT / subject / "ses-dmnelf" / "eeg"
               / (subject + "_ses-dmnelf_task-rest_run-01_desc-preproc"
                  + tag + "_eeg.fif"))
    if not fif_path.exists():
        return None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        raw = mne.io.read_raw_fif(str(fif_path), preload=False, verbose=False)
    drop = [ch for ch in raw.ch_names
            if any(x in ch.upper()
                   for x in ("ECG","EKG","EMG","EOG","STIM","STATUS","TP9","TP10"))]
    if drop:
        raw.drop_channels(drop)
    # Always set standard_1020 so electrode positions are present
    try:
        montage = mne.channels.make_standard_montage("standard_1020")
        raw.set_montage(montage, on_missing="ignore", verbose=False)
    except Exception as e:
        print("  montage warning: " + str(e))
    return raw

mne_info  = None
_n_tpl    = templates.shape[1]
print("Templates n_ch: " + str(_n_tpl))
for _sub in SUBJECTS:
    for _tag in ("500Hz", "250Hz"):
        _raw = _load_raw_channels(_sub, _tag)
        if _raw is None:
            continue
        print("  FIF " + _sub + " (" + _tag + ") n_ch=" + str(len(_raw.ch_names)))
        if len(_raw.ch_names) == _n_tpl:
            mne_info = _raw.info
            print("MNE info SET: " + _sub + " (" + _tag + ")"
                  + "  n_ch=" + str(len(_raw.ch_names)))
            break
    if mne_info is not None:
        break
if mne_info is None:
    print("WARNING: No FIF with " + str(_n_tpl) + " ch found — topomaps will be blank")

# ── Plotting helpers ──────────────────────────────────────
def make_topomap(ax, values, info, title="", cmap="RdBu_r", vlim=None):
    if info is None:
        ax.text(0.5, 0.5, "no info", ha="center", va="center",
                transform=ax.transAxes, color="gray", fontsize=9)
        ax.set_title(title, fontsize=8)
        return
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mne.viz.plot_topomap(
                values, info, axes=ax, show=False,
                cmap=cmap, vlim=vlim, contours=4
            )
    except Exception as _e:
        ax.text(0.5, 0.5, "ERR: " + str(_e)[:60], ha="center", va="center",
                transform=ax.transAxes, color="red", fontsize=7, wrap=True)
        print("  topomap error [" + title + "]: " + str(_e))
    ax.set_title(title, fontsize=8)

def make_bar_chart(ax, mean_pos, mean_neg, labels, title="",
                   sem_pos=None, sem_neg=None):
    x = np.arange(len(labels))
    w = 0.35
    ax.bar(x - w/2, mean_pos, w, color=COLOR_POS, alpha=0.85, label="PDA+",
           yerr=sem_pos, capsize=3, error_kw={"elinewidth": 1})
    ax.bar(x + w/2, mean_neg, w, color=COLOR_NEG, alpha=0.85, label="PDA-",
           yerr=sem_neg, capsize=3, error_kw={"elinewidth": 1})
    ax.axhline(0, color="black", lw=0.5, ls="--")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=7, rotation=30, ha="right")
    ax.set_ylabel("mean T-hat", fontsize=7)
    ax.set_title(title, fontsize=9, fontweight="bold")
    ax.legend(fontsize=6, loc="upper right")
    ax.tick_params(axis="y", labelsize=7)

# ── Main loop ─────────────────────────────────────────────
print()
print("=" * 55)
print("Processing subjects")
print("=" * 55)

all_subjects_mean_pos = []
all_subjects_mean_neg = []

for subject in SUBJECTS:
    print()
    print("--- " + subject + " ---")

    _plot_dir = PLOTS_ROOT / subject
    _plot_dir.mkdir(parents=True, exist_ok=True)
    _out_png  = _plot_dir / (subject + "_microstate_pda_epochs.png")

    # Accumulate T-hat by polarity across all tasks/runs
    task_mean_pos = {}
    task_mean_neg = {}
    all_pos = []
    all_neg = []

    for task in TASKS_ORDERED:
        _pos_list = []
        _neg_list = []
        for run in TASK_RUNS[task]:
            if subject in MISSING_RUNS:
                if (task, run) in MISSING_RUNS[subject]:
                    continue
            feat_path = (FEATURES_DIR / (subject + "_task-" + task
                          + "_run-" + run + "_250Hz_nohrf_features.npy"))
            pda_path  = (TARGETS_DIR  / (subject + "_task-" + task
                          + "_run-" + run + "_pda_direct.npy"))
            if not feat_path.exists() or not pda_path.exists():
                continue
            feats = np.load(str(feat_path))  # (n_vols, 9)
            pda   = np.load(str(pda_path))   # (n_vols,)
            n     = min(len(feats), len(pda))
            feats, pda = feats[:n], pda[:n]
            _pos_list.append(feats[pda >= 0, :7])
            _neg_list.append(feats[pda <  0, :7])
        if _pos_list:
            _cat_pos = np.concatenate(_pos_list)
            _cat_neg = np.concatenate(_neg_list)
            task_mean_pos[task] = _cat_pos.mean(axis=0)
            task_mean_neg[task] = _cat_neg.mean(axis=0)
            print("  " + task + ": n_pos=" + str(len(_cat_pos))
                  + "  n_neg=" + str(len(_cat_neg)))
            all_pos.append(_cat_pos)
            all_neg.append(_cat_neg)

    if not all_pos:
        print("  NO DATA — skipping")
        continue

    mean_pos_all = np.concatenate(all_pos).mean(axis=0)  # (7,)
    mean_neg_all = np.concatenate(all_neg).mean(axis=0)
    all_subjects_mean_pos.append(mean_pos_all)
    all_subjects_mean_neg.append(mean_neg_all)

    if _out_png.exists() and not OVERWRITE:
        print("  EXISTS (skip plot, accumulated for group)")
        continue

    # ── Per-subject figure: 2 rows × 4 cols ───────────────
    # Row 0: bar charts (rest | shortrest | feedback | ALL)
    # Row 1: topomaps  (PDA+ | PDA-  | difference | empty)
    fig = plt.figure(figsize=(16, 7))
    fig.suptitle(
        subject + "  |  Microstate T-hat: PDA+ (red) vs PDA- (blue)",
        fontsize=11, fontweight="bold"
    )
    for _ti, _task in enumerate(TASKS_ORDERED):
        _ax = fig.add_subplot(2, 4, _ti + 1)
        if _task in task_mean_pos:
            make_bar_chart(_ax, task_mean_pos[_task], task_mean_neg[_task],
                           labels, title=_task)
        else:
            _ax.text(0.5, 0.5, "missing", ha="center", va="center",
                     transform=_ax.transAxes, color="gray", fontsize=9)
            _ax.set_title(_task, fontsize=9)
    make_bar_chart(fig.add_subplot(2, 4, 4), mean_pos_all, mean_neg_all,
                   labels, title="ALL tasks")

    _topo_pos  = templates.T @ mean_pos_all
    _topo_neg  = templates.T @ mean_neg_all
    _topo_diff = _topo_pos - _topo_neg
    _vlim  = max(float(np.abs(_topo_pos).max()), float(np.abs(_topo_neg).max()))
    _dvlim = float(np.abs(_topo_diff).max())

    make_topomap(fig.add_subplot(2, 4, 5), _topo_pos, mne_info,
                 title="PDA+ topography", cmap="RdBu_r", vlim=(-_vlim, _vlim))
    make_topomap(fig.add_subplot(2, 4, 6), _topo_neg, mne_info,
                 title="PDA- topography", cmap="RdBu_r", vlim=(-_vlim, _vlim))
    make_topomap(fig.add_subplot(2, 4, 7), _topo_diff, mne_info,
                 title="Difference (PDA+ - PDA-)", cmap="RdBu_r",
                 vlim=(-_dvlim, _dvlim))
    fig.add_subplot(2, 4, 8).axis("off")

    plt.tight_layout()
    plt.savefig(str(_out_png), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved: " + _out_png.name)

# ── Group figure ──────────────────────────────────────────
print()
print("=" * 55)
print("Generating group figure")
print("=" * 55)

if len(all_subjects_mean_pos) >= 2:
    _gp   = np.stack(all_subjects_mean_pos)  # (n_sub, 7)
    _gn   = np.stack(all_subjects_mean_neg)
    _gm_p = _gp.mean(axis=0)
    _gm_n = _gn.mean(axis=0)
    _se_p = _gp.std(axis=0) / np.sqrt(len(_gp))
    _se_n = _gn.std(axis=0) / np.sqrt(len(_gn))

    _out_grp = PLOTS_ROOT / "group_microstate_pda_epochs.png"
    fig_g    = plt.figure(figsize=(18, 5))
    fig_g.suptitle(
        "Group (n=" + str(len(_gp)) + ")"
        + "  |  Microstate T-hat: PDA+ (red) vs PDA- (blue)",
        fontsize=12, fontweight="bold"
    )
    _gs     = gridspec.GridSpec(1, 7, figure=fig_g)
    _ax_bar = fig_g.add_subplot(_gs[0, :4])
    make_bar_chart(_ax_bar, _gm_p, _gm_n, labels,
                   title="Mean ± SEM across subjects",
                   sem_pos=_se_p, sem_neg=_se_n)

    _gt_p = templates.T @ _gm_p
    _gt_n = templates.T @ _gm_n
    _gt_d = _gt_p - _gt_n
    _gvl  = max(float(np.abs(_gt_p).max()), float(np.abs(_gt_n).max()))
    _gdvl = float(np.abs(_gt_d).max())

    make_topomap(fig_g.add_subplot(_gs[0, 4]), _gt_p, mne_info,
                 title="Group PDA+ topo", cmap="RdBu_r", vlim=(-_gvl, _gvl))
    make_topomap(fig_g.add_subplot(_gs[0, 5]), _gt_n, mne_info,
                 title="Group PDA- topo", cmap="RdBu_r", vlim=(-_gvl, _gvl))
    make_topomap(fig_g.add_subplot(_gs[0, 6]), _gt_d, mne_info,
                 title="Group difference", cmap="RdBu_r", vlim=(-_gdvl, _gdvl))

    plt.tight_layout()
    plt.savefig(str(_out_grp), dpi=150, bbox_inches="tight")
    plt.close(fig_g)
    print("  Saved: " + str(_out_grp))
else:
    print("  Too few subjects (" + str(len(all_subjects_mean_pos)) + ") — skipping")

print()
print("=" * 55)
print("DONE")
print("=" * 55)