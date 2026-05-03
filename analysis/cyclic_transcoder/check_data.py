"""
check_data.py
-------------
Audits all data prerequisites for the cyclic transcoder pipeline.
Run this interactively in VS Code before submitting any SLURM jobs.

Usage:
    python check_data.py --config config.yaml
    python check_data.py --config config.yaml --verbose

Output:
    Prints a colour-coded table per subject and a summary at the end.
    Writes check_data_report.tsv for inspection in Excel / pandas.
"""

import argparse
import importlib
import sys
from pathlib import Path

import yaml

# ── ANSI colours (disabled if not a terminal) ────────────────────────────────
USE_COLOR = sys.stdout.isatty()
def green(s):  return f"\033[92m{s}\033[0m" if USE_COLOR else s
def red(s):    return f"\033[91m{s}\033[0m" if USE_COLOR else s
def yellow(s): return f"\033[93m{s}\033[0m" if USE_COLOR else s
def bold(s):   return f"\033[1m{s}\033[0m"  if USE_COLOR else s

OK   = green("✓")
MISS = red("✗")
WARN = yellow("?")


# ── Config ────────────────────────────────────────────────────────────────────

def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)

def subject_list(cfg):
    excluded = set(cfg["data"]["subjects"].get("exclude", []))
    return [s for s in cfg["data"]["subjects"]["all"] if s not in excluded]

def all_tasks(cfg):
    d = cfg["data"]["tasks"]
    seen, out = set(), []
    for k in ("train", "validate", "apply"):
        for t in (d.get(k) or []):
            if t not in seen:
                seen.add(t)
                out.append(t)
    return out


# ── fMRI checks ───────────────────────────────────────────────────────────────

def find_bold(fprep_root, subject, task):
    """
    Return list of (bold_path, confounds_path | None) found for subject/task.
    Walks all ses-* directories and all run-* variants.
    """
    found = []
    sub_dir = Path(fprep_root) / f"sub-{subject}"
    if not sub_dir.exists():
        return found

    for ses_dir in sorted(sub_dir.glob("ses-*")):
        func_dir = ses_dir / "func"
        if not func_dir.exists():
            continue
        # Match any space/res variant
        for bold in sorted(func_dir.glob(f"*task-{task}*_desc-preproc_bold.nii.gz")):
            conf = bold.parent / bold.name.replace(
                "_desc-preproc_bold.nii.gz",
                "_desc-confounds_timeseries.tsv",
            )
            found.append((bold, conf if conf.exists() else None))
    return found


def check_bold_volumes(bold_path, expected_vols):
    """Quick header check — avoids loading the full image."""
    try:
        import nibabel as nib
        img = nib.load(str(bold_path))
        n = img.shape[3] if len(img.shape) == 4 else None
        return n, (n == expected_vols)
    except Exception as e:
        return None, False


# ── EEG checks ────────────────────────────────────────────────────────────────

def find_eeg(eeg_root, subject, task):
    """Return list of preprocessed .fif paths for subject/task."""
    found = []
    sub_dir = Path(eeg_root) / f"sub-{subject}"
    if not sub_dir.exists():
        return found
    for ses_dir in sorted(sub_dir.glob("ses-*")):
        eeg_dir = ses_dir / "eeg"
        if not eeg_dir.exists():
            # Also try flat structure
            eeg_dir = ses_dir
        for fif in sorted(eeg_dir.glob(f"*task-{task}*.fif")):
            found.append(fif)
    return found


def check_eeg_channels(fif_path, expected_ch):
    """Quick header check on .fif without loading data."""
    try:
        import mne
        info = mne.io.read_info(str(fif_path), verbose=False)
        n = len([c for c in info["ch_names"] if c not in (info.get("bads") or [])])
        return n, True   # just confirm it's readable
    except Exception as e:
        return None, False


# ── Personal mask checks ──────────────────────────────────────────────────────

def check_masks(mask_root, subject):
    dmn = Path(mask_root) / f"sub-{subject}" / "DMN.nii.gz"
    cen = Path(mask_root) / f"sub-{subject}" / "CEN.nii.gz"
    results = {}
    for label, path in [("DMN", dmn), ("CEN", cen)]:
        if path.exists():
            try:
                import nibabel as nib
                img = nib.load(str(path))
                n_vox = (img.get_fdata() > 0).sum()
                results[label] = (True, int(n_vox), str(path))
            except Exception:
                results[label] = (False, 0, str(path))
        else:
            results[label] = (False, 0, str(path))
    return results


# ── Already-extracted features check ─────────────────────────────────────────

def check_features(features_dir, subject, task):
    sub_dir = Path(features_dir) / f"sub-{subject}"
    npzs = list(sub_dir.glob(f"*task-{task}*_features.npz")) if sub_dir.exists() else []
    return npzs


# ── Python environment check ──────────────────────────────────────────────────

def check_environment():
    packages = {
        "torch":    "PyTorch (deep learning)",
        "mne":      "MNE-Python (EEG I/O)",
        "nilearn":  "Nilearn (fMRI parcellation)",
        "nibabel":  "NiBabel (NIfTI I/O)",
        "numpy":    "NumPy",
        "yaml":     "PyYAML",
        "pandas":   "Pandas (confounds TSV)",
    }
    results = {}
    for pkg, label in packages.items():
        try:
            mod = importlib.import_module(pkg)
            ver = getattr(mod, "__version__", "?")
            results[pkg] = (True, ver, label)
        except ImportError:
            results[pkg] = (False, None, label)
    return results


# ── Main audit ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Audit data for cyclic transcoder")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--verbose", action="store_true",
                        help="Print every file path found")
    args = parser.parse_args()

    cfg = load_config(args.config)
    subjects = subject_list(cfg)
    tasks = all_tasks(cfg)

    fprep_root   = cfg["data"]["fmriprep_dir"]
    eeg_root     = cfg["data"]["eeg_preproc_dir"]
    mask_root    = cfg["data"]["personal_masks_dir"]
    features_dir = cfg["data"]["features_dir"]
    expected_rest_vols = cfg["data"]["fmri"]["n_volumes_rest"]
    expected_ch  = cfg["data"]["eeg"]["n_channels"]

    print(bold("\n══════════════════════════════════════════════"))
    print(bold(" Cyclic Transcoder — Data Audit"))
    print(bold("══════════════════════════════════════════════\n"))

    # ── Environment ──────────────────────────────────────────────────────────
    print(bold("[ Python environment ]"))
    env = check_environment()
    env_ok = True
    for pkg, (found, ver, label) in env.items():
        if found:
            print(f"  {OK}  {label:<35} {ver}")
        else:
            marker = MISS if pkg in ("torch", "mne", "nilearn", "nibabel", "numpy") else WARN
            print(f"  {marker}  {label:<35} NOT FOUND")
            if pkg in ("torch", "mne", "nilearn", "nibabel"):
                env_ok = False
    if not env_ok:
        print(yellow("\n  ⚠  Missing critical packages. Install before proceeding:"))
        missing = [p for p, (f, *_) in env.items() if not f]
        print(f"     pip install {' '.join(missing)} --break-system-packages")
    print()

    # ── Root directories ─────────────────────────────────────────────────────
    print(bold("[ Root directories ]"))
    for label, path in [
        ("fMRIPrep output", fprep_root),
        ("EEG preprocessed", eeg_root),
        ("Personal masks", mask_root),
        ("Features output", features_dir),
    ]:
        exists = Path(path).exists()
        sym = OK if exists else MISS
        print(f"  {sym}  {label:<25} {path}")
    print()

    # ── Per-subject audit ────────────────────────────────────────────────────
    print(bold("[ Per-subject data ]"))
    col_w = 12

    # Header
    header = f"  {'Subject':<16}"
    for task in tasks:
        header += f"  {'BOLD-'+task:<{col_w}}  {'EEG-'+task:<{col_w}}"
    header += f"  {'DMN':<6}  {'CEN':<6}  {'Feat':<6}"
    print(bold(header))
    print("  " + "─" * (len(header) - 2))

    tsv_rows = []
    issues = []

    for subj in subjects:
        row = f"  {subj:<16}"
        tsv_row = {"subject": subj}

        for task in tasks:
            # BOLD
            bolds = find_bold(fprep_root, subj, task)
            if bolds:
                n_runs = len(bolds)
                # Check volume count for rest task
                if task in cfg["data"]["tasks"].get("train", []):
                    vol_issues = []
                    for bp, _ in bolds:
                        n_vols, ok = check_bold_volumes(bp, expected_rest_vols)
                        if not ok:
                            vol_issues.append(f"{n_vols}")
                        if args.verbose:
                            sym = OK if ok else WARN
                            print(f"\n      {sym} {bp.name}  (vols={n_vols})")
                    if vol_issues:
                        row += f"  {yellow(f'{n_runs}run/V!'):<{col_w+9}}"
                        issues.append(f"{subj} task={task}: unexpected volumes {vol_issues}")
                    else:
                        row += f"  {green(f'{n_runs} run(s)'):<{col_w+9}}"
                else:
                    row += f"  {green(f'{n_runs} run(s)'):<{col_w+9}}"
                    if args.verbose:
                        for bp, cp in bolds:
                            print(f"\n      {OK} {bp.name}")
                tsv_row[f"bold_{task}"] = n_runs
            else:
                row += f"  {red('MISSING'):<{col_w+9}}"
                tsv_row[f"bold_{task}"] = 0
                issues.append(f"{subj} task={task}: BOLD not found in {fprep_root}")

            # EEG
            eeg_files = find_eeg(eeg_root, subj, task)
            if eeg_files:
                n_runs = len(eeg_files)
                row += f"  {green(f'{n_runs} run(s)'):<{col_w+9}}"
                if args.verbose:
                    for fp in eeg_files:
                        print(f"\n      {OK} {fp.name}")
                tsv_row[f"eeg_{task}"] = n_runs
            else:
                row += f"  {red('MISSING'):<{col_w+9}}"
                tsv_row[f"eeg_{task}"] = 0
                issues.append(f"{subj} task={task}: EEG .fif not found in {eeg_root}")

        # Masks
        masks = check_masks(mask_root, subj)
        for label in ("DMN", "CEN"):
            found, n_vox, path = masks[label]
            if found:
                row += f"  {green(str(n_vox)):<15}"
            else:
                row += f"  {red('MISS'):<15}"
                issues.append(f"{subj}: {label} mask missing at {path}")
            tsv_row[f"mask_{label}_voxels"] = n_vox if found else 0

        # Already-extracted features
        feat_files = []
        for task in tasks:
            feat_files += check_features(features_dir, subj, task)
        n_feat = len(feat_files)
        row += f"  {green(str(n_feat)) if n_feat > 0 else yellow('0'):<15}"
        tsv_row["features_extracted"] = n_feat

        print(row)
        tsv_rows.append(tsv_row)

    # ── Summary ──────────────────────────────────────────────────────────────
    print()
    if issues:
        print(bold(yellow(f"[ {len(issues)} issue(s) found ]")))
        for i, iss in enumerate(issues, 1):
            print(f"  {i:3d}. {iss}")
    else:
        print(bold(green("[ All checks passed — ready to run extract_job.sh ]")))

    # ── Write TSV report ─────────────────────────────────────────────────────
    import csv
    report_path = Path("check_data_report.tsv")
    if tsv_rows:
        fieldnames = list(tsv_rows[0].keys())
        with open(report_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
            writer.writeheader()
            writer.writerows(tsv_rows)
        print(f"\n  Report written to: {report_path.resolve()}\n")

    # Return exit code 1 if critical issues
    critical = [i for i in issues if "BOLD" in i or "EEG" in i]
    sys.exit(1 if critical else 0)


if __name__ == "__main__":
    main()