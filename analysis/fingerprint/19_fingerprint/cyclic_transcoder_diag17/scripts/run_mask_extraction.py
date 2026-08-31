"""
run_mask_extraction.py
----------------------
Runs ICA-based mask extraction for subjects missing personal DMN/CEN masks
in the network_masks directory. Uses the same pineuro pipeline that generated
existing masks for all other subjects.

Outputs match the naming convention of existing masks:
    sub-{subject}_space-MNI152NLin6Asym_res-2_dmn_mask.nii.gz
    sub-{subject}_space-MNI152NLin6Asym_res-2_cen_mask.nii.gz

Usage:
    python run_mask_extraction.py --subject dmnelf012 --config config.yaml
    python run_mask_extraction.py --subject dmnelf013 --config config.yaml
    python run_mask_extraction.py --all-missing --config config.yaml
"""

import argparse
import shutil
import sys
from pathlib import Path

import yaml


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def get_fmriprep_inputs(cfg, subject):
    """
    Find fMRIPrep outputs needed for mask extraction.
    Returns (bold_path, mask_path, boldref_path) or raises FileNotFoundError.

    Uses resting-state run-01 as input (longest, best SNR for ICA).
    All files in MNI152NLin6Asym res-2 space — matches existing masks.
    """
    fprep   = Path(cfg["data"]["fmriprep_dir"])
    ses     = cfg["data"]["session"]          # ses-dmnelf
    space   = cfg["data"]["fmri"]["space"]    # MNI152NLin6Asym
    res     = cfg["data"]["fmri"]["res"]      # 2
    sub     = f"sub-{subject}"
    func_dir = fprep / sub / ses / "func"

    if not func_dir.exists():
        raise FileNotFoundError(f"func dir not found: {func_dir}")

    # Prefer run-01 rest; fall back to run-02 if needed
    bold_path = None
    for run in ["run-01", "run-02"]:
        candidate = (
            func_dir /
            f"{sub}_{ses}_task-rest_{run}"
            f"_space-{space}_res-{res}_desc-preproc_bold.nii.gz"
        )
        if candidate.exists():
            bold_path = candidate
            run_used = run
            break

    if bold_path is None:
        raise FileNotFoundError(f"No rest BOLD found in {func_dir}")

    mask_path = (
        func_dir /
        f"{sub}_{ses}_task-rest_{run_used}"
        f"_space-{space}_res-{res}_desc-brain_mask.nii.gz"
    )
    if not mask_path.exists():
        raise FileNotFoundError(f"Brain mask not found: {mask_path}")

    # boldref in MNI space — same space as BOLD, used as registration reference
    boldref_path = (
        func_dir /
        f"{sub}_{ses}_task-rest_{run_used}"
        f"_space-{space}_res-{res}_boldref.nii.gz"
    )
    if not boldref_path.exists():
        raise FileNotFoundError(f"boldref not found: {boldref_path}")

    return bold_path, mask_path, boldref_path


def mask_already_exists(cfg, subject):
    """Check if both DMN and CEN masks already exist for this subject."""
    mask_dir = (
        Path(cfg["data"]["personal_masks_dir"]) / f"sub-{subject}"
    )
    m = cfg["data"]["masks"]
    space, res = m["space"], m["res"]
    prefix = f"sub-{subject}_space-{space}_res-{res}"
    dmn = mask_dir / f"{prefix}_{m['dmn_suffix']}"
    cen = mask_dir / f"{prefix}_{m['cen_suffix']}"
    return dmn.exists() and cen.exists()


def run_extraction(subject, cfg, overwrite=False):
    """Run mask extraction for one subject and copy outputs to network_masks."""

    print(f"\n{'='*60}")
    print(f"  Mask extraction: {subject}")
    print(f"{'='*60}")

    # ── Import pineuro pipeline ─────────────────────────────────────────
    try:
        from pineuro.mask_extraction import mask_extraction_rest
    except ImportError:
        print("ERROR: pineuro not importable.")
        print("  Activate the environment that has pineuro installed:")
        print("  conda activate <pineuro_env> && python run_mask_extraction.py ...")
        sys.exit(1)

    # ── Find inputs ─────────────────────────────────────────────────────
    bold_path, mask_path, boldref_path = get_fmriprep_inputs(cfg, subject)
    print(f"  BOLD    : {bold_path.name}")
    print(f"  Mask    : {mask_path.name}")
    print(f"  Boldref : {boldref_path.name}")

    # ── Set up output directories ────────────────────────────────────────
    work_dir = (
        Path(cfg["data"]["personal_masks_dir"])
        / f"sub-{subject}"
        / "mask_extraction_work"
    )
    work_dir.mkdir(parents=True, exist_ok=True)

    out_dir = Path(cfg["data"]["personal_masks_dir"]) / f"sub-{subject}"
    out_dir.mkdir(parents=True, exist_ok=True)

    space = cfg["data"]["fmri"]["space"]  # MNI152NLin6Asym
    res   = cfg["data"]["fmri"]["res"]    # 2
    sub   = f"sub-{subject}"
    prefix = f"{sub}_space-{space}_res-{res}"

    dmn_out = out_dir / f"{prefix}_{cfg['data']['masks']['dmn_suffix']}"
    cen_out = out_dir / f"{prefix}_{cfg['data']['masks']['cen_suffix']}"

    if dmn_out.exists() and cen_out.exists() and not overwrite:
        print(f"  Masks already exist — skipping (use --overwrite to force)")
        return

    # ── Run ICA mask extraction ──────────────────────────────────────────
    # Yeo-17: DefaultA = DMN, ContA = CEN
    # Matches network targets used for all other DMNELF subjects
    print(f"  Running ICA mask extraction (this takes ~10-20 min)...")

    def progress(step, detail=""):
        msg = f"  [{step}]"
        if detail:
            msg += f" {detail}"
        print(msg)

    thr_files = mask_extraction_rest(
        participant_id=sub,
        root_dir=work_dir,
        input_file=bold_path,
        mask_file=mask_path,
        ref_file=boldref_path,
        n_yeo=17,
        yeo_targets=["DefaultA", "ContA"],
        network_atlas="yeo_17",
        n_components=30,
        threshold=2000,
        overwrite=overwrite,
        qc_enabled=True,
        progress_callback=progress,
    )

    # ── Copy and rename outputs to match network_masks convention ────────
    # mask_extraction_rest outputs:
    #   work_dir/03_masks/{sub}_DefaultA_thr.nii.gz  → DMN mask
    #   work_dir/03_masks/{sub}_ContA_thr.nii.gz     → CEN mask

    masks_dir = work_dir / "03_masks"
    network_map = {
        "DefaultA": dmn_out,
        "ContA":    cen_out,
    }

    print(f"\n  Copying outputs to {out_dir.name}/")
    for network, dest in network_map.items():
        src = masks_dir / f"{sub}_{network}_thr.nii.gz"
        if not src.exists():
            print(f"  WARNING: expected output not found: {src.name}")
            continue
        shutil.copy2(src, dest)
        print(f"  {OK} {dest.name}")

    # Also copy the parcel_weights JSON and masks overview if present
    for extra in masks_dir.glob("*.json"):
        shutil.copy2(extra, out_dir / extra.name)

    # Copy QC PNG if generated
    qc_png = work_dir.parent / "qc" / "mask_extraction_qc.html"
    if qc_png.exists():
        shutil.copy2(qc_png, out_dir / f"{sub}_mask_extraction_qc.html")
        print(f"  QC report copied")

    print(f"\n  Done: {subject}")
    print(f"  DMN mask : {dmn_out.stat().st_size // 1024} KB")
    print(f"  CEN mask : {cen_out.stat().st_size // 1024} KB")

    return dmn_out, cen_out


# ANSI colour for terminal output
OK = "\033[92m✓\033[0m" if sys.stdout.isatty() else "✓"


def main():
    parser = argparse.ArgumentParser(
        description="Run ICA mask extraction for subjects missing DMN/CEN masks"
    )
    parser.add_argument("--subject",     type=str, default=None,
                        help="Single subject ID, e.g. dmnelf012")
    parser.add_argument("--all-missing", action="store_true",
                        help="Run for all subjects missing masks")
    parser.add_argument("--config",      default="config.yaml")
    parser.add_argument("--overwrite",   action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    excluded = set(cfg["data"]["subjects"].get("exclude", []))

    if args.all_missing:
        # Find all subjects listed in config that are missing masks
        # Check both paired and no_masks lists
        all_subs = (
            cfg["data"]["subjects"].get("all", [])
            + cfg["data"]["subjects"].get("no_masks", [])
        )
        targets = [
            s for s in dict.fromkeys(all_subs)
            if s not in excluded and not mask_already_exists(cfg, s)
        ]
        if not targets:
            print("All subjects already have masks.")
            return
        print(f"Subjects missing masks: {targets}")
        for subj in targets:
            try:
                run_extraction(subj, cfg, overwrite=args.overwrite)
            except Exception as e:
                print(f"  ERROR for {subj}: {e}")

    elif args.subject:
        run_extraction(args.subject, cfg, overwrite=args.overwrite)

    else:
        # Default: show who is missing
        print("\nSubjects missing masks:")
        all_subs = (
            cfg["data"]["subjects"].get("all", [])
            + cfg["data"]["subjects"].get("no_masks", [])
        )
        for s in dict.fromkeys(all_subs):
            if s in excluded:
                continue
            exists = mask_already_exists(cfg, s)
            sym = OK if exists else "\033[91m✗\033[0m"
            print(f"  {sym}  {s}")
        print("\nRun with --subject dmnelf012 or --all-missing")


if __name__ == "__main__":
    main()
