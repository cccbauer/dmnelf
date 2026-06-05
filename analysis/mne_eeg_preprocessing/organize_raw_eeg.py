#!/usr/bin/env python3
"""
organize_raw_eeg.py
Organize raw EEG files from Brain Vision export to BIDS-compliant structure.

Current state:
  /projects/swglab/data/DMNELF/sourcedata/eeg_data/eeg_preprocessed/
    sub-dmnelf013_task_feedback-run03_Segment 1.edf

Target state:
  /projects/swglab/data/DMNELF/rawdata_eeg/
    sub-dmnelf013/ses-dmnelf/eeg/
      sub-dmnelf013_ses-dmnelf_task-feedback_run-03_desc-bvaAC1kHz_eeg.edf

Usage:
  python organize_raw_eeg.py --source /path/to/source --target /path/to/target --dry-run
  python organize_raw_eeg.py --source /path/to/source --target /path/to/target --move
"""

import argparse
import re
import shutil
from pathlib import Path
from typing import Dict, Optional, Tuple


def parse_bv_filename(filename: str) -> Optional[Dict]:
    """
    Parse Brain Vision filename to extract subject, task, run.
    
    Examples:
        'sub-dmnelf013_task_feedback-run03_Segment 1.edf' 
        → {'subject': 'sub-dmnelf013', 'task': 'feedback', 'run': '03'}
        
        'dmnelf1001_task_feedback-run02_Segment 1.edf' (old format)
        → {'subject': 'sub-dmnelf1001', 'task': 'feedback', 'run': '02'}
    """
    # Remove " Segment 1" suffix and .edf
    base = filename.replace(' Segment 1.edf', '').replace('.edf', '')
    
    # Try different patterns - new format first, then old format
    patterns = [
        # New format (with sub- prefix)
        (r"(sub-dmnelf\d{3,4})_task_([a-z]+)-run(\d{2})", False),  # sub-dmnelf013_task_feedback-run03
        (r"(sub-dmnelf\d{3,4})_task_([a-z]+)_run(\d{2})", False),   # variant with underscore
        (r"(sub-dmnelf\d{3,4})_task_([a-z]+)", False),              # no run specified
        # Old format (without sub- prefix) - convert to new format
        (r"(dmnelf\d{3,4})_task_([a-z]+)-run(\d{2})", True),        # dmnelf1001_task_feedback-run02
        (r"(dmnelf\d{3,4})_task_([a-z]+)_run(\d{2})", True),        # variant with underscore
        (r"(dmnelf\d{3,4})_task_([a-z]+)", True),                   # no run specified
    ]
    
    for pattern, is_old_format in patterns:
        match = re.search(pattern, base)
        if match:
            groups = match.groups()
            subject = groups[0]
            
            # Convert old format to new format (add sub- prefix)
            if is_old_format:
                subject = f"sub-{subject}"
            
            result = {
                'subject': subject,
                'task': groups[1] if len(groups) > 1 else None,
                'run': groups[2] if len(groups) > 2 else None,
            }
            return result
    
    return None


def get_bids_filename(subject: str, task: str, run: Optional[str] = None, 
                      session: str = "dmnelf") -> str:
    """
    Generate BIDS-compliant filename.
    
    Format: {subject}_ses-{session}_task-{task}_run-{run}_desc-bvaAC1kHz_eeg.edf
    """
    parts = [subject, f"ses-{session}", f"task-{task}"]
    
    if run:
        parts.append(f"run-{run}")
    
    parts.append("desc-bvaAC1kHz_eeg.edf")
    
    return "_".join(parts)


def organize_eeg_files(source_dir: Path, target_dir: Path,
                       dry_run: bool = True, verbose: bool = True,
                       subjects: Optional[list] = None) -> Dict:
    """
    Organize EEG files from source to target BIDS structure.

    Parameters:
        subjects: optional list of subject IDs (e.g. ['sub-dmnelf014']) to
                  restrict processing to. If None, all parseable files are used.

    Returns:
        Dictionary with counts: {'moved': N, 'failed': N, 'skipped': N}
    """

    source_dir = Path(source_dir)
    target_dir = Path(target_dir)

    if not source_dir.exists():
        raise FileNotFoundError(f"Source directory not found: {source_dir}")

    results = {'moved': 0, 'failed': 0, 'skipped': 0, 'files': []}

    # Find all EDF files
    edf_files = list(source_dir.glob("*Segment 1.edf")) + list(source_dir.glob("*.edf"))
    edf_files = list(set(edf_files))  # Remove duplicates

    print(f"\nFound {len(edf_files)} EDF files to organize\n")
    if subjects:
        print(f"Restricting to subjects: {', '.join(subjects)}\n")
    print("=" * 80)

    for source_file in sorted(edf_files):
        filename = source_file.name

        # Parse filename
        parsed = parse_bv_filename(filename)

        if parsed is None:
            print(f"⚠ SKIPPED (couldn't parse): {filename}")
            results['skipped'] += 1
            continue

        subject = parsed['subject']
        task = parsed['task']
        run = parsed['run']

        if subjects and subject not in subjects:
            results['skipped'] += 1
            continue

        if task is None:
            print(f"⚠ SKIPPED (no task): {filename}")
            results['skipped'] += 1
            continue
        
        # Generate BIDS filename
        bids_filename = get_bids_filename(subject, task, run)
        
        # Create target directory structure
        target_subdir = target_dir / subject / "ses-dmnelf" / "eeg"
        target_file = target_subdir / bids_filename
        
        # Show what will happen
        print(f"{'[DRY-RUN]' if dry_run else '[MOVE]'} {subject} task-{task} run-{run or 'N/A'}")
        print(f"  From: {source_file.name}")
        print(f"  To:   {target_file.relative_to(target_dir)}")
        
        if dry_run:
            results['files'].append({
                'source': source_file,
                'target': target_file,
                'subject': subject,
                'task': task,
                'run': run,
            })
            results['moved'] += 1
        else:
            try:
                # Create target directory
                target_subdir.mkdir(parents=True, exist_ok=True)
                
                # Copy file
                shutil.copy2(source_file, target_file)
                print(f"  ✓ Copied")
                results['files'].append({
                    'source': source_file,
                    'target': target_file,
                    'subject': subject,
                    'task': task,
                    'run': run,
                })
                results['moved'] += 1
            except Exception as e:
                print(f"  ✗ ERROR: {e}")
                results['failed'] += 1
    
    print("=" * 80)
    print(f"\nSummary:")
    print(f"  Moved:   {results['moved']}")
    print(f"  Failed:  {results['failed']}")
    print(f"  Skipped: {results['skipped']}")
    print()
    
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Organize raw EEG files from Brain Vision export to BIDS structure"
    )
    parser.add_argument(
        "--source",
        type=str,
        default="/projects/swglab/data/DMNELF/sourcedata/eeg_data/eeg_preprocessed",
        help="Source directory with Brain Vision EDF files"
    )
    parser.add_argument(
        "--target",
        type=str,
        default="/projects/swglab/data/DMNELF/rawdata_eeg",
        help="Target directory for BIDS-organized EEG files"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=True,
        help="Show what would be done without doing it (default: True)"
    )
    parser.add_argument(
        "--move",
        action="store_true",
        help="Actually move/copy files (disables dry-run)"
    )
    parser.add_argument(
        "--subject",
        type=str,
        nargs="+",
        default=None,
        help="Restrict to one or more subject IDs (e.g. sub-dmnelf014 sub-dmnelf015)"
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress verbose output"
    )
    
    args = parser.parse_args()
    
    # If --move is specified, disable dry-run
    dry_run = not args.move
    
    print("\n" + "=" * 80)
    print("EEG File Organization")
    print("=" * 80)
    print(f"Source: {args.source}")
    print(f"Target: {args.target}")
    print(f"Mode:   {'DRY-RUN (preview only)' if dry_run else 'MOVE (will copy files)'}")
    print("=" * 80 + "\n")
    
    try:
        results = organize_eeg_files(
            Path(args.source),
            Path(args.target),
            dry_run=dry_run,
            verbose=not args.quiet,
            subjects=args.subject,
        )
        
        if dry_run:
            print("⚠ This was a DRY-RUN. Use --move to actually copy files.")
            print("\nCommand to execute:")
            print(f"  python organize_raw_eeg.py --source '{args.source}' --target '{args.target}' --move")
        else:
            print("✓ File organization complete!")
            
    except Exception as e:
        print(f"ERROR: {e}")
        exit(1)
