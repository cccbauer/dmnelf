#!/usr/bin/env bash
# pull_data.sh — download everything needed to run eeg_bold_coupling from scratch,
# in case the cluster is unavailable. Mirrors the cluster derivatives/ layout locally.
#
# Pulls:
#   1. EEG preprocessed .fif   (ALL tasks, both desc-preproc500Hz variants)
#   2. fMRI features .npz       (cyclic_features, all subjects)                   ~3 M
#   3. fMRIPrep confounds .tsv  (ALL tasks: FD / global_signal / dvars)
#   4. Preprocessed BOLD .nii.gz + brain masks (ALL tasks)                       ~91 G
#      -> only needed to RE-EXTRACT targets (e.g. GSR'd DMN/CEN, new ROIs).
#
# Code + results are already in this git repo, so they are NOT downloaded here.
# The DiFuMo-64 atlas is fetched by nilearn on first use (needs internet, not the cluster).
#
# Resumable: rsync -P skips already-transferred files, so it is safe to re-run.
set -euo pipefail

REMOTE=cccbauer@explorer.northeastern.edu
SRC=/projects/swglab/data/DMNELF/derivatives
DEST=/Users/cccbauer/Documents/GitHub/dmnelf/data/DMNELF/derivatives

mkdir -p "$DEST"

echo ">>> [1/3] EEG preprocessed .fif (ALL tasks, preproc500Hz + ISp01) ..."
rsync -avhP --prune-empty-dirs \
  --include='*/' \
  --include='*desc-preproc500Hz*_eeg.fif' \
  --include='*desc-preproc500Hz*_eeg.json' \
  --exclude='*' \
  "$REMOTE:$SRC/eeg_preprocessed/" "$DEST/eeg_preprocessed/"

echo ">>> [2/3] fMRI features .npz (cyclic_features, all subjects) ..."
rsync -avhP "$REMOTE:$SRC/cyclic_features/" "$DEST/cyclic_features/"

echo ">>> [3/4] fMRIPrep confounds .tsv (ALL tasks) ..."
rsync -avhP --prune-empty-dirs \
  --include='*/' \
  --include='*desc-confounds_timeseries.tsv' \
  --exclude='*' \
  "$REMOTE:$SRC/fmriprep_25.2.5_fmap/" "$DEST/fmriprep_25.2.5_fmap/"

echo ">>> [4/4] Preprocessed BOLD + brain masks (ALL tasks, ~91 G) ..."
rsync -avhP --prune-empty-dirs \
  --include='*/' \
  --include='*desc-preproc_bold.nii.gz' \
  --include='*desc-preproc_bold.json' \
  --include='*desc-brain_mask.nii.gz' \
  --include='*desc-brain_mask.json' \
  --exclude='*' \
  "$REMOTE:$SRC/fmriprep_25.2.5_fmap/" "$DEST/fmriprep_25.2.5_fmap/"

echo
echo ">>> DONE. Data is under: $DEST"
echo ">>> To run the pipeline locally, point config.yaml at the local mirror:"
echo "      eeg_preproc_dir:     $DEST/eeg_preprocessed"
echo "      features_dir_local:  $DEST/cyclic_features"
