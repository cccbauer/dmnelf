#!/bin/bash
# Download one feedback run (ses-nf run-01) per rtBPD subject from the cluster.
# Idempotent (rsync -a skips existing). NOTE: run under bash, not zsh (zsh needs ${=SUBS} to split).
# Cluster: explorer.northeastern.edu (key auth).  Local mirror matches the DMNELF layout so the
# cross_project_test.py finder works (sub-<id>/ses-nf/eeg/..._desc-preproc500Hz_eeg.fif).
set -e
SUBS="rtbpd002 rtbpd003 rtbpd004 rtbpd009 rtbpd010 rtbpd011 rtbpd012 rtbpd013 rtbpd015 rtbpd018 rtbpd020 rtbpd021 rtbpd022 rtbpd024 rtbpd026 rtbpd027 rtbpd028 rtbpd030 rtbpd034 rtbpd038 rtbpd040"
REMOTE=/projects/swglab/data/rtBPD/derivatives/eeg_preprocessed
LOCAL=/Users/cccbauer/Documents/GitHub/dmnelf/data/rtBPD/derivatives/eeg_preprocessed
L=/tmp/rtbpd_files.txt; : > "$L"
for s in $SUBS; do
  echo "sub-$s/ses-nf/eeg/sub-${s}_ses-nf_task-feedback_run-01_desc-preproc500Hz_eeg.fif" >> "$L"
done
mkdir -p "$LOCAL"
rsync -a --files-from="$L" -e "ssh -o BatchMode=yes" \
  "cccbauer@explorer.northeastern.edu:$REMOTE/" "$LOCAL/"
echo "rtBPD run-01 files: $(find "$LOCAL" -name '*run-01*.fif' | wc -l)/21"
