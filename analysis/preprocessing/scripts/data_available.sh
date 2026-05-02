#!/bin/bash
# data_available.sh
# Check raw + fMRIPrep data availability for DMNELF or rtBPD.
#
# Usage:
#   bash data_available.sh dmnelf
#   bash data_available.sh rtbpd
#
# Deploy:
#   scp scripts/data_available.sh cccbauer@explorer.northeastern.edu:/projects/swglab/data/DMNELF/analysis/MNE/jupyter/microstate_pda_v3/scripts/
#   ssh cccbauer@explorer.northeastern.edu "bash /projects/swglab/data/DMNELF/analysis/MNE/jupyter/microstate_pda_v3/scripts/data_available.sh dmnelf"
#   ssh cccbauer@explorer.northeastern.edu "bash /projects/swglab/data/DMNELF/analysis/MNE/jupyter/microstate_pda_v3/scripts/data_available.sh rtbpd"

STUDY="${1:-dmnelf}"

# ── Study-specific config ──────────────────────────────────
if [ "$STUDY" = "dmnelf" ]; then
    RAWDATA="/projects/swglab/data/DMNELF/rawdata"
    FMRIPREP="/projects/swglab/data/DMNELF/derivatives/fmriprep_25.2.5_fmap"
    PREFIX="sub-dmnelf"
    SUBJECTS=$(ls $RAWDATA | grep "^sub-dmnelf" | sort)
    SESSIONS="ses-dmnelf"
    declare -A TASK_RUNS
    TASK_RUNS["rest"]="01 02"
    TASK_RUNS["shortrest"]="01"
    TASK_RUNS["feedback"]="01 02 03 04"

elif [ "$STUDY" = "rtbpd" ]; then
    RAWDATA="/projects/swglab/data/rtBPD/rawdata"
    FMRIPREP="/projects/swglab/data/rtBPD/derivatives/fmriprep_24.1.1"
    PREFIX="sub-rtbpd"
    SUBJECTS=$(ls $RAWDATA | grep "^sub-rtbpd" | sort)
    SESSIONS="ses-loc ses-nf1 ses-nf2"
    declare -A TASK_RUNS
    # loc session
    TASK_RUNS["ses-loc:rest"]="01 02"
    TASK_RUNS["ses-loc:selfref"]="01 02"
    # nf1 and nf2 sessions
    TASK_RUNS["ses-nf1:rest"]="01 02 03 04"
    TASK_RUNS["ses-nf1:feedback"]="01 02 03 04 05"
    TASK_RUNS["ses-nf2:rest"]="01 02 03 04"
    TASK_RUNS["ses-nf2:feedback"]="01 02 03 04 05"
else
    echo "Unknown study: $STUDY"
    echo "Usage: $0 dmnelf|rtbpd"
    exit 1
fi

# ── Colors ─────────────────────────────────────────────────
GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m'

echo "======================================================="
echo " Data Availability Check: $STUDY"
echo " $(date)"
echo "======================================================="
echo ""

printf "%-20s %-10s %-14s %-6s %-6s\n" "Subject" "Session" "Task/Run" "RAW" "FMRI"
echo "-------------------------------------------------------"

COMPLETE_SUBJECTS=()
INCOMPLETE_SUBJECTS=()

for subject in $SUBJECTS; do
    subject_complete=true

    if [ "$STUDY" = "dmnelf" ]; then
        for task in rest shortrest feedback; do
            runs="${TASK_RUNS[$task]}"
            for run in $runs; do
                raw_f="$RAWDATA/$subject/ses-dmnelf/func/${subject}_ses-dmnelf_task-${task}_run-${run}_bold.nii.gz"
                fmri_f="$FMRIPREP/$subject/ses-dmnelf/func/${subject}_ses-dmnelf_task-${task}_run-${run}_space-MNI152NLin6Asym_res-2_desc-preproc_bold.nii.gz"

                raw_ok="NO"; fmri_ok="NO"
                [ -f "$raw_f"  ] && raw_ok="YES"
                [ -f "$fmri_f" ] && fmri_ok="YES"
                [ "$raw_ok" = "NO" ] || [ "$fmri_ok" = "NO" ] && subject_complete=false

                raw_col="${RED}NO ${NC}"; fmri_col="${RED}NO ${NC}"
                [ "$raw_ok"  = "YES" ] && raw_col="${GREEN}YES${NC}"
                [ "$fmri_ok" = "YES" ] && fmri_col="${GREEN}YES${NC}"

                printf "%-20s %-10s %-14s " "$subject" "ses-dmnelf" "${task}/run-${run}"
                echo -e "${raw_col}  ${fmri_col}"
            done
        done

    elif [ "$STUDY" = "rtbpd" ]; then
        for session in ses-loc ses-nf1 ses-nf2; do
            # Skip session if rawdata dir doesn't exist
            if [ ! -d "$RAWDATA/$subject/$session/func" ]; then
                printf "%-20s %-10s %-14s " "$subject" "$session" "NO_SESSION"
                echo -e "${RED}MISS${NC}  ${RED}MISS${NC}"
                subject_complete=false
                continue
            fi

            if [ "$session" = "ses-loc" ]; then
                tasks="rest selfref"
            else
                tasks="rest feedback"
            fi

            for task in $tasks; do
                key="${session}:${task}"
                runs="${TASK_RUNS[$key]}"
                for run in $runs; do
                    raw_f="$RAWDATA/$subject/$session/func/${subject}_${session}_task-${task}_run-${run}_bold.nii.gz"
                    fmri_f="$FMRIPREP/$subject/$session/func/${subject}_${session}_task-${task}_run-${run}_space-MNI152NLin6Asym_res-2_desc-preproc_bold.nii.gz"

                    raw_ok="NO"; fmri_ok="NO"
                    [ -f "$raw_f"  ] && raw_ok="YES"
                    [ -f "$fmri_f" ] && fmri_ok="YES"
                    [ "$raw_ok" = "NO" ] || [ "$fmri_ok" = "NO" ] && subject_complete=false

                    raw_col="${RED}NO ${NC}"; fmri_col="${RED}NO ${NC}"
                    [ "$raw_ok"  = "YES" ] && raw_col="${GREEN}YES${NC}"
                    [ "$fmri_ok" = "YES" ] && fmri_col="${GREEN}YES${NC}"

                    printf "%-20s %-10s %-14s " "$subject" "$session" "${task}/run-${run}"
                    echo -e "${raw_col}  ${fmri_col}"
                done
            done
        done
    fi

    if [ "$subject_complete" = true ]; then
        COMPLETE_SUBJECTS+=("$subject")
    else
        INCOMPLETE_SUBJECTS+=("$subject")
    fi
    echo ""
done

# ── Summary ────────────────────────────────────────────────
echo "======================================================="
echo " Summary: $STUDY"
echo "======================================================="
echo ""
echo -e "${GREEN}Complete subjects:${NC}"
for s in "${COMPLETE_SUBJECTS[@]}"; do echo "  $s"; done

echo ""
echo -e "${RED}Incomplete subjects:${NC}"
for s in "${INCOMPLETE_SUBJECTS[@]}"; do echo "  $s"; done

echo ""
echo "Total:      $(echo $SUBJECTS | wc -w)"
echo "Complete:   ${#COMPLETE_SUBJECTS[@]}"
echo "Incomplete: ${#INCOMPLETE_SUBJECTS[@]}"

echo ""
echo "======================================================="
echo " Python list of complete subjects:"
echo "======================================================="
echo "SUBJECTS = ["
for s in "${COMPLETE_SUBJECTS[@]}"; do echo "    \"${s}\","; done
echo "]"