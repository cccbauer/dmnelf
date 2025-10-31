#!/bin/bash
# DMNELF Real-time DMN/CEN Neurofeedback Launch Script
# Modified for DMNELF study protocol

# Initialize conda environment
if [ -f ~/anaconda3/etc/profile.d/conda.sh ]; then
    source ~/anaconda3/etc/profile.d/conda.sh
    conda activate base
elif [ -f ~/.bashrc ]; then
    source ~/.bashrc
    if command -v conda &> /dev/null; then
        conda activate base
    fi
fi

# Set up logging
LOG_DIR="../logs"
mkdir -p "$LOG_DIR"
MAIN_LOG="$LOG_DIR/dmnelf_session_$(date +%F_%H-%M-%S).log"

# Function to log with timestamp
log_message() {
    echo "[$(date +%F_%T)] $1" | tee -a "$MAIN_LOG"
}

# Function to check system requirements
check_requirements() {
    log_message "Checking system requirements..."
    
    # Check for required software
    local missing_deps=()
    
    command -v fsl >/dev/null 2>&1 || missing_deps+=("FSL")
    command -v singularity >/dev/null 2>&1 || missing_deps+=("Singularity")
    python -c "import nipype" 2>/dev/null || missing_deps+=("nipype")
    python -c "import nilearn" 2>/dev/null || missing_deps+=("nilearn")
    
    if [ ${#missing_deps[@]} -ne 0 ]; then
        zenity --error --text="Missing dependencies: ${missing_deps[*]}\nPlease install before continuing."
        exit 1
    fi
    
    # Check CPU cores for MELODIC optimization
    local cpu_cores=$(nproc)
    export OMP_NUM_THREADS=$cpu_cores
    export FSL_CPU_CORES=$cpu_cores
    log_message "System configured for $cpu_cores CPU cores"
}

# Main GUI function for DMNELF
main_gui() {
    # Step 1: Always ask for subject ID first
    if [[ -z $MURFI_SUBJECT_NAME ]] || [[ "$1" == "--new-subject" ]]; then
        # Get subject number
        subject_number=$(zenity --entry --title="DMNELF Real-time DMN/CEN Neurofeedback" \
            --text="DMNELF Study - DMN/CEN Neurofeedback Training\n\nEnter subject number (e.g., 001, 999, 123...):\n\nWill create: dmnelf[NUMBER]" \
            --width=450 \
            --cancel-label "Exit")
        
        ret=$?
        if [[ $ret == 1 ]] || [[ -z "$subject_number" ]]; then
            log_message "DMNELF session cancelled by user"
            exit 0
        fi
        
        # Validate subject number
        if [[ "$subject_number" =~ ^[0-9]+$ ]]; then
            participant_id="dmnelf${subject_number}"
            log_message "DMNELF subject ID: $participant_id"
        else
            zenity --error --text="Invalid subject number. Please enter numbers only (e.g., 1, 999, 123)"
            exit 1
        fi
        
        # Step 2: Check if subject directory exists
        subject_dir="../subjects/$participant_id"
        
        if [[ -d "$subject_dir" ]]; then
            # Subject exists - ask overwrite or proceed
            if zenity --question --text="<span foreground='red'><b>Subject $participant_id already exists!</b></span>\n\nDo you want to overwrite the existing directory?" --width=450; then
                # Overwrite - delete and recreate with template files
                log_message "Overwriting existing subject directory for $participant_id"
                rm -rf "$subject_dir"
                
                # Create fresh directory structure for DMNELF
                mkdir -p "$subject_dir"/{xml,mask/{mni,lps},xfm,rest,img}
                
                log_message "Fresh directory created for $participant_id"
                zenity --info --text="Directory recreated successfully!\n\nFresh structure ready for $participant_id" --width=400
                
                # Create command log
                echo "# DMNELF MURFI Command Log for $participant_id" > "$subject_dir/murfi_command_log.txt"
            else
                # Proceed with existing directory
                log_message "Using existing directory for $participant_id"
                zenity --info --text="Using existing directory for $participant_id" --width=400
            fi
        else
            # New subject - create directory structure
            log_message "Creating new subject directory for $participant_id"
            
            mkdir -p "$subject_dir"/{xml,mask/{mni,lps},xfm,rest,img}
            
            log_message "New directory created for $participant_id"
            zenity --info --text="New subject directory created!\n\nStructure ready for $participant_id" --width=400
            
            # Create command log
            echo "# DMNELF MURFI Command Log for $participant_id" > "$subject_dir/murfi_command_log.txt"
        fi
        
        # Step 3: DMNELF action menu
        export MURFI_SUBJECT_NAME="$participant_id"
        
        action=$(zenity --list --title="DMNELF Subject: $participant_id Ready" \
            --text="Subject $participant_id is ready.\nChoose next action:" \
            --column="Action" --column="Description" \
            "create" "Create XML files for subject" \
            "setup" "Setup system to start neurofeedback" \
            --width=900 --height=300 \
            --cancel-label="Exit")
        
        if [[ -z "$action" ]]; then
            log_message "No action selected, exiting"
            exit 0
        fi
        
        # Execute selected action
        execute_dmnelf_step "$participant_id" "$action" "accelerated"
        
    else
        # Subject already loaded - show DMNELF step selection
        step=$(zenity --list --title="DMNELF Real-time DMN/CEN Neurofeedback" \
            --text="DMNELF PARTICIPANT: ${MURFI_SUBJECT_NAME}\nDMN/CEN Training Protocol\n\nSystem: $(nproc) CPU cores | Accelerated Processing\n\nSelect processing step:" \
            --column="Step" --column="Description" \
            "create" "Create XML files for subject" \
            "setup" "Setup system to start neurofeedback" \
            "2vol" "Receive 2-volume scan for registration" \
            "short_rest" "Short resting state scan" \
            "extract_rs_networks" "Extract resting state networks (MELODIC)" \
            "process_roi_masks_native" "Process ROI masks in native space" \
            "register_native" "Register native space masks to 2vol" \
            "experience_sampling" "Experience sampling protocol" \
            "feedback" "Real-time neurofeedback session" \
            "resting_state" "Full resting state scan" \
            "cleanup_backup" "Clean up and backup data" \
            --width=900 --height=600 \
            --cancel-label "Exit")
        
        ret=$?
        if [[ $ret == 1 ]]; then
            log_message "DMNELF session ended by user for ${MURFI_SUBJECT_NAME}"
            exit 0
        fi
        
        # Execute selected step
        execute_dmnelf_step "$MURFI_SUBJECT_NAME" "$step" "accelerated"
    fi
}

# DMNELF-specific step execution
execute_dmnelf_step() {
    local participant_id=$1
    local step=$2
    local processing_mode=$3
    
    local subj_dir="../subjects/$participant_id"
    local cmd_log="$subj_dir/murfi_command_log.txt"
    
    log_message "Executing DMNELF step: $step for $participant_id in $processing_mode mode"
    
    case "$step" in
        'create')
            log_message "Creating XML files and copying templates for $participant_id"
            echo "[$(date +%F_%T)] DMNELF createxml.sh $participant_id setup" >> "$cmd_log"
            
            local template_dir="../subjects/template"
            local xml_source="$template_dir/xml/xml_orig"
            local mask_source="$template_dir/mask"
            local xml_dest="$subj_dir/xml"
            local mask_dest="$subj_dir/mask"
            
            # First run createxml.sh to create directory structure
            if [[ -f "createxml.sh" ]]; then
                chmod +x createxml.sh
                log_message "Running createxml.sh for directory creation"
                source createxml.sh "$participant_id" setup 2>&1 | tee -a "$MAIN_LOG"
            else
                log_message "createxml.sh not found, creating directories manually"
                # Create directory structure manually if createxml.sh is missing
                mkdir -p "$subj_dir"/{xml,mask/{mni,lps},xfm,rest,img,log,fsfs}
            fi
            
            # Verify and copy XML files
            if [[ -d "$xml_source" ]]; then
                xml_files=$(find "$xml_source" -name "*.xml" -type f)
                if [[ -n "$xml_files" ]]; then
                    log_message "Copying XML files from $xml_source to $xml_dest"
                    cp "$xml_source"/*.xml "$xml_dest/" 2>&1 | tee -a "$MAIN_LOG"
                    xml_copied=$(find "$xml_dest" -name "*.xml" | wc -l)
                    log_message "Copied $xml_copied XML files"
                else
                    log_message "WARNING: No XML files found in $xml_source"
                fi
            else
                log_message "ERROR: XML source directory not found: $xml_source"
            fi
            
            # Verify and copy mask files
            if [[ -d "$mask_source" ]]; then
                log_message "Copying mask files from $mask_source to $mask_dest"
                # Use rsync for better copying with subdirectories
                rsync -av "$mask_source"/ "$mask_dest"/ 2>&1 | tee -a "$MAIN_LOG"
                mask_files=$(find "$mask_dest" -type f | wc -l)
                log_message "Copied mask directory structure with $mask_files files"
            else
                log_message "ERROR: Mask source directory not found: $mask_source"
            fi
            
            # Final verification
            xml_final_count=$(find "$xml_dest" -name "*.xml" 2>/dev/null | wc -l)
            mask_final_count=$(find "$mask_dest" -type f 2>/dev/null | wc -l)
            
            log_message "Final verification - XML files: $xml_final_count, Mask files: $mask_final_count"
            
            if [[ $xml_final_count -gt 0 ]]; then
                log_message "Template copying completed successfully for $participant_id"
                
                # Show detailed success message
                zenity --info --text="Subject $participant_id created successfully!\n\nCopied files:\n• XML templates: $xml_final_count files\n• Mask files: $mask_final_count files\n\nReady for DMNELF protocol." --width=450
                
                # List the copied XML files in the log
                log_message "XML files copied:"
                find "$xml_dest" -name "*.xml" -exec basename {} \; | while read xmlfile; do
                    log_message "  - $xmlfile"
                done
                
            else
                zenity --error --text="Template copying failed!\n\nNo XML files were copied to:\n$xml_dest\n\nCheck the log file:\n$MAIN_LOG" --width=500
                log_message "ERROR: No XML files copied for $participant_id"
                return 1
            fi
            ;;
        'setup')
            log_message "Running system setup for DMNELF neurofeedback"
            echo "[$(date +%F_%T)] DMNELF feedback.sh $participant_id setup" >> "$cmd_log"
            
            zenity --info --text="Running system setup in terminal..." --width=400
            bash feedback.sh "$participant_id" setup
            
            setup_result=$?
            if [[ $setup_result -eq 0 ]]; then
                log_message "Setup completed successfully for $participant_id"
            else
                log_message "Setup completed with issues for $participant_id"
            fi
            ;;
        'extract_rs_networks')
            log_message "Running MELODIC with $FSL_CPU_CORES cores for accelerated DMNELF processing"
            export MELODIC_NPROC=$FSL_CPU_CORES
            echo "[$(date +%F_%T)] DMNELF feedback.sh $participant_id $step accelerated" >> "$cmd_log"
            source feedback.sh "$participant_id" "$step"
            ;;
        'cleanup_backup')
            log_message "Running cleanup and backup for $participant_id"
            
            # Confirm cleanup operation
            if zenity --question --title="Delete files?" \
                --text="Are you sure you want to clean up the directory and delete files for ${participant_id}?\n\nThe entire img folder will be deleted, as well as raw bold data from the rest directory" \
                --cancel-label "Exit" --ok-label "Delete files" \
                --width=500; then
                
                log_message "User confirmed cleanup for $participant_id"
                
                # Delete img folder and large bold files from the rest folder
                local subj_dir_full="../subjects/$participant_id"
                rm -rf "$subj_dir_full/img"
                rm -f "$subj_dir_full/rest/"*bold.nii.gz
                rm -f "$subj_dir_full/rest/"*bold_mcflirt.nii.gz
                rm -f "$subj_dir_full/rest/"*bold_mcflirt_masked.nii.gz
                
                log_message "Cleanup completed for $participant_id"
                
                # After cleanup, prompt for username for rsync backup
                username=$(zenity --entry --title="Rsync Authentication" \
                    --text="Enter your username for data transfer to server:\n\nPassword will be requested in terminal" \
                    --cancel-label "Skip Transfer" \
                    --width=400)
                
                # If user cancels, skip the rsync transfer
                if [[ -z "$username" ]]; then
                    log_message "Data transfer cancelled by user"
                    zenity --info --text="Cleanup completed.\nData transfer skipped." --width=300
                    return
                fi
                
                # Perform rsync with authentication to DMNELF directory
                log_message "Starting data transfer for $participant_id to DMNELF directory"
                zenity --info --text="Starting data transfer...\n\nYou will be prompted for password in terminal.\nCheck terminal window." --width=400
                
                # Use standard rsync for DMNELF path
                rsync -avrz --chmod=g+rwx --perms "$subj_dir_full" "${username}@xfer.discovery.neu.edu:/work/swglab/data/DMNELF/sourcedata/murfi/"
                rsync_ret=$?
                
                # Check rsync result
                if [[ $rsync_ret == 0 ]]; then
                    log_message "Data transfer completed successfully for $participant_id"
                    zenity --info --text="Data transfer completed successfully!\n\nData backed up to DMNELF server directory." --width=400
                else
                    log_message "Data transfer failed for $participant_id"
                    zenity --error --text="Data transfer failed.\n\nPlease check your credentials and network connection." --width=400
                fi
            else
                log_message "Cleanup cancelled by user"
                zenity --info --text="Cleanup operation cancelled." --width=300
            fi
            ;;
        *)
            echo "[$(date +%F_%T)] DMNELF feedback.sh $participant_id $step accelerated" >> "$cmd_log"
            source feedback.sh "$participant_id" "$step"
            ;;
    esac
    
    # Check execution status
    if [[ $? -eq 0 ]]; then
        log_message "DMNELF step '$step' completed successfully for $participant_id"
        
        # Auto-advance logic for DMNELF workflow
        case "$step" in
            'setup')
                log_message "DMNELF setup step completed for $participant_id"
                
                # Ask user about network connectivity with yellow warning
                if zenity --question --icon=warning \
                    --text="<span foreground='#FF8C00'><b>Network Connectivity Check</b></span>\n\nDid the ping tests to the scanner and stimulation computer work correctly?\n\n• Scanner (192.168.2.1)\n• Stim Computer (192.168.2.6)\n\nSelect your choice:" \
                    --ok-label="Pings Worked - Continue" \
                    --cancel-label="Retry Setup" \
                    --width=500; then
                    
                    log_message "User confirmed network connectivity is working"
                    zenity --info --text="Setup verified and complete for $participant_id!\n\nNetwork connectivity confirmed.\nReady for DMNELF workflow." --width=400
                else
                    log_message "User reported network issues, retrying setup"
                    zenity --info --text="Retrying setup for network connectivity...\n\nRunning setup again." --width=400
                    execute_dmnelf_step "$participant_id" "setup" "accelerated"
                    return
                fi
                ;;
            '2vol')
                log_message "2-volume scan completed for $participant_id"
                zenity --info --text="2-volume scan complete!\n\nReady for next step." --width=400
                ;;
            'short_rest')
                log_message "Short rest completed for $participant_id"
                zenity --info --text="Short rest scan complete!\n\nReady for network extraction." --width=400
                ;;
            'extract_rs_networks')
                log_message "Network extraction completed for $participant_id"
                zenity --info --text="Resting state networks extracted!\n\nReady for mask processing." --width=400
                ;;
        esac
    else
        log_message "ERROR: DMNELF step '$step' failed for $participant_id"
        zenity --error --text="DMNELF step '$step' failed for $participant_id.\nCheck logs for details."
    fi
}

# Main execution for DMNELF
main() {
    log_message "Starting DMNELF Real-time DMN/CEN Neurofeedback System"
    
    # Clear any existing subject name to force new input
    unset MURFI_SUBJECT_NAME
    
    # System checks
    check_requirements
    
    # Set DMNELF environment
    export MURFI_SUBJECTS_DIR="$(dirname $(pwd))/subjects/"
    
    # Display DMNELF study info first
    zenity --info --text="DMNELF Study\nReal-time DMN/CEN Neurofeedback\n\nSystem ready with $(nproc) CPU cores\nAccelerated processing mode" --width=400
    
    # Run main GUI (will always prompt for subject ID now)
    main_gui --new-subject
    
    # After completing a step, go directly back to step selection
    while true; do
        if [[ -n "$MURFI_SUBJECT_NAME" ]]; then
            main_gui
        else
            # If subject name is cleared, restart from beginning
            main_gui --new-subject
        fi
    done
    
    log_message "DMNELF MURFI session ended"
}

# Execute main function
main "$@"
