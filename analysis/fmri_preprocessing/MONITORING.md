# Monitoring fMRI Preprocessing

Tools to monitor job progress and check for errors.

## Quick Check (from local machine)

```bash
cd /Users/cccbauer/Documents/GitHub/dmnelf/analysis/fmri_preprocessing
conda activate dmnelf_preproc
python check_fmri_status.py
```

Shows:
- Active SLURM jobs
- Latest job logs (last 30 lines)
- Output file counts and sizes
- Error summary

## Detailed Monitoring (on cluster)

```bash
ssh cccbauer@explorer.northeastern.edu << 'EOF'
cd /projects/swglab/data/DMNELF/analysis/fmri_preprocessing
bash monitor_fmri_pipeline.sh
EOF
```

## Real-time Log Viewing

```bash
# Watch latest log
ssh cccbauer@explorer.northeastern.edu "tail -f /projects/swglab/data/DMNELF/analysis/MNE/jupyter/microstate_pda_v3/logs/fmri_pipeline_*.out"

# Or specific job
ssh cccbauer@explorer.northeastern.edu "tail -f /projects/swglab/data/DMNELF/analysis/MNE/jupyter/microstate_pda_v3/logs/fmri_pipeline_6516675.out"
```

## Check Queue

```bash
# Local
ssh cccbauer@explorer.northeastern.edu "squeue -u cccbauer"

# Just fMRI jobs
ssh cccbauer@explorer.northeastern.edu "squeue -u cccbauer | grep fmri"
```

## Output Locations

**Logs:**
```
/projects/swglab/data/DMNELF/analysis/MNE/jupyter/microstate_pda_v3/logs/
├── fmri_pipeline_6516675.out
└── fmri_pipeline_6516676.out
```

**Outputs:**
```
/projects/swglab/data/DMNELF/derivatives/
├── fmri_microstates/sub-dmnelf012/
├── fmri_microstates/sub-dmnelf013/
├── pda_features/sub-dmnelf012/
└── pda_features/sub-dmnelf013/
```

## Typical Output Structure

After completion, expect:
```
sub-dmnelf012/
├── ses-dmnelf/
│   ├── func/
│   │   ├── sub-dmnelf012_ses-dmnelf_task-rest_run-01_desc-difumo64_timeseries.npy
│   │   ├── sub-dmnelf012_ses-dmnelf_task-rest_run-01_desc-microstates_map.npy
│   │   └── ...
│   └── derivatives/
│       ├── microstate_stats.csv
│       ├── pda_distinctiveness.csv
│       └── ...
```

## Common Issues

**Jobs still queued?**
- Check: `squeue -j JOBID`
- Logs may not yet exist

**Job failed?**
- Check stdout: `tail -100 logs/fmri_pipeline_*.out`
- Look for ERROR or Traceback
- Common: Missing input files (check rawdata_eeg organization)

**No output files?**
- Job still running (check queue)
- Check if logs show completion
- Verify subject exists in config.py

## Monitoring Script

Both scripts do the same checks:

| Check | Purpose |
|-------|---------|
| SLURM Queue | See active job status & time |
| Job Logs | Most recent output & errors |
| Output Files | Count & size of generated data |
| Error Summary | Quick error scan |

Run check script every 5-10 minutes to track progress.
