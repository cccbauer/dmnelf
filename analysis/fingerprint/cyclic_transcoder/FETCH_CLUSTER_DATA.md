# Fetch Full Ground Truth Data from Cluster

The downloaded `cyclic_features_local/` only contains 50-parcel fMRI. To get full 528-timepoint visualization, we need the complete predictions with 66-feature fMRI (64 DiFuMo + 2 personal ROIs).

## Step 1: Determine Your Cluster Host

If you don't know it, check your SSH config:
```bash
cat ~/.ssh/config | grep -A 5 explorer
# or
cat ~/.ssh/config | grep -A 5 northeastern
```

Or ask: `ssh-keyscan <hostname>` (common options: `explorer.ccv.brown.edu`, `hpc.northeastern.edu`, `explorer.ccs.neu.edu`)

## Step 2: Run SCP Commands

Replace `YOUR_USERNAME` and `CLUSTER_HOST` in the commands below, then copy and paste into terminal.

### Full Dataset (All Subjects - ~1 GB)
```bash
# Download all prediction files with full 66-feature data
scp -r cccbauer@explorer.northeastern.edu:/projects/swglab/data/DMNELF/derivatives/cyclic_features/sub-*/predictions/ \
    /Users/cccbauer/Documents/GitHub/dmnelf/analysis/fingerprint/cyclic_transcoder/cyclic_features_full/

# Verify download
ls -lh /Users/cccbauer/Documents/GitHub/dmnelf/analysis/fingerprint/cyclic_transcoder/cyclic_features_full/*/predictions/*.npz | wc -l
# Should show 14 files (one per subject)
```

### Or Download Individual Subjects

**Best subject (dmnelf005):**
```bash
scp YOUR_USERNAME@CLUSTER_HOST:/projects/swglab/data/DMNELF/derivatives/cyclic_features/sub-dmnelf005/predictions/*.npz \
    /Users/cccbauer/Documents/GitHub/dmnelf/analysis/fingerprint/cyclic_transcoder/cyclic_features_full/sub-dmnelf005/predictions/
```

**Worst subject (dmnelf010 for comparison):**
```bash
scp YOUR_USERNAME@CLUSTER_HOST:/projects/swglab/data/DMNELF/derivatives/cyclic_features/sub-dmnelf010/predictions/*.npz \
    /Users/cccbauer/Documents/GitHub/dmnelf/analysis/fingerprint/cyclic_transcoder/cyclic_features_full/sub-dmnelf010/predictions/
```

**All 14 subjects:**
```bash
for subj in dmnelf001 dmnelf004 dmnelf005 dmnelf006 dmnelf007 dmnelf008 dmnelf009 dmnelf010 dmnelf011 dmnelf1001 dmnelf1002 dmnelf1003; do
  echo "Downloading $subj..."
  scp YOUR_USERNAME@CLUSTER_HOST:/projects/swglab/data/DMNELF/derivatives/cyclic_features/sub-$subj/predictions/*.npz \
      /Users/cccbauer/Documents/GitHub/dmnelf/analysis/fingerprint/cyclic_transcoder/cyclic_features_full/sub-$subj/predictions/
done
```

## Step 3: Verify Full Data

After download, check that you have 66-feature fMRI (not 50-parcel):
```bash
python << 'EOF'
import numpy as np

npz_path = "/Users/cccbauer/Documents/GitHub/dmnelf/analysis/fingerprint/cyclic_transcoder/cyclic_features_full/sub-dmnelf005/predictions/sub-dmnelf005_task-feedback_pda_prediction.npz"
data = np.load(npz_path, allow_pickle=True)

print("✓ Full data structure:")
for key in sorted(data.files):
    val = data[key]
    shape = val.shape if hasattr(val, 'shape') else 'scalar'
    print(f"  {key:25s}: {str(shape):20s}")

fmri_true = data['fmri_true']
print(f"\n✓ fmri_true shape: {fmri_true.shape}")
if fmri_true.shape[0] == 66:
    print("  SUCCESS: Full 66-feature data (64 DiFuMo + 2 personal ROIs)")
elif fmri_true.shape[0] == 50:
    print("  WARNING: Only 50-parcel data (missing personal ROIs)")
else:
    print(f"  UNKNOWN: Unexpected {fmri_true.shape[0]} parcels")

print(f"\n✓ pda_predicted shape: {data['pda_predicted'].shape}")
print(f"✓ dmn_idx: {int(data['dmn_idx'])}, cen_idx: {int(data['cen_idx'])}")
EOF
```

## Step 4: Generate Full Visualization

Once data is downloaded, update the script to use the full data:

```bash
python plot_all_feedback_runs_full.py --subject dmnelf005 --prediction-dir cyclic_features_full --save
```

This will generate:
- 3-panel plot with all 528 timepoints
- Correlation on full feedback run (10 minutes)
- CSV with complete timeseries data
