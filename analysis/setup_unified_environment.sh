#!/bin/bash
# setup_unified_environment.sh
# Set up single unified conda environment for both EEG and fMRI preprocessing
# Run this ONCE from the analysis/ folder to set up everything

set -e

CONDA_ENV="dmnelf_preproc"

echo "============================================================"
echo "DMNELF Unified Preprocessing Environment Setup"
echo "============================================================"
echo ""
echo "This creates ONE environment: '$CONDA_ENV'"
echo "Used by both:"
echo "  • mne_eeg_preprocessing/   (EEG → FIF files + QC)"
echo "  • fmri_preprocessing/      (fMRI → Microstates + PDA)"
echo ""

# Check if conda is available
if ! command -v conda &> /dev/null; then
    echo "❌ ERROR: conda not found"
    echo "Please install Anaconda or Miniconda first."
    exit 1
fi
echo "✓ Conda found: $(conda --version)"

# Check if environment already exists
if conda env list | grep -q "^$CONDA_ENV "; then
    echo ""
    echo "⚠️  Environment '$CONDA_ENV' already exists"
    read -p "Recreate it? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "Removing existing environment..."
        conda env remove -n "$CONDA_ENV" -y
        echo "Creating new environment from mne_eeg_preprocessing/environment.yml..."
        conda env create -f mne_eeg_preprocessing/environment.yml -n "$CONDA_ENV"
    else
        echo "Using existing environment. Verifying imports..."
    fi
else
    echo ""
    echo "📦 Creating conda environment '$CONDA_ENV'..."
    conda env create -f mne_eeg_preprocessing/environment.yml -n "$CONDA_ENV"
fi

echo ""
echo "🔧 Activating environment..."
eval "$(conda shell.bash hook)"
conda activate "$CONDA_ENV"
echo "✓ Activated: $(python --version)"

echo ""
echo "📋 Verifying imports for both EEG and fMRI..."
python << 'PYTHON_TEST'
import sys
modules_needed = {
    'mne': 'EEG/fMRI processing',
    'neurokit2': 'EEG signal processing',
    'nilearn': 'fMRI analysis & parcellations',
    'nibabel': 'Neuroimaging file I/O',
    'numpy': 'Numerical computing',
    'scipy': 'Scientific computing',
    'pandas': 'Data analysis',
    'matplotlib': 'Plotting',
    'paramiko': 'SSH/SCP utilities',
    'sklearn': 'Machine learning',
}
failed = []
for mod, desc in modules_needed.items():
    try:
        __import__(mod if mod != 'sklearn' else 'sklearn')
        print(f'  ✓ {mod:15} {desc}')
    except ImportError as e:
        print(f'  ✗ {mod:15} FAILED: {e}')
        failed.append(mod)
if failed:
    print(f'\n❌ Failed imports: {failed}')
    sys.exit(1)
PYTHON_TEST

echo ""
echo "✓ Testing utilities in EEG module..."
(cd mne_eeg_preprocessing && python -c "
import sys
sys.path.insert(0, '.')
from utils import run_ssh, scp_to, scp_from
from config import CLUSTER_SSH, SUBJECTS
print('  ✓ EEG utils and config load OK')
print(f'    Subjects: {SUBJECTS}')
")

echo ""
echo "✓ Testing utilities in fMRI module..."
(cd fmri_preprocessing && python -c "
import sys
sys.path.insert(0, '.')
from utils import run_ssh, scp_to, scp_from
from config import CLUSTER_SSH, SUBJECTS
print('  ✓ fMRI utils and config load OK')
print(f'    Subjects: {SUBJECTS}')
")

echo ""
echo "============================================================"
echo "✅ Setup complete!"
echo "============================================================"
echo ""
echo "Activate environment:"
echo "  conda activate $CONDA_ENV"
echo ""
echo "Deploy EEG preprocessing:"
echo "  cd mne_eeg_preprocessing"
echo "  python deploy_scripts/eeg_preproc_deploy.py --subject sub-dmnelf013"
echo ""
echo "Deploy fMRI preprocessing:"
echo "  cd fmri_preprocessing"
echo "  python deploy_scripts/fmri_preproc_deploy.py --subject sub-dmnelf013"
echo ""
echo "Monitor jobs:"
echo "  ssh cccbauer@explorer.northeastern.edu 'squeue -u cccbauer'"
echo ""
