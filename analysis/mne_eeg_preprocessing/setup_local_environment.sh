#!/bin/bash
# setup_local_environment.sh
# Sets up the unified Python environment for EEG + fMRI preprocessing

set -e  # Exit on error

CONDA_ENV="dmnelf_preproc"

echo "==============================================="
echo "Setting up DMNELF preprocessing environment"
echo "EEG + fMRI + Microstate + PDA"
echo "==============================================="

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

echo ""
echo "Step 1: Checking conda..."
if ! command -v conda &> /dev/null; then
    echo "ERROR: conda not found. Please install Anaconda or Miniconda first."
    exit 1
fi
echo "✓ conda found: $(conda --version)"

echo ""
echo "Step 2: Creating '$CONDA_ENV' environment..."
if conda env list | grep -q "$CONDA_ENV"; then
    echo "Environment '$CONDA_ENV' already exists."
    read -p "Do you want to recreate it? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "Removing existing environment..."
        conda env remove -n "$CONDA_ENV" -y
        echo "Creating new environment..."
        conda env create -f environment.yml
    else
        echo "Using existing environment"
    fi
else
    echo "Creating new environment..."
    conda env create -f environment.yml
fi

echo ""
echo "Step 3: Activating environment..."
eval "$(conda shell.bash hook)"
conda activate "$CONDA_ENV"
echo "✓ Environment activated: $(python --version)"

echo ""
echo "Step 4: Verifying imports..."
python -c "
import sys
imports = {
    'mne': 'EEG/fMRI preprocessing',
    'neurokit2': 'EEG signal processing',
    'nilearn': 'fMRI analysis',
    'nibabel': 'Neuroimaging file I/O',
    'numpy': 'Numerical computing',
    'scipy': 'Scientific computing',
    'pandas': 'Data analysis',
    'matplotlib': 'Plotting',
    'paramiko': 'SSH/SCP utilities',
}
failed = []
for module, desc in imports.items():
    try:
        __import__(module)
        print(f'✓ {module:15} ({desc})')
    except ImportError as e:
        print(f'✗ {module:15} ERROR: {e}')
        failed.append(module)
if failed:
    print(f'\n❌ Failed imports: {failed}')
    sys.exit(1)
"

echo ""
echo "Step 5: Testing deployment utilities..."
python -c "
import sys
sys.path.insert(0, '.')
from utils import run_ssh, scp_to
print('✓ Deployment utilities loaded OK')
"

echo ""
echo "Step 6: Testing config..."
python -c "
import sys
sys.path.insert(0, '.')
import config
print(f'✓ Config loaded OK')
print(f'  LOCAL_BASE: {config.LOCAL_BASE}')
print(f'  CLUSTER_SSH: {config.CLUSTER_SSH}')
print(f'  EEG_ROOT: {config.EEG_ROOT}')
print(f'  SUBJECTS: {config.SUBJECTS}')
"

echo ""
echo "==============================================="
echo "✓ Environment setup complete!"
echo "==============================================="
echo ""
echo "To activate in future sessions:"
echo "  conda activate $CONDA_ENV"
echo ""
echo "EEG Preprocessing:"
echo "  cd mne_eeg_preprocessing"
echo "  python deploy_scripts/eeg_preproc_deploy.py --subject sub-dmnelf013"
echo ""
echo "fMRI Preprocessing:"
echo "  cd fmri_preprocessing"
echo "  python deploy_scripts/fmri_preproc_deploy.py --subject sub-dmnelf013"
echo ""
echo "  conda activate eeg_preproc"
echo ""
echo "To run preprocessing:"
echo "  cd $(pwd)"
echo "  conda activate eeg_preproc"
echo "  python deploy_scripts/eeg_preproc_deploy.py --subject sub-dmnelf012"
echo ""
