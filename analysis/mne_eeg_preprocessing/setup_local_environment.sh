#!/bin/bash
# setup_local_environment.sh
# Sets up the local Python environment for EEG preprocessing

set -e  # Exit on error

echo "==============================================="
echo "Setting up EEG preprocessing environment"
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
echo "Step 2: Creating 'eeg_preproc' environment..."
if conda env list | grep -q "eeg_preproc"; then
    echo "Environment 'eeg_preproc' already exists."
    read -p "Do you want to recreate it? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "Removing existing environment..."
        conda env remove -n eeg_preproc -y
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
conda activate eeg_preproc
echo "✓ Environment activated: $(python --version)"

echo ""
echo "Step 4: Verifying imports..."
python -c "
import sys
try:
    import mne
    print('✓ mne OK')
except ImportError as e:
    print(f'✗ mne ERROR: {e}')
    sys.exit(1)

try:
    import neurokit2
    print('✓ neurokit2 OK')
except ImportError as e:
    print(f'✗ neurokit2 ERROR: {e}')
    sys.exit(1)

try:
    import numpy, scipy, pandas, matplotlib
    print('✓ numpy, scipy, pandas, matplotlib OK')
except ImportError as e:
    print(f'✗ ERROR: {e}')
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
"

echo ""
echo "==============================================="
echo "✓ Environment setup complete!"
echo "==============================================="
echo ""
echo "To activate the environment in future sessions, run:"
echo "  conda activate eeg_preproc"
echo ""
echo "To run preprocessing:"
echo "  cd $(pwd)"
echo "  conda activate eeg_preproc"
echo "  python deploy_scripts/eeg_preproc_deploy.py --subject sub-dmnelf012"
echo ""
