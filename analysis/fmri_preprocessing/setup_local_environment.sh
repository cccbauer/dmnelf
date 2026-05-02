#!/bin/bash
# setup_local_environment.sh
# Set up unified conda environment for EEG + fMRI preprocessing

set -e

CONDA_ENV="dmnelf_preproc"

echo "=========================================="
echo "Setting up DMNELF preprocessing environment"
echo "EEG + fMRI + Microstate + PDA"
echo "=========================================="

# Check if conda is available
if ! command -v conda &> /dev/null; then
    echo "❌ Conda not found. Please install Miniconda or Anaconda."
    exit 1
fi

# Create conda environment from yml
echo "📦 Creating conda environment from environment.yml..."
conda env create -f environment.yml -n $CONDA_ENV || conda env update -f environment.yml -n $CONDA_ENV

# Activate environment
echo "🔧 Activating environment..."
source $(conda info --base)/etc/profile.d/conda.sh
conda activate $CONDA_ENV

# Test imports
echo "✓ Testing imports..."
python -c "import numpy; print('  ✓ numpy')"
python -c "import scipy; print('  ✓ scipy')"
python -c "import pandas; print('  ✓ pandas')"
python -c "import mne; print('  ✓ mne')"
python -c "import neurokit2; print('  ✓ neurokit2')"
python -c "import nilearn; print('  ✓ nilearn')"
python -c "import nibabel; print('  ✓ nibabel')"
python -c "import paramiko; print('  ✓ paramiko')"

# Test utilities
echo "✓ Testing utilities..."
python -c "from utils import run_ssh, scp_to, scp_from; print('  ✓ utils.py loads OK')"
python -c "from config import CLUSTER_SSH, CLUSTER_BASE; print('  ✓ config.py loads OK')"

echo ""
echo "=========================================="
echo "✓ Setup complete!"
echo "=========================================="
echo ""
echo "Usage:"
echo "  conda activate $CONDA_ENV"
echo ""
echo "EEG Preprocessing:"
echo "  cd ../mne_eeg_preprocessing"
echo "  python deploy_scripts/eeg_preproc_deploy.py --subject sub-dmnelf013"
echo ""
echo "fMRI Preprocessing:"
echo "  cd ../fmri_preprocessing"
echo "  python deploy_scripts/fmri_preproc_deploy.py --subject sub-dmnelf013"
echo ""
