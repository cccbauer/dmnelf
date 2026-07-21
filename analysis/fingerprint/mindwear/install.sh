#!/usr/bin/env bash
# install.sh — one-command setup for the MindWear neurofeedback console.
#
#   ./install.sh                 # create the `mindwear` conda env, verify, print next steps
#   ./install.sh --name myenv    # use a different env name
#   ./install.sh --force         # remove an existing env of that name first
#
# Requires conda (Miniconda/Anaconda/Miniforge) on PATH. After it finishes:
#   conda activate mindwear
#   python launch_gui.py
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_NAME="mindwear"
FORCE=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --name) ENV_NAME="$2"; shift 2 ;;
    --force) FORCE=1; shift ;;
    -h|--help) grep '^#' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done

echo "== MindWear installer =="

if ! command -v conda >/dev/null 2>&1; then
  echo "ERROR: conda not found on PATH. Install Miniconda/Miniforge first:" >&2
  echo "  https://conda-forge.org/download/" >&2
  exit 1
fi
# make `conda activate` usable inside this non-interactive shell
source "$(conda info --base)/etc/profile.d/conda.sh"

if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  if [[ "$FORCE" == "1" ]]; then
    echo "-- removing existing env '$ENV_NAME' (--force)"
    conda env remove --name "$ENV_NAME" -y
  else
    echo "Env '$ENV_NAME' already exists. Re-run with --force to recreate, or:"
    echo "  conda env update --name $ENV_NAME -f \"$HERE/environment.yml\" --prune"
    exit 1
  fi
fi

echo "-- creating env '$ENV_NAME' from environment.yml (this pulls a large scientific stack; give it a few minutes)"
# environment.yml pins name: mindwear; -n lets the caller override it.
conda env create -n "$ENV_NAME" -f "$HERE/environment.yml"

echo "-- verifying the install"
conda run -n "$ENV_NAME" python - <<'PY'
import importlib
mods = ["numpy", "scipy", "sklearn", "mne", "yaml", "flet", "flet_charts", "psychopy", "pylsl", "websocket"]
missing = []
for m in mods:
    try:
        importlib.import_module(m)
    except Exception as e:
        missing.append(f"{m}: {e}")
if missing:
    print("VERIFY FAILED — missing/broken imports:")
    for x in missing:
        print("  -", x)
    raise SystemExit(1)
print("verify OK — all runtime imports resolve")
PY

cat <<EOF

== Done. ==
To launch MindWear:
    conda activate $ENV_NAME
    cd "$HERE"
    python launch_gui.py

First launch opens an empty Study Manager — click "New Study" to configure one:
  * Source  -> LSL for a live headset (enable the EmotivPRO LSL outlet first), or
              Replay to stream a recorded .fif with no hardware, or
              Cortex (needs a raw-EEG license — see credentials.example.yaml).
  * Decoder -> pick a montage: EPOC-X (12 ch) or research cap (32 ch).
EOF
