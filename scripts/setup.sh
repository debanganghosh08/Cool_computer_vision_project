#!/usr/bin/env bash
# Setup for macOS and Linux.
#
#   bash scripts/setup.sh
#
# Creates a virtual environment in .venv, installs the pinned dependencies into
# it, fetches the hand landmark model, and runs the tests. Nothing is installed
# outside this folder.

set -euo pipefail
cd "$(dirname "$0")/.."

echo
echo "[1/4] Python"
PY=${PYTHON:-python3}
command -v "$PY" >/dev/null || { echo "Python 3 not found. Set PYTHON=/path/to/python3 and retry."; exit 1; }
"$PY" --version

if [ ! -d .venv ]; then
  echo
  echo "[2/4] Creating .venv"
  "$PY" -m venv .venv
else
  echo
  echo "[2/4] .venv already exists, reusing it"
fi

VENV_PY=".venv/bin/python"
[ -x "$VENV_PY" ] || { echo "Could not find $VENV_PY - delete the .venv folder and run this again."; exit 1; }

echo
echo "[3/4] Installing dependencies (this pulls ~400 MB, give it a minute)"
"$VENV_PY" -m pip install --upgrade pip --quiet
if ! "$VENV_PY" -m pip install -r requirements.txt; then
  echo
  echo "Install failed. If the message mentions mediapipe and you are on Python 3.13,"
  echo "see docs/INSTALL.md - the short version is: pip install 'mediapipe>=0.10.30' opencv-python numpy"
  exit 1
fi

echo
echo "[4/4] Model and tests"
"$VENV_PY" scripts/get_model.py
"$VENV_PY" tests/test_geometry.py

echo
echo "Done. Start it with:"
echo "    source .venv/bin/activate"
echo "    python -m handportal --input 0 --preview"
echo
