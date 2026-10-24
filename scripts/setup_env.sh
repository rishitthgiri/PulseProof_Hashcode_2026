#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON_BIN="${PYTHON_BIN:-python3.12}"

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
    echo "Python 3.12 not found."
    echo "Set PYTHON_BIN to an installed Python 3.11 or 3.12 interpreter."
    exit 1
fi

if [ ! -d .venv312 ]; then
    "$PYTHON_BIN" -m venv .venv312
fi

source .venv312/bin/activate

python -m pip install --upgrade pip setuptools wheel

if [ -f requirements.txt ]; then
    python -m pip install -r requirements.txt
fi

python -m pip install \
    numpy scipy pandas scikit-learn \
    opencv-python mediapipe matplotlib streamlit

echo
echo "Python environment is ready."
python --version
python -m pip --version

if command -v ffmpeg >/dev/null 2>&1; then
    ffmpeg -version | head -n 1
else
    echo "WARNING: FFmpeg is not installed or not on PATH."
fi
