#!/bin/bash
# serve.sh - Start Kokoro TTS Server

# Source conda (adjust path if needed)
CONDA_PATH="$HOME/miniforge3/etc/profile.d/conda.sh"
if [ -f "$CONDA_PATH" ]; then
    source "$CONDA_PATH"
else
    echo "Conda not found at $CONDA_PATH"
    exit 1
fi

conda activate kokoro

# Start the server
python kokoro_svr.py --port 8803
