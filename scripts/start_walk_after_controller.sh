#!/bin/bash

# Setup Instructions for Raspberry Pi OS Lite (Auto-start on boot):
# 1. Make this script executable:
#    chmod +x start_walk_after_controller.sh
# 2. Add to crontab:
#    crontab -e
#    Add the following line at the end:
#    @reboot /bin/bash /home/pi/Open_Duck_Mini_Runtime/scripts/start_walk_after_controller.sh >> /home/pi/startup.log 2>&1
# 3. Ensure your ONNX model and duck_config.json are in the home directory as specified below.

set -euo pipefail

# Ensure HOME is defined (crontab sometimes has a limited environment)
export HOME="${HOME:-/home/duck0}"

# Initialize Conda for the current shell session
CONDA_PROFILE="$HOME/miniconda3/etc/profile.d/conda.sh"
if [ -f "$CONDA_PROFILE" ]; then
    source "$CONDA_PROFILE"
    conda activate base
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

cd "$SCRIPT_DIR"

exec python3 "$SCRIPT_DIR/start_walk_after_controller.py" \
    --duck_config_path "$HOME/duck_config.json" \
    --onnx_model_path "$HOME/BEST_WALK_ONNX_2.onnx"