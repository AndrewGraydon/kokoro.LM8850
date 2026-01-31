#!/bin/bash
# startup.sh - Generate systemd service for Kokoro TTS

SERVICE_NAME="kokoro"
USER_NAME=$(whoami)
WORKING_DIR=$(pwd)
PYTHON_PATH="/home/andrew/miniforge3/envs/kokoro_test/bin/python"

cat <<EOF > ${SERVICE_NAME}.service
[Unit]
Description=Kokoro TTS Service for LLM-8850
After=network.target

[Service]
Type=simple
User=${USER_NAME}
WorkingDirectory=${WORKING_DIR}
# We can run directly with the env python or use serve.sh.
# Using serve.sh to ensure environment variables are set if complex activation is needed, 
# but direct python call is usually cleaner for systemd if paths are absolute.
# However, the user specifically asked for serve.sh to activate the environment.
# Let's use serve.sh but ensure it works in non-interactive shell.
ExecStart=${WORKING_DIR}/serve.sh
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
EOF

echo "Generated ${SERVICE_NAME}.service"
echo "To install:"
echo "sudo mv ${SERVICE_NAME}.service /etc/systemd/system/"
echo "sudo systemctl daemon-reload"
echo "sudo systemctl enable ${SERVICE_NAME}"
echo "sudo systemctl start ${SERVICE_NAME}"
