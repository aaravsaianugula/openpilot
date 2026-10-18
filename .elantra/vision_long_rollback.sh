#!/usr/bin/env bash
# Vision Long rollback on the comma 4, as `comma` over SSH:
#
#   bash /data/openpilot/.elantra/vision_long_rollback.sh
#
# Restores the SHAs the install saved in /data/vision-long-rollback-*, removes the runtime
# config, the link units and the USB gadget, keeps /data/vision_long (key, toggle, logs), and
# restarts the comma service detached. Python reads the whole script before the checkout
# removes it.
set -euo pipefail

cd /data/openpilot
exec /usr/local/venv/bin/python3 .elantra/vision_long_link/install.py rollback
