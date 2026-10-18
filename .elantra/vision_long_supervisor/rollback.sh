#!/usr/bin/env bash
# Remove the Vision Long laptop monitor's units, as `comma` over SSH, parked:
#
#   bash /data/openpilot/.elantra/vision_long_supervisor/rollback.sh
#
# Keeps /data/vision_long (key, logs). Removes the marker, so the boot hook stops putting the
# monitor back. The USB link units are R4's and are left alone (.elantra/vision_long_rollback.sh).
set -euo pipefail
cd /data/openpilot
exec /usr/local/venv/bin/python3 .elantra/vision_long_supervisor/persist.py rollback
