#!/usr/bin/env bash
# Vision Long laptop monitor install on the comma 4. Parked, ignition OFF, as `comma` over SSH,
# after the Vision Long install (which creates /data/vision_long and the link key):
#
#   bash /data/openpilot/.elantra/vision_long_supervisor/install.sh
#
# Every check runs before anything changes; a failure after the first change is undone. A rerun
# is safe. The logic and the receipt are in persist.py. It does not restart the comma service.
set -euo pipefail
cd /data/openpilot
exec /usr/local/venv/bin/python3 .elantra/vision_long_supervisor/persist.py install
