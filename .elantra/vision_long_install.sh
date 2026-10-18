#!/usr/bin/env bash
# Vision Long one-command install on the comma 4. Parked, ignition OFF, as `comma` over SSH:
#
#   cd /data/openpilot && git fetch https://github.com/aaravsaianugula/openpilot.git vision-long-companion \
#     && git show FETCH_HEAD:.elantra/vision_long_install.sh | bash -s -- BASE_OP BASE_DBC TARGET_OP TARGET_DBC
#
# The script is read out of the fetched branch, so the first install works from the base
# checkout, where this file does not exist yet. A rerun (after an AGNOS update removed the link
# units) is the same command. All checks run before anything changes; the logic and the
# receipt are in .elantra/vision_long_link/install.py.
set -euo pipefail

REPO=/data/openpilot
REMOTE=https://github.com/aaravsaianugula/openpilot.git
BRANCH=vision-long-companion
PYTHON=/usr/local/venv/bin/python3

if [ "$#" -ne 4 ]; then
  echo "usage: vision_long_install.sh BASE_OPENPILOT BASE_OPENDBC TARGET_OPENPILOT TARGET_OPENDBC" >&2
  exit 2
fi

cd "$REPO"
git fetch -q "$REMOTE" "$BRANCH"
git show FETCH_HEAD:.elantra/vision_long_link/install.py | "$PYTHON" - install \
  --base-openpilot "$1" --base-opendbc "$2" --target-openpilot "$3" --target-opendbc "$4"
