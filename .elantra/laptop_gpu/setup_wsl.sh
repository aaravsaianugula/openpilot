#!/bin/sh
# One-time setup inside the laptop's WSL Ubuntu, as root. Called by setup.ps1.
set -e
BRANCH="${1:-master}"
apt-get update -qq && apt-get install -y -qq git python3-venv python3-pip >/dev/null
if [ ! -d /root/openpilot ]; then
  # blobless + sparse: the server needs only .elantra, laptop.py and tinygrad, plus history to look up model refs
  GIT_LFS_SKIP_SMUDGE=1 git clone -q --filter=blob:none --sparse -b "$BRANCH" https://github.com/aaravsaianugula/openpilot /root/openpilot
  git -C /root/openpilot sparse-checkout set .elantra openpilot/selfdrive/modeld tinygrad_repo
fi
git -C /root/openpilot submodule update -q --init tinygrad_repo
[ -x /root/venv/bin/python ] || python3 -m venv /root/venv
/root/venv/bin/pip install -q numpy onnxruntime nvidia-cuda-nvrtc-cu12
sh /root/openpilot/.elantra/laptop_gpu/run.sh --prepare
