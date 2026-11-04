#!/bin/sh
# Run the laptop GPU server inside WSL, on this branch's latest code. Extra args go to server.py.
cd "$(dirname "$0")/../.." || exit 1
timeout 30 git pull -q --ff-only && timeout 60 git submodule update -q --init tinygrad_repo || echo "offline: serving the code already here"
NVRTC_PATH=$(ls /root/venv/lib/python3*/site-packages/nvidia/cuda_nvrtc/lib/libnvrtc.so.12)
export NVRTC_PATH DEV="${DEV:-CUDA}"  # DEV=AMD for a Radeon laptop
exec /root/venv/bin/python .elantra/laptop_gpu/server.py "$@"
