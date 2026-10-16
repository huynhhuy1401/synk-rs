#!/bin/sh
# Regenerates the trainer's lock files (crates/synk-pipeline/locks/): every
# package of the Python training environment, pinned with hashes, per
# backend and platform. `synk setup` installs exactly these. Run after
# changing the packages or versions in requirements.in, then run the tests
# and a real `synk setup` (CUDA on a cloud GPU) before a release.
set -eu
cd "$(dirname "$0")/../crates/synk-pipeline/locks"
DATE=2026-10-14T00:00:00Z
lock() { # name platform torch-index-or-empty
  index=""
  [ -n "$3" ] && index="--index $3 --index-strategy unsafe-best-match"
  # shellcheck disable=SC2086
  uv pip compile requirements.in --quiet --python-version 3.12 \
    --python-platform "$2" $index --emit-index-url \
    --exclude-newer "$DATE" --exclude-newer-package torch=false \
    --generate-hashes --no-header -o "$1.txt"
  echo "$1: $(grep -c '==' "$1.txt") packages"
}
CPU=https://download.pytorch.org/whl/cpu
CUDA=https://download.pytorch.org/whl/cu130
lock cpu-windows-x86_64 x86_64-pc-windows-msvc "$CPU"
lock cpu-linux-x86_64 x86_64-unknown-linux-gnu "$CPU"
lock cpu-linux-aarch64 aarch64-unknown-linux-gnu "$CPU"
# PyTorch 2.14 publishes macOS wheels for 14.0 and newer only.
MACOSX_DEPLOYMENT_TARGET=14.0 lock cpu-macos-aarch64 aarch64-apple-darwin ""
lock cuda-windows-x86_64 x86_64-pc-windows-msvc "$CUDA"
lock cuda-linux-x86_64 x86_64-unknown-linux-gnu "$CUDA"
