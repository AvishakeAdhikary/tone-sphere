#!/usr/bin/env bash
# Builds the ToneSphere engine on Linux (PipeWire backend) and runs the unit tests.
#   scripts/build-linux.sh [build-dir]          (default: ~/ts-build)
# Needs: build-essential cmake ninja-build pkg-config libpipewire-0.3-dev git
# From Windows/WSL: wsl -e bash /mnt/c/Projects/ToneSphere/scripts/build-linux.sh
set -euo pipefail
root="$(cd "$(dirname "$0")/.." && pwd)"
build="${1:-$HOME/ts-build}"
cmake -S "$root" -B "$build" -G Ninja -DCMAKE_BUILD_TYPE=Release -DTONESPHERE_BUILD_TESTS=ON
cmake --build "$build" -j"$(nproc)"
ctest --test-dir "$build" --output-on-failure --timeout 600
echo "Build output: $build/bin"
