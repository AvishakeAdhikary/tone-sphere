#!/usr/bin/env bash
# Installs what the Linux build needs (Ubuntu 24.04 / WSL), from a fresh clone.
#   scripts/setup-dev.sh
# Packages: compiler, CMake, Ninja, PipeWire (headers + daemon for the integration test), and
# Node.js 22 in ~/.local/node when no Node.js is installed.
set -euo pipefail
sudo apt-get update
sudo apt-get install -y build-essential cmake ninja-build pkg-config git curl xz-utils \
  libpipewire-0.3-dev pipewire pipewire-pulse wireplumber
if ! command -v node >/dev/null 2>&1 && [ ! -x "$HOME/.local/node/bin/node" ]; then
  ver=v22.20.0
  mkdir -p "$HOME/.local/node"
  curl -fsSL "https://nodejs.org/dist/$ver/node-$ver-linux-x64.tar.xz" | tar -xJ -C "$HOME/.local/node" --strip-components=1
  echo 'Node.js installed in ~/.local/node (add ~/.local/node/bin to PATH)'
fi
echo 'Ready: scripts/build-linux.sh, then scripts/package-linux.sh'
