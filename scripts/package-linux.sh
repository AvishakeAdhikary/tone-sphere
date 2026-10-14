#!/usr/bin/env bash
# Builds the Linux AppImage (UI + engine) in a staging copy on the Linux filesystem.
#   scripts/package-linux.sh [engine-build-dir]    (default: ~/nt-build, see build-linux.sh)
# Output: ui/release/NexTone-<version>-x86_64.AppImage (copied back into the repository).
# Needs Node.js 22+ on PATH (or in ~/.local/node).
set -euo pipefail
root="$(cd "$(dirname "$0")/.." && pwd)"
build="${1:-$HOME/nt-build}"
stage="$HOME/nt-pkg"
export PATH="$HOME/.local/node/bin:$PATH"

mkdir -p "$stage/ui" "$stage/build/linux-release/bin"
# Sources only: node_modules and build outputs are platform-specific.
tar -C "$root/ui" --exclude=node_modules --exclude='release*' --exclude=dist --exclude=dist-electron \
    --exclude=test-results --exclude=playwright-report -cf - . | tar -C "$stage/ui" -xf -
cp "$root/LICENSE" "$stage/LICENSE"
cp "$build/bin/nextone-engine" "$build/bin/nextone-plugscan" "$stage/build/linux-release/bin/"
# Protocol types are shared with the engine docs; the UI imports them from src/shared.
cd "$stage/ui"
npm ci --no-audit --no-fund
npm run dist:linux
mkdir -p "$root/ui/release"
cp release/*.AppImage "$root/ui/release/"
ls -la "$root/ui/release/"*.AppImage
