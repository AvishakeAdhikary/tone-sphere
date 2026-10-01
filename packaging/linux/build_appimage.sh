#!/usr/bin/env bash
# The Linux download: one AppImage a user marks executable and double-clicks.
#
#   packaging/linux/build_appimage.sh <version>     (after `uv run pyinstaller tonesphere.spec`)
#
# Wraps PyInstaller's one-folder build (dist/ToneSphere) in an AppDir and packs it with
# appimagetool. appimagetool's current runtime is static, so the AppImage runs without
# libfuse2 installed — which Ubuntu 22.04 and later no longer have by default.
set -euo pipefail

version="${1:?usage: build_appimage.sh <version>}"
root="$(cd "$(dirname "$0")/../.." && pwd)"
appdir="$root/build/ToneSphere.AppDir"
out="$root/release/ToneSphere-$version-x86_64.AppImage"

rm -rf "$appdir"
mkdir -p "$appdir/usr/lib/tonesphere" "$root/release"
cp -a "$root/dist/ToneSphere/." "$appdir/usr/lib/tonesphere/"
cp "$root/packaging/linux/ToneSphere.desktop" "$appdir/tonesphere.desktop"
cp "$root/assets/images/ToneSphere.png" "$appdir/tonesphere.png"
cat > "$appdir/AppRun" <<'EOF'
#!/bin/sh
here="$(dirname "$(readlink -f "$0")")"
exec "$here/usr/lib/tonesphere/ToneSphere" "$@"
EOF
chmod +x "$appdir/AppRun"

tool="$root/build/appimagetool-x86_64.AppImage"
if [ ! -x "$tool" ]; then
  curl -sSL -o "$tool" https://github.com/AppImage/appimagetool/releases/download/continuous/appimagetool-x86_64.AppImage
  chmod +x "$tool"
fi
# CI runners have no FUSE to mount the tool itself; it can unpack and run instead.
ARCH=x86_64 APPIMAGE_EXTRACT_AND_RUN=1 "$tool" --no-appstream "$appdir" "$out"
echo "$out"
