#!/usr/bin/env bash
# The macOS download: ToneSphere.app in a disk image, beside a link to Applications.
#
#   packaging/macos/build_dmg.sh <version>     (after `uv run pyinstaller tonesphere.spec`)
#
# The app is signed ad hoc only: there is no Apple Developer ID, so Gatekeeper asks the user
# to confirm it the first time (right-click -> Open), as the release notes say. Ad-hoc signing
# still matters: Apple silicon refuses to run unsigned code at all.
set -euo pipefail

version="${1:?usage: build_dmg.sh <version>}"
root="$(cd "$(dirname "$0")/../.." && pwd)"
app="$root/dist/ToneSphere.app"
stage="$root/build/dmg"
out="$root/release/ToneSphere-$version-macos.dmg"

codesign --force --deep --sign - "$app"
codesign --verify --deep --strict "$app"

rm -rf "$stage"
mkdir -p "$stage" "$root/release"
cp -R "$app" "$stage/"
ln -s /Applications "$stage/Applications"
hdiutil create -volname "ToneSphere $version" -srcfolder "$stage" -ov -format UDZO "$out"
echo "$out"
