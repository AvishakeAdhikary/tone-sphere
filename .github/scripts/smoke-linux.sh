#!/usr/bin/env bash
# Runs the Linux download on a clean runner: unpacks the AppImage, starts its engine the way the
# app does (--background) and checks that it answers with the released version.
#   .github/scripts/smoke-linux.sh ToneSphere-1.0.0-x86_64.AppImage 1.0.0
set -euo pipefail
appimage="$(realpath "$1")"
version="$2"
work="$(mktemp -d)"
cd "$work"
chmod +x "$appimage"
"$appimage" --appimage-extract >/dev/null
engine="$work/squashfs-root/resources/engine/tonesphere-engine"
test -x "$engine" || { echo "FAIL  engine missing from the AppImage"; exit 1; }
export XDG_RUNTIME_DIR="$work/run" XDG_DATA_HOME="$work/data"
mkdir -p "$XDG_RUNTIME_DIR"
"$engine" --background &
pid=$!
disc="$XDG_RUNTIME_DIR/tonesphere/engine.json"
for _ in $(seq 1 60); do [ -f "$disc" ] && break; sleep 0.5; done
test -f "$disc" || { echo "FAIL  the engine did not start"; kill "$pid" || true; exit 1; }
port=$(python3 -c "import json,sys;print(json.load(open(sys.argv[1]))['port'])" "$disc")
token=$(python3 -c "import json,sys;print(json.load(open(sys.argv[1]))['token'])" "$disc")
got=$(curl -fsS -X POST "http://127.0.0.1:$port/rpc" -H "Authorization: Bearer $token" -H 'Content-Type: application/json' \
  -d '{"id":1,"method":"engine.getState","params":{}}' | python3 -c "import json,sys;print(json.load(sys.stdin)['result']['version'])")
kill "$pid" || true
if [ "$got" != "$version" ]; then echo "FAIL  version $got, expected $version"; exit 1; fi
echo "PASS  the AppImage's engine starts and answers (version $got)"
