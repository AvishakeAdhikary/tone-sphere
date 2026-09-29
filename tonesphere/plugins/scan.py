"""
Find VST3 plugins and read what they are, without letting a broken one take ToneSphere down.

Loading a plugin module runs the plugin's own code (its DLL initialisation and its
factory), so every module is scanned in a subprocess with a timeout. A module that
crashes, hangs or is built for another architecture is recorded with the reason, and
never loaded into the main process by the scanner. Results are cached by path, size and
modification time, so an unchanged plugin is not reloaded on every start.

    python -m tonesphere.plugins.scan <path.vst3>     # scan one module, print JSON
"""

import json
import os
import struct
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

from tonesphere.plugins import PluginError, PluginInfo, classes_in

SCAN_TIMEOUT_S = 30.0

OK = 'ok'
CRASHED = 'crashed'
TIMED_OUT = 'timed out'
FAILED = 'failed'
WRONG_ARCHITECTURE = 'wrong architecture'
NOT_A_PLUGIN = 'no audio effect classes'

_MACHINES = {0x8664: 'x64', 0x14C: 'x86', 0xAA64: 'arm64', 0xA641: 'arm64ec'}


def standard_paths() -> list[Path]:
    """The VST3 folders the VST3 specification defines for Windows."""
    paths = []
    common = os.environ.get('COMMONPROGRAMFILES')
    if common:
        paths.append(Path(common) / 'VST3')
    local = os.environ.get('LOCALAPPDATA')
    if local:
        paths.append(Path(local) / 'Programs' / 'Common' / 'VST3')
    return paths


def find_modules(roots) -> list[Path]:
    """
    Every VST3 module under `roots`: bundle directories (`X.vst3/Contents/...`) and legacy
    single-file `.vst3` DLLs, found recursively — vendors nest them in their own folders.
    A bundle is not searched inside: its contents are one plugin, not several.
    """
    found = []
    for root in roots:
        root = Path(root)
        if not root.exists():
            continue
        if root.suffix.lower() == '.vst3':
            found.append(root)
            continue
        for dirpath, dirnames, filenames in os.walk(root):
            here = Path(dirpath)
            bundles = [d for d in dirnames if d.lower().endswith('.vst3')]
            for d in bundles:
                found.append(here / d)
            dirnames[:] = [d for d in dirnames if d not in bundles]
            found.extend(here / f for f in filenames if f.lower().endswith('.vst3'))
    return sorted(set(found))


def module_binary(module: Path) -> Path | None:
    """The DLL that actually loads: inside a bundle's x86_64-win folder, or the file itself."""
    if module.is_file():
        return module
    arch = module / 'Contents' / 'x86_64-win'
    if arch.is_dir():
        candidates = sorted(arch.glob('*.vst3'))
        if candidates:
            return candidates[0]
    return None


def architecture(binary: Path) -> str:
    """The PE machine type, read from the header without loading anything."""
    try:
        with open(binary, 'rb') as f:
            head = f.read(4096)
    except OSError:
        return 'unreadable'
    if len(head) < 0x40 or head[:2] != b'MZ':
        return 'not a DLL'
    offset = struct.unpack_from('<I', head, 0x3C)[0]
    if offset + 6 > len(head) or head[offset:offset + 4] != b'PE\0\0':
        return 'not a DLL'
    machine = struct.unpack_from('<H', head, offset + 4)[0]
    return _MACHINES.get(machine, f'machine 0x{machine:04x}')


@dataclass
class ScanResult:
    path: str
    status: str
    detail: str = ''
    classes: list[PluginInfo] = field(default_factory=list)
    architecture: str = ''

    @property
    def effects(self) -> list[PluginInfo]:
        return [c for c in self.classes if c.is_audio_effect]

    def to_dict(self) -> dict:
        return {'path': self.path, 'status': self.status, 'detail': self.detail,
                'architecture': self.architecture, 'classes': [c.to_dict() for c in self.classes]}

    @classmethod
    def from_dict(cls, d: dict) -> "ScanResult":
        return cls(d['path'], d['status'], d.get('detail', ''), [PluginInfo.from_dict(c) for c in d.get('classes', [])],
                   d.get('architecture', ''))


def _scanner_command(path: Path) -> list[str]:
    if getattr(sys, 'frozen', False):
        return [sys.executable, 'scan-plugin', str(path)]
    return [sys.executable, '-m', 'tonesphere.plugins.scan', str(path)]


def scan_module(path: Path, timeout: float = SCAN_TIMEOUT_S) -> ScanResult:
    path = Path(path)
    binary = module_binary(path)
    if binary is None:
        return ScanResult(str(path), FAILED, "no x86_64-win binary inside the bundle")
    arch = architecture(binary)
    if arch != 'x64':
        return ScanResult(str(path), WRONG_ARCHITECTURE, f"built for {arch}; ToneSphere is x64", architecture=arch)

    try:
        done = subprocess.run(_scanner_command(path), capture_output=True, text=True, timeout=timeout,
                              creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
    except subprocess.TimeoutExpired:
        return ScanResult(str(path), TIMED_OUT, f"did not finish loading within {timeout:.0f} s", architecture=arch)

    lines = [line for line in done.stdout.splitlines() if line.startswith('{')]
    if done.returncode != 0 or not lines:
        code = done.returncode & 0xFFFFFFFF
        return ScanResult(str(path), CRASHED, f"the scanner process died (exit code 0x{code:08X}) while loading it",
                          architecture=arch)
    report = json.loads(lines[-1])
    if 'error' in report:
        status = CRASHED if 'crashed' in report['error'] else FAILED
        return ScanResult(str(path), status, report['error'], architecture=arch)
    classes = [PluginInfo.from_dict(c) for c in report['classes']]
    status = OK if any(c.is_audio_effect for c in classes) else NOT_A_PLUGIN
    return ScanResult(str(path), status, '', classes, architecture=arch)


class ScanCache:
    """Scan results keyed by module path, trusted only while the binary's size and mtime match."""

    def __init__(self, file: Path):
        self.file = Path(file)
        try:
            self._entries = json.loads(self.file.read_text(encoding='utf-8'))
        except (OSError, ValueError):
            self._entries = {}

    @staticmethod
    def _stamp(path: Path) -> list | None:
        binary = module_binary(path)
        if binary is None:
            return None
        s = binary.stat()
        return [s.st_size, int(s.st_mtime)]

    def get(self, path: Path) -> ScanResult | None:
        entry = self._entries.get(str(path))
        if entry and entry.get('stamp') == self._stamp(path):
            return ScanResult.from_dict(entry['result'])
        return None

    def put(self, result: ScanResult):
        self._entries[result.path] = {'stamp': self._stamp(Path(result.path)), 'result': result.to_dict(),
                                      'scanned': time.time()}

    def save(self):
        self.file.parent.mkdir(parents=True, exist_ok=True)
        self.file.write_text(json.dumps(self._entries, indent=1), encoding='utf-8')


def scan(roots=None, cache: ScanCache | None = None, timeout: float = SCAN_TIMEOUT_S) -> list[ScanResult]:
    """
    Scan every module under `roots` (the standard VST3 folders by default). Duplicate
    classes — the same class ID in two modules, usually two installed versions — are
    reported on the later module instead of silently shadowing the earlier one.
    """
    results = []
    for module in find_modules(roots if roots is not None else standard_paths()):
        result = cache.get(module) if cache else None
        if result is None:
            result = scan_module(module, timeout)
            if cache:
                cache.put(result)
        results.append(result)
    seen: dict[str, str] = {}
    for result in results:
        for c in result.effects:
            if c.uid in seen and seen[c.uid] != result.path:
                result.detail = (result.detail + '; ' if result.detail else '') + \
                    f"{c.name} duplicates the class in {seen[c.uid]}"
            seen.setdefault(c.uid, result.path)
    if cache:
        cache.save()
    return results


def scan_one_cli(path: str) -> int:
    """The subprocess side: load one module, print one JSON line, exit."""
    if os.environ.get('TONESPHERE_SCAN_TEST_ABORT') == '1':
        os.abort()  # stands in for a crash no handler can catch, for the tests
    try:
        classes = classes_in(path)
    except PluginError as e:
        print(json.dumps({'error': str(e)}))
        return 0
    print(json.dumps({'classes': [c.to_dict() for c in classes]}))
    return 0


if __name__ == '__main__':
    raise SystemExit(scan_one_cli(sys.argv[1]))
