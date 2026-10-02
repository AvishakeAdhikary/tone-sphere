"""
The plugin scanner's rules, with a stand-in for the subprocess so no plugin or native DLL is
needed: what it believes, what it caches, and what it retries. The real scanner against real
modules is in tests/native/test_vst3.py and tests/hardware/test_vst3_third_party.py.
"""

import json
import struct
import sys
import textwrap
import time

import pytest

from tonesphere.plugins import scan

CLASS = {'path': 'x', 'uid': '0' * 32, 'name': 'Amp', 'vendor': 'Someone', 'version': '7.0',
         'category': 'Audio Module Class', 'subcategories': 'Fx|Distortion', 'sdk_version': 'VST 3.7',
         'is_audio_effect': True}


def x64_module(path):
    """Enough of a PE header for the architecture check: the scanner never loads it here."""
    head = bytearray(512)
    head[:2] = b'MZ'
    struct.pack_into('<I', head, 0x3C, 0x80)
    head[0x80:0x84] = b'PE\0\0'
    struct.pack_into('<H', head, 0x84, 0x8664)
    path.write_bytes(bytes(head))
    return path


def stand_in(monkeypatch, tmp_path, body: str):
    """The scanner subprocess replaced by a Python script running `body`, with `result` its result file."""
    script = tmp_path / 'scanner.py'
    script.write_text('import json, os, sys\nresult = sys.argv[sys.argv.index("--result") + 1]\n'
                      + textwrap.dedent(body), encoding='utf-8')
    monkeypatch.setattr(scan, '_scanner_command', lambda path, result: [sys.executable, str(script), str(path),
                                                                        '--result', str(result)])


def test_a_report_written_before_a_crash_in_teardown_is_believed(monkeypatch, tmp_path):
    """Big commercial modules can fault while unloading, after their classes were read."""
    stand_in(monkeypatch, tmp_path, f'''
        open(result, "w").write(json.dumps({{"classes": [{CLASS!r}]}}))
        os._exit(0xC0000005)
    ''')
    result = scan.scan_module(x64_module(tmp_path / 'Amp.vst3'))
    assert result.status == scan.OK and [c.name for c in result.classes] == ['Amp']


def test_a_scanner_that_hangs_after_reporting_is_ended_and_believed(monkeypatch, tmp_path):
    """Guitar Rig 7's first load in a session hung the scanner's exit; the scan timed out at 120 s."""
    monkeypatch.setattr(scan, 'EXIT_GRACE_S', 0.5)
    stand_in(monkeypatch, tmp_path, f'''
        import time
        open(result, "w").write(json.dumps({{"classes": [{CLASS!r}]}}))
        time.sleep(60)
    ''')
    started = time.monotonic()
    result = scan.scan_module(x64_module(tmp_path / 'Amp.vst3'), timeout=30)
    assert result.status == scan.OK and [c.name for c in result.classes] == ['Amp']
    assert time.monotonic() - started < 15


def test_a_scanner_that_never_reports_times_out(monkeypatch, tmp_path):
    stand_in(monkeypatch, tmp_path, 'import time\ntime.sleep(60)\n')
    started = time.monotonic()
    result = scan.scan_module(x64_module(tmp_path / 'Amp.vst3'), timeout=1)
    assert result.status == scan.TIMED_OUT and time.monotonic() - started < 15


def test_the_real_scanner_reports_and_leaves(tmp_path):
    """The child's own side: the report is in place, whole, by the time the process has gone."""
    import subprocess
    result = tmp_path / 'result.json'
    code = subprocess.run([sys.executable, '-c', 'import sys; from tonesphere.plugins.scan import scan_one_cli; '
                           'scan_one_cli(sys.argv[1:])', str(tmp_path / 'missing.vst3'), '--result', str(result)],
                          cwd=scan.SOURCE_ROOT, timeout=60).returncode
    assert code == 0
    assert 'error' in json.loads(result.read_text(encoding='utf-8'))
    assert not (tmp_path / 'result.json.part').exists()


def test_a_process_that_dies_before_reporting_is_a_crash_with_its_exit_code(monkeypatch, tmp_path):
    stand_in(monkeypatch, tmp_path, '''
        sys.stderr.write("loading the module failed\\n")
        os._exit(3)
    ''')
    result = scan.scan_module(x64_module(tmp_path / 'Amp.vst3'))
    assert result.status == scan.CRASHED
    assert '0x00000003' in result.detail and 'loading the module failed' in result.detail


def test_anything_a_plugin_prints_does_not_break_the_scan(monkeypatch, tmp_path):
    stand_in(monkeypatch, tmp_path, f'''
        sys.stdout.buffer.write(b"\\xff\\xfe{{ not json\\n")
        open(result, "w").write(json.dumps({{"classes": [{CLASS!r}]}}))
    ''')
    assert scan.scan_module(x64_module(tmp_path / 'Amp.vst3')).status == scan.OK


def test_a_failed_scan_is_cached_and_retried_when_the_user_asks(monkeypatch, tmp_path):
    folder = tmp_path / 'VST3'
    folder.mkdir()
    module = x64_module(folder / 'Amp.vst3')
    cache = scan.ScanCache(tmp_path / 'cache.json')
    stand_in(monkeypatch, tmp_path, 'os._exit(1)\n')
    assert scan.scan([folder], cache)[0].status == scan.CRASHED

    stand_in(monkeypatch, tmp_path, f'open(result, "w").write(json.dumps({{"classes": [{CLASS!r}]}}))\n')
    assert scan.scan([folder], cache)[0].status == scan.CRASHED, "opening the browser reuses the cache"
    assert scan.scan([folder], cache, retry_failed=True)[0].status == scan.OK, "pressing Scan retries it"
    assert json.loads((tmp_path / 'cache.json').read_text())[str(module)]['result']['status'] == scan.OK


def test_the_scanner_child_reports_through_its_result_file(tmp_path):
    """A windowed build has no stdout: the report must reach the parent another way."""
    result = tmp_path / 'r.json'
    done = __import__('subprocess').run(
        [sys.executable, '-c', 'import sys; sys.stdout = None\n'
         'from tonesphere.plugins import scan\n'
         'scan.classes_in = lambda path: (_ for _ in ()).throw(scan.PluginError("not a plugin"))\n'
         f'scan.scan_one_cli([r"{tmp_path / "x.vst3"}", "--result", r"{result}"])'],
        cwd=scan.SOURCE_ROOT, capture_output=True, text=True, timeout=60)
    assert done.returncode == 0, done.stderr
    assert json.loads(result.read_text(encoding='utf-8')) == {'error': 'not a plugin'}


@pytest.mark.skipif(sys.platform != 'win32', reason="the frozen scanner command is the Windows build's")
def test_a_frozen_build_runs_itself_as_the_scanner(monkeypatch, tmp_path):
    monkeypatch.setattr(sys, 'frozen', True, raising=False)
    command = scan._scanner_command(tmp_path / 'Amp.vst3', tmp_path / 'r.json')
    assert command == [sys.executable, 'scan-plugin', str(tmp_path / 'Amp.vst3'), '--result', str(tmp_path / 'r.json')]
