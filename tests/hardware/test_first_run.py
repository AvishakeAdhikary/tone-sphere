"""
The frozen app a user downloads, monitoring a guitar on an interface's input 1 into its
headphones, measured from outside the app: what the output's loopback carries is compared
with what the input carries (tests/hardware/first_run_probe.py, a separate process).

v0.2.0 showed moving meters and played nothing. This proves the app started from the
session a user's first run leaves — the Monitor dialog's route, then Guitar Rig on the
input — plays the guitar into the headphones at the level its mixing law says, and that
the restored plugin really shapes that audio.

Marked `hardware`: it needs the built app (`dist/ToneSphere`, or TONESPHERE_APP), an
interface whose endpoint names contain TONESPHERE_INTERFACE (default "AI-04") with an
instrument on input 1 (its hum is enough), and, for the second test, Guitar Rig 7. The
GUI-driven run of the same path is recorded in docs/FIRST_RUN_VERIFICATION.md.
"""

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest
import yaml

from tonesphere.native import available

pytestmark = [pytest.mark.hardware,
              pytest.mark.skipif(sys.platform != 'win32' or not available(), reason="needs the Windows native engine")]

ROOT = Path(__file__).resolve().parents[2]
INTERFACE = os.environ.get('TONESPHERE_INTERFACE', 'AI-04')
APP = Path(os.environ.get('TONESPHERE_APP', ROOT / 'dist' / 'ToneSphere' / 'ToneSphere.exe'))
# Input 1 alone, to both ears at constant power: each ear is the input less 3.01 dB.
MONO_TO_BOTH_DB = -3.01


def interface():
    from tonesphere.native.wasapi import endpoints
    found = {e.flow: e for e in endpoints() if INTERFACE in e.name}
    if set(found) != {'capture', 'render'}:
        pytest.skip(f"no interface named like '{INTERFACE}' with an input and an output")
    return found['capture'], found['render']


def session(plugins: dict | None = None) -> dict:
    capture, render = interface()
    source, dest = f"device:Windows WASAPI::{capture.id}", f"device:Windows WASAPI::{render.id}"
    return {'version': 2, 'name': 'session',
            'engine': {'sample_rate': 48000, 'buffer_size': 128, 'host_api': 'Windows WASAPI', 'exclusive': False},
            'routes': [{'source': source, 'dest': dest, 'source_name': capture.name, 'dest_name': render.name,
                        'gain': 1.0, 'muted': False, 'pan': 0.0, 'inverted': False, 'source_channel': 0}],
            'plugins': {source: {'input': plugins}} if plugins else {}}


def run_app_and_measure(tmp_path: Path, preset: dict, seconds: float = 8.0) -> dict:
    if not APP.is_file():
        pytest.skip(f"no built app at {APP} (uv run pyinstaller tonesphere.spec)")
    data = tmp_path / 'data'
    data.mkdir()
    (data / 'session.yaml').write_text(yaml.safe_dump(preset, sort_keys=False), encoding='utf-8')
    app = subprocess.Popen([str(APP)], env={**os.environ, 'TONESPHERE_DATA_DIR': str(data)})
    try:
        log = data / 'logs' / 'tonesphere.log'

        def restored():
            return log.exists() and "Preset 'session'" in log.read_text('utf-8')

        deadline = time.monotonic() + 90
        while time.monotonic() < deadline and not restored():
            assert app.poll() is None, f"the app exited with {app.returncode}"
            time.sleep(0.5)
        assert restored(), "the app never restored the session"
        time.sleep(3.0)
        result = tmp_path / 'probe.json'
        subprocess.run([sys.executable, str(ROOT / 'tests' / 'hardware' / 'first_run_probe.py'), str(seconds),
                        str(result)], cwd=ROOT, env={**os.environ, 'PYTHONPATH': str(ROOT)}, check=True, timeout=120)
        measured = json.loads(result.read_text('utf-8'))
        print('\n' + log.read_text('utf-8').splitlines()[-1])
    finally:
        app.terminate()
        app.wait(30)
    if measured['input_1_db'] < -70:
        pytest.skip(f"input 1 is silent ({measured['input_1_db']:.1f} dBFS): plug an instrument in")
    return measured


def heard_db(m: dict) -> float:
    return m['left_db'] - m['input_1_db']


def test_the_monitored_guitar_reaches_both_ears_at_the_mono_law(tmp_path):
    m = run_app_and_measure(tmp_path, session())
    print(f"input 1 {m['input_1_db']:.2f} dBFS ({m['input_1_hz']:.1f} Hz); headphones L {m['left_db']:.2f} / "
          f"R {m['right_db']:.2f} dBFS ({heard_db(m):+.2f} dB); coherence with input 1 {m['left_coherence']:.3f}")
    assert m['left_coherence'] > 0.8 and m['right_coherence'] > 0.8, "what the headphones carry is not the guitar"
    assert m['left_hz'] == pytest.approx(m['input_1_hz'], abs=1.0)
    assert m['left_db'] == pytest.approx(m['right_db'], abs=0.1)
    assert heard_db(m) == pytest.approx(MONO_TO_BOTH_DB, abs=0.5)


def test_guitar_rig_restored_on_the_input_shapes_what_is_heard(tmp_path):
    from tonesphere.plugins import PluginInstance
    from tonesphere.plugins import scan as scanner

    rig = next((c for r in scanner.scan() for c in r.effects if c.name.startswith('Guitar Rig')), None)
    if rig is None:
        pytest.skip("Guitar Rig is not installed in a standard VST3 folder")
    with PluginInstance(rig, 48000, 128, 2) as plugin:
        volume = next(p for p in plugin.parameters() if p.title == 'Rack Master Volume')
        plugin.set_parameter(volume.id, 0.6)
        shown = next(p for p in plugin.parameters() if p.id == volume.id).display.strip()
        state = plugin.state()
    assert shown == '-12.0dB', shown
    entry = {'kind': 'vst3', 'path': rig.path, 'uid': rig.uid, 'name': rig.name, 'vendor': rig.vendor,
             'version': rig.version, 'bypassed': False, 'state': state.to_dict()}
    m = run_app_and_measure(tmp_path, session([entry]))
    print(f"through Guitar Rig at {shown}: headphones {heard_db(m):+.2f} dB against input 1, "
          f"coherence {m['left_coherence']:.3f}")
    assert m['left_coherence'] > 0.6, "the guitar no longer reaches the headphones through the plugin"
    assert heard_db(m) == pytest.approx(MONO_TO_BOTH_DB - 12.0, abs=0.75)
