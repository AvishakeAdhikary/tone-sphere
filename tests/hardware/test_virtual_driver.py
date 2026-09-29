"""
The ToneSphere virtual audio driver as Windows sees it. Runs only where the driver is
installed — which, by AGENTS.md, is only a test VM with test-signing on (install with
vm_kit/driver_install.ps1). Everywhere else it skips, saying so.

Proven here, when it runs: Windows enumerates both endpoints; another process (PortAudio,
not ToneSphere's engine) can open them; audio rendered into "ToneSphere Cable Input" by one
application arrives at another application capturing "ToneSphere Cable Output"; and the
same in both directions with ToneSphere as one of the two applications, through a VST3 plugin.
"""

import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pytest

from tests.signals import dominant_frequency, rms, sine
from tonesphere.native import NativeEngine, Node, Route, available
from tonesphere.native.wasapi import StreamSpec, endpoints

NOT_INSTALLED = ("VIRTUAL DEVICE VERIFICATION: the ToneSphere driver is not installed on this machine "
                 "(it is installed only in a test VM; see docs/VIRTUAL_AUDIO_DRIVER.md)")
RATE = 48000
PEER = Path(__file__).with_name('cable_peer.py')
ROOT = Path(__file__).resolve().parents[2]

pytestmark = [pytest.mark.hardware, pytest.mark.skipif(not available(), reason="needs tonesphere_native.dll")]


@pytest.fixture(scope='module')
def cable():
    found = {e.flow: e for e in endpoints() if 'ToneSphere' in e.name}
    if 'render' not in found or 'capture' not in found:
        pytest.skip(NOT_INSTALLED)
    return found


def peer(*args, timeout=30):
    return subprocess.run([sys.executable, str(PEER), *map(str, args)], capture_output=True, text=True,
                          timeout=timeout, check=True)


def settled(captured, seconds=0.5):
    loud = np.nonzero(np.abs(captured[:, 0]) > 1e-5)[0]
    assert len(loud) > RATE // 4, "nothing crossed the cable"
    return captured[loud[0] + RATE // 10: loud[0] + RATE // 10 + int(RATE * seconds)]


def test_windows_enumerates_both_endpoints_with_one_format(cable):
    render, capture = cable['render'], cable['capture']
    assert 'Cable Input' in render.name or 'ToneSphere' in render.name
    assert render.mix_sample_rate == capture.mix_sample_rate == 48000
    assert render.mix_channels == capture.mix_channels == 2
    print(f"\nrender: {render.name} [{render.id}]\ncapture: {capture.name} [{capture.id}]")


def test_one_application_to_another_through_the_cable(cable, tmp_path):
    """Two PortAudio processes, neither of them ToneSphere: the cable alone carries the tone."""
    out = tmp_path / 'heard.npy'
    recorder = subprocess.Popen([sys.executable, str(PEER), 'record', 'ToneSphere', 3.0, out])
    time.sleep(0.5)
    peer('play', 'ToneSphere', 2.0, 1000, 0.2)
    recorder.wait(timeout=30)
    heard = settled(np.load(out))
    assert dominant_frequency(heard) == pytest.approx(1000.0, abs=3.0)
    assert rms(heard) == pytest.approx(0.2 / np.sqrt(2), rel=0.05)


def test_an_application_into_tonesphere(cable):
    """Another process plays into Cable Input; ToneSphere's native engine captures Cable Output."""
    with NativeEngine(RATE, 480) as engine:
        engine.apply_plan([Node.source(1, 2), Node.sink(2, 2, ring_frames=RATE * 6)], [Route(1, 2)])
        engine.start_wasapi([StreamSpec(1, 'capture', 2, cable['capture'].id)])
        player = subprocess.Popen([sys.executable, str(PEER), 'play', 'ToneSphere', 2.0, 1000, 0.2])
        player.wait(timeout=30)
        time.sleep(0.3)
        heard = engine.port_read(2, RATE * 6)
        engine.stop_backend()
    heard = settled(heard)
    assert dominant_frequency(heard) == pytest.approx(1000.0, abs=3.0)
    assert rms(heard) == pytest.approx(0.2 / np.sqrt(2), rel=0.05)


def test_tonesphere_through_a_plugin_into_another_application(cable, tmp_path):
    """ToneSphere renders a tone through its VST3 test plugin (x0.5) into Cable Input; another process records."""
    from tonesphere.native import VST3, Insert
    from tonesphere.plugins import PluginInstance, classes_in

    module = sorted(ROOT.glob("native/build/*/VST3/Release/tonesphere_test_gain.vst3"))[0]
    info = next(c for c in classes_in(module) if c.name == "ToneSphere Test Gain")
    out = tmp_path / 'heard.npy'
    with PluginInstance(info, RATE, 4096, 2) as plugin, NativeEngine(RATE, 4096) as engine:
        plugin.set_parameter(0, 0.25)
        engine.apply_plan([Node.source(1, 2, ring_frames=RATE * 4), Node.sink(2, 2)], [Route(1, 2)],
                          [Insert(1, 0, VST3, plugin=plugin.handle)])
        engine.port_write(1, sine(RATE * 3, 1000.0, amplitude=0.2))
        recorder = subprocess.Popen([sys.executable, str(PEER), 'record', 'ToneSphere', 2.5, out])
        time.sleep(0.3)
        engine.start_wasapi([StreamSpec(2, 'render', 2, cable['render'].id, period_frames=480)])
        recorder.wait(timeout=30)
        engine.stop_backend()
    heard = settled(np.load(out))
    assert dominant_frequency(heard) == pytest.approx(1000.0, abs=3.0)
    assert rms(heard) == pytest.approx(0.2 / np.sqrt(2) * 0.5, rel=0.05), "the plugin's gain must cross the cable"


def test_nothing_rendered_is_silence(cable):
    with NativeEngine(RATE, 480) as engine:
        engine.apply_plan([Node.source(1, 2), Node.sink(2, 2, ring_frames=RATE * 2)], [Route(1, 2)])
        engine.start_wasapi([StreamSpec(1, 'capture', 2, cable['capture'].id)])
        time.sleep(1.0)
        heard = engine.port_read(2, RATE * 2)
        engine.stop_backend()
    assert len(heard) > RATE // 2 and np.max(np.abs(heard)) == 0.0, "an idle cable must be exact silence"
