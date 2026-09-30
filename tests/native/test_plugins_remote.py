"""
Effect chains from the REST API and the CLI, on the native engine with no device: a bus
feeding a network send's ring, so what the chain did is read back exactly.

Through REST, ToneSphere's own test plugin (gain parameter normalised 0..1 -> x0..x2) goes
onto a bus, its gain is set to x0.5 by parameter id, and a 1 kHz tone written into the bus
must come out at exactly half its level; a built-in EQ is added, moved ahead of it, and
removed. Through the CLI, the same with a built-in effect. Each is the same engine call the
desktop window makes; what is proved is that the remote surfaces reach it.
"""

import sys
import time
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from tests.native.test_engine_native_host import collect, network_sink
from tests.signals import RATE, dominant_frequency, rms, sine

pytestmark = pytest.mark.skipif(sys.platform != 'win32', reason="the native host is Windows-only")

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def test_gain_path() -> str:
    found = sorted(ROOT.glob("native/build/*/VST3/Release/tonesphere_test_gain.vst3"))
    if not found:
        pytest.fail("the test plugin is not built (scripts/build_native.py)")
    return str(found[0])


@pytest.fixture
def rig():
    from fastapi.testclient import TestClient

    import tonesphere.api.server as server
    from tonesphere.core.engine_factory import UnifiedAudioEngine

    unified = UnifiedAudioEngine()
    unified.initialize()
    engine = unified.engine
    started, message = engine.start_udp_transport('127.0.0.1', 0)
    assert started, message
    bus = engine.create_virtual_input("fx", channels=2)
    source, dest = network_sink(engine, bus)
    engine.start_engine()
    time.sleep(0.2)
    server.audio_engine = unified
    try:
        yield TestClient(server.app), unified, bus, source, dest
    finally:
        server.audio_engine = None
        unified.cleanup()


def through(engine, bus, source, dest, tone):
    """Feed the tone in as the engine takes it, reading the ring as it fills: both rings are small."""
    collect(engine, source, dest, 0.1)
    written, out = 0, []
    deadline = time.time() + len(tone) / RATE + 2.0
    while written < len(tone) and time.time() < deadline:
        written += engine.write_to_bus(bus, tone[written:written + 2048])
        block = engine.host.read_available(source, dest, RATE)
        if block is not None and len(block):
            out.append(block)
        time.sleep(0.005)
    assert written == len(tone)
    out.append(collect(engine, source, dest, 0.4))
    heard = np.concatenate(out)[:, 0].astype(np.float64)
    loud = np.nonzero(np.abs(heard) > 1e-4)[0]
    assert len(loud), "nothing came through the bus"
    return heard[loud[0] + RATE // 10: loud[0] + RATE // 10 + RATE // 4]


def test_a_plugin_driven_entirely_over_rest(rig, test_gain_path):
    client, unified, bus, source, dest = rig

    listed = client.get("/plugins", params={"path": str(Path(test_gain_path).parent)}).json()
    assert any(c['name'] == 'ToneSphere Test Gain' for r in listed for c in r['classes'])

    response = client.post(f"/chains/{bus}/vst3", json={"path": test_gain_path})
    assert response.status_code == 200, response.text
    assert [e['name'] for e in response.json()['chain']] == ['ToneSphere Test Gain']

    parameters = client.get(f"/chains/{bus}/0/parameters").json()
    gain = next(p for p in parameters if p['title'] == 'Gain')
    set_to = client.put(f"/chains/{bus}/0/parameters/{gain['id']}", params={"normalized": 0.25}).json()
    assert set_to['normalized'] == pytest.approx(0.25)

    time.sleep(0.1)
    tone = sine(RATE, 1000.0, amplitude=0.4)
    heard = through(unified.engine, bus, source, dest, tone)
    print(f"\nover REST: Test Gain at {set_to['display']} -> rms {rms(heard):.5f} of {rms(tone[:, 0]):.5f}")
    assert dominant_frequency(heard) == pytest.approx(1000.0, abs=5.0)
    assert rms(heard) == pytest.approx(0.5 * rms(tone[:, 0]), rel=0.01)

    assert client.post(f"/chains/{bus}/builtin", json={"type": "eq"}).status_code == 200
    assert client.put(f"/chains/{bus}/1/move", params={"to": 0}).status_code == 200
    assert [e['kind'] for e in client.get(f"/chains/{bus}").json()] == ['builtin', 'vst3']
    assert client.put(f"/chains/{bus}/1/bypass", params={"bypassed": True}).status_code == 200
    assert client.get(f"/chains/{bus}").json()[1]['bypassed'] is True
    assert client.delete(f"/chains/{bus}/0").status_code == 200
    assert client.delete(f"/chains/{bus}/5").status_code == 404
    assert client.post(f"/chains/{bus}/builtin", json={"type": "reverb"}).status_code == 400
    assert client.post(f"/chains/{bus}/vst3", json={"path": "C:/nowhere/none.vst3"}).status_code == 404


def test_an_effect_added_and_set_from_the_cli(rig, capsys):
    from tonesphere.cli.interface import AudioEngineCLI

    _client, unified, bus, source, dest = rig
    cli = AudioEngineCLI()
    cli.engine = unified
    with patch('builtins.input', side_effect=[str(bus), 'output', 'eq']):
        cli.add_effect()
    assert 'EQ' in capsys.readouterr().out
    # Band 1: type (id 0) to high-pass (4 of 5), frequency (id 1) to 1 kHz (log scale 20 Hz..20 kHz)
    with patch('builtins.input', side_effect=[str(bus), '0', 'output', '0', str(4 / 5)]):
        cli.set_effect_parameter()
    with patch('builtins.input', side_effect=[str(bus), '0', 'output', '1', str(np.log(50) / np.log(1000))]):
        cli.set_effect_parameter()
    assert '1.00 kHz' in capsys.readouterr().out
    low = through(unified.engine, bus, source, dest, sine(RATE, 100.0, amplitude=0.4))
    assert 20 * np.log10(rms(low) / (0.4 / np.sqrt(2))) <= -20.0
