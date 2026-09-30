"""
The ToneSphere virtual cables as Windows sees them. Runs only where the driver is installed
— which, by AGENTS.md, is only a test VM with test-signing on (vm_kit/driver_install.ps1,
then `main.py cable-admin install-cables`). Everywhere else it skips, saying so.

Proven here, when it runs: a fresh install lists two cables, each its own device with its
own named endpoints; another process (PortAudio, not ToneSphere) plays through each; a tone
in one cable is exact silence in the other; ToneSphere reads and writes them, through a
VST3 plugin; a cable is added, renamed and uninstalled through the same code the Virtual
Cables dialog calls, and the dialog itself drives a real disable and enable; a cable
disabled while ToneSphere streams from it stops that stream, leaves the other cable
playing, and is reopened, with no one asking, when it is enabled again; and ffmpeg, a third
program, records what ToneSphere plays into a cable.
"""

import os
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
ONE, TWO = 'ToneSphere Cable 1', 'ToneSphere Cable 2'
TONE_RMS = 0.2 / np.sqrt(2)

pytestmark = [pytest.mark.hardware, pytest.mark.skipif(not available(), reason="needs tonesphere_native.dll")]


def cable_endpoints() -> dict[str, dict]:
    """Endpoints by cable name: Windows calls them 'Speakers (<cable>)' and 'Microphone Array (<cable>)'."""
    found: dict[str, dict] = {}
    for e in endpoints():
        if e.name.endswith(')') and '(' in e.name:
            name = e.name[e.name.rindex('(') + 1:-1]
            found.setdefault(name, {})[e.flow] = e
    return found


@pytest.fixture(scope='module')
def cables():
    found = cable_endpoints()
    if ONE not in found or TWO not in found:
        pytest.skip(NOT_INSTALLED)
    return found


def peer(*args, timeout=30):
    return subprocess.run([sys.executable, str(PEER), *map(str, args)], capture_output=True, text=True,
                          timeout=timeout, check=True)


def spawn(*args):
    return subprocess.Popen([sys.executable, str(PEER), *map(str, args)])


def report(what, heard, sent_rms):
    print(f"\n{what}: {dominant_frequency(heard):.2f} Hz, rms {rms(heard):.5f} (expected {sent_rms:.5f}, "
          f"{20 * np.log10(rms(heard) / sent_rms):+.3f} dB)")


def settled(captured, seconds=0.5):
    loud = np.nonzero(np.abs(captured[:, 0]) > 1e-5)[0]
    assert len(loud) > RATE // 4, "nothing crossed the cable"
    return captured[loud[0] + RATE // 10: loud[0] + RATE // 10 + int(RATE * seconds)]


def through_peers(cable: str, tmp_path, name='heard.npy') -> np.ndarray:
    out = tmp_path / name
    recorder = spawn('record', f'({cable})', 3.0, out)
    time.sleep(0.5)
    peer('play', f'({cable})', 2.0, 1000, 0.2)
    recorder.wait(timeout=30)
    return np.load(out)


def test_a_fresh_install_lists_two_cables_each_with_its_own_endpoints(cables):
    from tonesphere.engine import virtual_cables

    listed = virtual_cables.cables()
    print("\n" + "\n".join(f"{c.instance_id}: {c.name!r} {c.state}" for c in listed))
    assert [c.name for c in listed] == [ONE, TWO]
    assert all(c.state == 'working' for c in listed)
    for name in (ONE, TWO):
        render, capture = cables[name]['render'], cables[name]['capture']
        assert render.name == f'Speakers ({name})' and capture.name == f'Microphone Array ({name})'
        assert render.mix_sample_rate == capture.mix_sample_rate == 48000
        assert render.mix_channels == capture.mix_channels == 2


@pytest.mark.parametrize('cable', [ONE, TWO])
def test_one_application_to_another_through_each_cable(cables, tmp_path, cable):
    """Two PortAudio processes, neither of them ToneSphere: the cable alone carries the tone."""
    heard = settled(through_peers(cable, tmp_path))
    report(f"PortAudio -> {cable} -> PortAudio", heard, TONE_RMS)
    assert dominant_frequency(heard) == pytest.approx(1000.0, abs=3.0)
    assert rms(heard) == pytest.approx(TONE_RMS, rel=0.05)


def test_the_cables_are_isolated_from_each_other(cables, tmp_path):
    """A tone played into cable 1 is heard on cable 1 and is exact silence on cable 2."""
    one, two = tmp_path / 'one.npy', tmp_path / 'two.npy'
    recorders = [spawn('record', f'({ONE})', 3.0, one), spawn('record', f'({TWO})', 3.0, two)]
    time.sleep(0.5)
    peer('play', f'({ONE})', 2.0, 1000, 0.2)
    for r in recorders:
        r.wait(timeout=30)
    heard_one, heard_two = np.load(one), np.load(two)
    print(f"\ninto {ONE}: {ONE} rms {rms(settled(heard_one)):.5f}, {TWO} peak {np.max(np.abs(heard_two)):.1e}")
    assert rms(settled(heard_one)) == pytest.approx(TONE_RMS, rel=0.05)
    assert len(heard_two) > RATE and np.max(np.abs(heard_two)) == 0.0, "audio leaked into the other cable"


def test_an_application_into_tonesphere(cables):
    """Another process plays into cable 1; ToneSphere's native engine captures its other end."""
    with NativeEngine(RATE, 480) as engine:
        engine.apply_plan([Node.source(1, 2), Node.sink(2, 2, ring_frames=RATE * 6)], [Route(1, 2)])
        engine.start_wasapi([StreamSpec(1, 'capture', 2, cables[ONE]['capture'].id)])
        player = spawn('play', f'({ONE})', 2.0, 1000, 0.2)
        player.wait(timeout=30)
        time.sleep(0.3)
        heard = engine.port_read(2, RATE * 6)
        engine.stop_backend()
    heard = settled(heard)
    report(f"PortAudio -> {ONE} -> ToneSphere", heard, TONE_RMS)
    assert dominant_frequency(heard) == pytest.approx(1000.0, abs=3.0)
    assert rms(heard) == pytest.approx(TONE_RMS, rel=0.05)


def test_tonesphere_through_a_plugin_into_another_application(cables, tmp_path):
    """ToneSphere renders a tone through its VST3 test plugin (x0.5) into cable 2; another process records."""
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
        recorder = spawn('record', f'({TWO})', 2.5, out)
        time.sleep(0.3)
        engine.start_wasapi([StreamSpec(2, 'render', 2, cables[TWO]['render'].id, period_frames=480)])
        recorder.wait(timeout=30)
        engine.stop_backend()
    heard = settled(np.load(out))
    report(f"ToneSphere -> Test Gain x0.5 -> {TWO} -> PortAudio", heard, TONE_RMS * 0.5)
    assert dominant_frequency(heard) == pytest.approx(1000.0, abs=3.0)
    assert rms(heard) == pytest.approx(TONE_RMS * 0.5, rel=0.05), "the plugin's gain must cross the cable"


def test_nothing_rendered_is_silence(cables):
    with NativeEngine(RATE, 480) as engine:
        engine.apply_plan([Node.source(1, 2), Node.sink(2, 2, ring_frames=RATE * 2)], [Route(1, 2)])
        engine.start_wasapi([StreamSpec(1, 'capture', 2, cables[ONE]['capture'].id)])
        time.sleep(1.0)
        heard = engine.port_read(2, RATE * 2)
        engine.stop_backend()
    assert len(heard) > RATE // 2 and np.max(np.abs(heard)) == 0.0, "an idle cable must be exact silence"


def test_a_new_capture_does_not_hear_what_was_played_before_it(cables):
    """Up to 100 ms can queue in a cable; a recording that starts later must begin empty."""
    peer('play', f'({ONE})', 0.5, 1000, 0.2)
    with NativeEngine(RATE, 480) as engine:
        engine.apply_plan([Node.source(1, 2), Node.sink(2, 2, ring_frames=RATE * 2)], [Route(1, 2)])
        engine.start_wasapi([StreamSpec(1, 'capture', 2, cables[ONE]['capture'].id)])
        time.sleep(0.5)
        heard = engine.port_read(2, RATE * 2)
        engine.stop_backend()
    assert len(heard) > RATE // 4 and np.max(np.abs(heard)) == 0.0, "old audio was replayed into a new capture"


@pytest.fixture
def engine():
    from tonesphere.core.engine import AudioEngine

    e = AudioEngine(sample_rate=RATE, buffer_size=480, exclusive=False)
    e.initialize()
    yield e
    e.cleanup()


def test_the_engine_reports_every_cable_with_its_endpoints(cables, engine):
    """What the Diagnostics view and the Virtual Cables dialog show."""
    status = engine.virtual_device_status()
    assert status['installed'], status
    listed = {c['name']: c for c in status['cables']}
    for name in (ONE, TWO):
        assert listed[name]['state'] == 'working'
        assert listed[name]['render']['name'] == f'Speakers ({name})'
        assert listed[name]['capture']['name'] == f'Microphone Array ({name})'


def wait_for(predicate, seconds=15.0):
    deadline = time.time() + seconds
    while time.time() < deadline:
        if predicate():
            return True
        time.sleep(0.25)
    return predicate()


def test_a_cable_added_renamed_and_uninstalled_through_the_app(cables, engine, tmp_path):
    """The Virtual Cables dialog's own calls: each change a single device, the others untouched."""
    from tonesphere.engine import virtual_cables

    ok, message = engine.manage_virtual_cable('add', 'Chat')
    assert ok, message
    assert wait_for(lambda: 'Chat' in cable_endpoints()), cable_endpoints().keys()
    heard = settled(through_peers('Chat', tmp_path, 'chat.npy'))
    assert rms(heard) == pytest.approx(TONE_RMS, rel=0.05)

    chat = next(c for c in virtual_cables.cables() if c.name == 'Chat')
    ok, message = engine.manage_virtual_cable('rename', chat.instance_id, 'Game')
    assert ok, message
    assert wait_for(lambda: 'Game' in cable_endpoints() and 'Chat' not in cable_endpoints())

    ok, message = engine.manage_virtual_cable('remove', chat.instance_id)
    assert ok, message
    assert wait_for(lambda: 'Game' not in cable_endpoints())
    assert [c.name for c in virtual_cables.cables()] == [ONE, TWO]
    heard = settled(through_peers(ONE, tmp_path, 'after.npy'))
    assert rms(heard) == pytest.approx(TONE_RMS, rel=0.05), "uninstalling one cable disturbed another"
    print(f"\nadded 'Chat' ({chat.instance_id}), renamed it 'Game', uninstalled it; {ONE} still carries audio")


def test_a_cable_disabled_mid_stream_is_reopened_when_it_returns(cables, engine, tmp_path):
    """
    ToneSphere captures cable 2 into a bus sent to a network ring, and plays a tone into
    cable 1. Cable 2 is disabled: its stream stops, the tone into cable 1 carries on, and the
    engine says what left. Cable 2 is enabled again: the engine reopens it by itself, and
    what another program then plays into cable 2 reaches ToneSphere again.
    """
    from tests.native.test_engine_native_host import collect, network_sink
    from tonesphere.engine import virtual_cables

    devices = engine.get_devices()

    def device_id(name, direction):
        return next(d['id'] for d in devices if d['name'] == name and d['direction'] == direction)

    assert engine.start_udp_transport('127.0.0.1', 0)[0]
    listen = device_id(f'Microphone Array ({TWO})', 'input')
    heard_bus = engine.create_virtual_input('heard', channels=2)
    tone_bus = engine.create_virtual_input('tone', channels=2)
    assert engine.create_routing(listen, heard_bus)[0]
    assert engine.create_routing(tone_bus, device_id(f'Speakers ({ONE})', 'output'))[0]
    source, dest = network_sink(engine, heard_bus)
    engine.start_engine()
    time.sleep(0.5)

    def captured_tone(seconds=2.0):
        # What piled up in the ring while nothing read it (a restart's worth of silence) is
        # drained first, and the window leaves room for a player starting on a device that
        # has only just come back.
        collect(engine, source, dest, 0.3)
        player = spawn('play', f'({TWO})', seconds, 1000, 0.2)
        audio = collect(engine, source, dest, seconds + 2.0)
        player.wait(timeout=30)
        loud = np.nonzero(np.abs(audio[:, 0]) > 1e-5)[0] if len(audio) else []
        print(f"\ncaptured {len(audio) / RATE:.2f} s, the tone from {loud[0] / RATE if len(loud) else None} s")
        return audio

    before = captured_tone()
    assert rms(settled(before)) == pytest.approx(TONE_RMS, rel=0.05)

    two = next(c for c in virtual_cables.cables() if c.name == TWO)
    ok, message = engine.manage_virtual_cable('disable', two.instance_id)
    assert ok, message
    assert wait_for(lambda: f'Microphone Array ({TWO})' in ((engine.last_device_change() or {}).get('left') or [])), \
        f"the engine did not see it go: {engine.last_device_change()}"
    change = engine.last_device_change()
    print(f"\ndisabled {TWO}: {change}")
    assert engine.host.is_running, "the engine did not start again without the cable"

    # The other cable plays on while cable 2 is gone.
    one_out = tmp_path / 'one.npy'
    recorder = spawn('record', f'({ONE})', 2.0, one_out)
    # Fed at the rate it plays, 100 ms ahead: faster, the bus's ring would drop what it
    # cannot hold and splice the tone.
    written = 0
    tone = sine(RATE * 4, 440.0, amplitude=0.2)
    start = time.time()
    while time.time() - start < 2.5:
        due = min(len(tone), int((time.time() - start + 0.1) * RATE))
        if due > written:
            written += engine.write_to_bus(tone_bus, tone[written:due])
        time.sleep(0.02)
    recorder.wait(timeout=30)
    assert dominant_frequency(settled(np.load(one_out), 1.0)) == pytest.approx(440.0, abs=2.0)

    ok, message = engine.manage_virtual_cable('enable', two.instance_id)
    assert ok, message
    assert wait_for(lambda: f'Microphone Array ({TWO})' in ((engine.last_device_change() or {}).get('arrived') or []))
    change = engine.last_device_change()
    print(f"enabled {TWO}: {change}")
    assert change['reopened'] is True
    time.sleep(1.0)
    after = captured_tone()
    heard = settled(after)
    report(f"after {TWO} returned, PortAudio -> {TWO} -> ToneSphere (reopened with no user action)", heard, TONE_RMS)
    assert dominant_frequency(heard) == pytest.approx(1000.0, abs=3.0)
    assert rms(heard) == pytest.approx(TONE_RMS, rel=0.05)


# A second program running ToneSphere's native engine, capturing a cable until its stdin
# closes. Not killed: a venv's python.exe is a launcher, and killing it leaves the real
# interpreter — and its stream — running.
HOLDER = """
import sys, time
from tonesphere.native import NativeEngine, Node, Route, wasapi
mic = next(e for e in wasapi.endpoints() if e.flow == 'capture' and e.name.endswith(sys.argv[1]))
with NativeEngine(48000, 480) as engine:
    engine.apply_plan([Node.source(1, 2), Node.sink(2, 2, ring_frames=48000)], [Route(1, 2)])
    engine.start_wasapi([wasapi.StreamSpec(1, 'capture', 2, mic.id)])
    print('holding', flush=True)
    sys.stdin.readline()
"""


def test_a_cable_in_use_elsewhere_is_disabled_or_refused_cleanly_never_left_pending_a_restart(cables, engine,
                                                                                              tmp_path):
    """
    What Windows does when another program has the cable open depends on how it opened it.
    A PortAudio recorder (plain shared mode) does not stop it: the cable is disabled under
    it. A second program running ToneSphere's native engine does — Windows' audio engine
    (audiodg.exe) vetoes the removal — and the change must then come back refused, the cable
    still working and still changeable, not "pending a restart" (what a plain pnputil disable
    leaves behind); and go through once that program lets go.
    """
    from tonesphere.engine import virtual_cables

    two = next(c for c in virtual_cables.cables() if c.name == TWO)

    recorder = spawn('record', f'({TWO})', 5.0, tmp_path / 'held.npy')
    try:
        time.sleep(1.5)
        ok, message = engine.manage_virtual_cable('disable', two.instance_id)
        assert ok, message
        assert next(c for c in virtual_cables.cables() if c.name == TWO).state == 'disabled'
    finally:
        recorder.wait(timeout=60)
    ok, message = engine.manage_virtual_cable('enable', two.instance_id)
    assert ok, message
    assert wait_for(lambda: TWO in cable_endpoints())

    holder = subprocess.Popen([sys.executable, '-c', HOLDER, f'({TWO})'], cwd=ROOT, stdin=subprocess.PIPE,
                              stdout=subprocess.PIPE, text=True)
    try:
        assert holder.stdout.readline().strip() == 'holding'
        ok, refused = engine.manage_virtual_cable('disable', two.instance_id)
        assert not ok and 'has this cable open' in refused, refused
        assert next(c for c in virtual_cables.cables() if c.name == TWO).state == 'working'
        assert TWO in cable_endpoints()
    finally:
        holder.stdin.close()
        holder.wait(timeout=30)
    ok, message = engine.manage_virtual_cable('disable', two.instance_id)
    assert ok, f"once the program let go: {message}"
    assert next(c for c in virtual_cables.cables() if c.name == TWO).state == 'disabled'
    ok, message = engine.manage_virtual_cable('enable', two.instance_id)
    assert ok, message
    assert wait_for(lambda: TWO in cable_endpoints())
    print(f"\nwhile a PortAudio program recorded {TWO}: disabled under it. While another native-engine program "
          f"held it: refused ({refused}); disabled and enabled once it let go")


def test_the_virtual_cables_dialog_disables_and_enables_a_real_cable(cables, engine, monkeypatch):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication

    from tonesphere.core.engine_factory import UnifiedAudioEngine
    from tonesphere.ui.cables_view import CablesDialog

    app = QApplication.instance() or QApplication([])
    unified = UnifiedAudioEngine.__new__(UnifiedAudioEngine)
    unified.engine = engine
    dialog = CablesDialog(unified)
    try:
        assert dialog.tasks.flush()
        names = [dialog.table.item(r, 0).text() for r in range(dialog.table.rowCount())]
        assert names == [ONE, TWO]
        dialog.table.selectRow(1)
        dialog.toggle_selected()
        assert dialog.tasks.flush(60) and dialog.tasks.flush(60)
        assert dialog.table.item(1, 1).text() == 'disabled', dialog.status.text()
        assert f'Speakers ({TWO})' not in [e.name for e in endpoints()]
        artifacts = os.environ.get('TONESPHERE_ARTIFACTS')
        if artifacts:
            dialog.resize(860, 380)
            dialog.show()
            app.processEvents()
            dialog.grab().save(str(Path(artifacts) / 'virtual_cables_dialog.png'))
        dialog.table.selectRow(1)
        dialog.toggle_selected()
        assert dialog.tasks.flush(60) and dialog.tasks.flush(60)
        assert dialog.table.item(1, 1).text() == 'working', dialog.status.text()
        assert wait_for(lambda: f'Speakers ({TWO})' in [e.name for e in endpoints()])
    finally:
        dialog.done(0)


def test_ffmpeg_records_what_tonesphere_plays_into_a_cable(cables, tmp_path):
    """A third program, not PortAudio: ffmpeg's DirectShow capture of cable 1's input."""
    ffmpeg = os.environ.get('TONESPHERE_FFMPEG')
    if not ffmpeg or not Path(ffmpeg).is_file():
        pytest.skip("ffmpeg not provided (TONESPHERE_FFMPEG)")
    wav = tmp_path / 'ffmpeg.wav'
    # Long enough to outlast ffmpeg's start: building a DirectShow graph takes seconds in a
    # non-interactive session.
    with NativeEngine(RATE, 480) as engine:
        engine.apply_plan([Node.source(1, 2, ring_frames=RATE * 40), Node.sink(2, 2)], [Route(1, 2)])
        engine.port_write(1, sine(RATE * 38, 1000.0, amplitude=0.2))
        engine.start_wasapi([StreamSpec(2, 'render', 2, cables[ONE]['render'].id)])
        started = time.time()
        subprocess.run([ffmpeg, '-hide_banner', '-loglevel', 'error', '-f', 'dshow', '-i',
                        f'audio=Microphone Array ({ONE})', '-t', '3', '-ar', '48000', '-ac', '2', '-y', str(wav)],
                       check=True, timeout=30)
        took = time.time() - started
        engine.stop_backend()
    print(f"\nffmpeg took {took:.1f} s to record 3 s")
    import wave

    with wave.open(str(wav)) as f:
        data = np.frombuffer(f.readframes(f.getnframes()), dtype=np.int16).reshape(-1, 2) / 32768.0
    heard = settled(data.astype(np.float32))
    report(f"ToneSphere -> {ONE} -> ffmpeg (DirectShow)", heard, TONE_RMS)
    assert dominant_frequency(heard) == pytest.approx(1000.0, abs=3.0)
    assert rms(heard) == pytest.approx(TONE_RMS, rel=0.05)
