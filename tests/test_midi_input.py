"""
MIDI input ports (winmm). With a port, its notes go to an instrument's native MIDI queue;
without one — the development machine has none — what can be checked is that enumeration
works and that every refusal says why. Playing from a hardware keyboard is UNVERIFIED.
"""

import sys

import pytest

from tonesphere.engine import midi_input

pytestmark = pytest.mark.skipif(sys.platform != 'win32', reason="MIDI input is implemented on Windows only")


def test_windows_lists_its_midi_input_ports():
    ports = midi_input.inputs()
    assert isinstance(ports, list) and all(isinstance(name, str) and name for name in ports)
    print(f"\nMIDI input ports: {ports or 'none'}")


def test_a_port_needs_an_instrument_to_play_and_a_port_to_exist():
    from tonesphere.core.engine import AudioEngine

    engine = AudioEngine()
    engine.initialize()
    try:
        bus = engine.create_virtual_input('no instrument')
        ok, message = engine.connect_midi_input(0, bus)
        assert not ok and 'No instrument' in message
        if not midi_input.inputs():
            with pytest.raises(OSError, match='midiInOpen'):
                midi_input.MidiInput(0, lambda *m: None)
    finally:
        engine.cleanup()
