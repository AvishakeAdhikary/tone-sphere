"""
Virtual bus behaviour.

Several of these are regression tests for bugs where the code reported success and
did nothing. They assert on observed audio, never on a return value alone.
"""

import time

import numpy as np
import pytest

from tonesphere.devices.native_virtual import NativeVirtualDevice
from tonesphere.devices.virtual_device_manager import VirtualDeviceManager

BUFFER_SIZE = 128
SAMPLE_RATE = 48000


def wait_for_frame(device, timeout=2.0):
    """Wait for a non-silent frame, returning its peak (0.0 if none arrived)."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        peak = float(np.max(np.abs(device.current_frame)))
        if peak > 0.0:
            return peak
        time.sleep(0.005)
    return 0.0


@pytest.fixture
def manager():
    mgr = VirtualDeviceManager(SAMPLE_RATE, BUFFER_SIZE, max_inputs=4, max_outputs=4)
    yield mgr
    mgr.clear_all()


def test_input_device_receives_written_audio(manager):
    device_id = manager.create_input(channels=2)
    device = manager.get_device(device_id)

    device.write_audio(np.full((BUFFER_SIZE, 2), 0.5, dtype=np.float32))

    assert wait_for_frame(device) == pytest.approx(0.5, abs=1e-6)


def test_output_device_receives_written_audio(manager):
    """
    Regression: write_audio() fed `input_buffer` while the output-device loop only
    read `output_buffer`, so audio written to any output was silently dropped.
    """
    device_id = manager.create_output(channels=2)
    device = manager.get_device(device_id)

    device.write_audio(np.full((BUFFER_SIZE, 2), 0.5, dtype=np.float32))

    assert wait_for_frame(device) == pytest.approx(0.5, abs=1e-6)


def test_routing_carries_audio_between_buses(manager):
    """
    Regression: the engine created devices in one registry and routed them through a
    second, empty one. create_routing() returned success and no audio ever moved.
    """
    source_id = manager.create_input(channels=2)
    dest_id = manager.create_output(channels=2)

    assert manager.route_audio(source_id, dest_id, volume=1.0) is True

    source = manager.get_device(source_id)
    dest = manager.get_device(dest_id)

    source.write_audio(np.full((BUFFER_SIZE, 2), 0.5, dtype=np.float32))
    wait_for_frame(source)
    manager.process_routing()

    assert wait_for_frame(dest) == pytest.approx(0.5, abs=1e-6)


def test_routing_applies_volume(manager):
    source_id = manager.create_input(channels=2)
    dest_id = manager.create_output(channels=2)
    manager.route_audio(source_id, dest_id, volume=0.25)

    source = manager.get_device(source_id)
    dest = manager.get_device(dest_id)

    source.write_audio(np.full((BUFFER_SIZE, 2), 0.8, dtype=np.float32))
    wait_for_frame(source)
    manager.process_routing()

    assert wait_for_frame(dest) == pytest.approx(0.2, abs=1e-5)


def test_route_to_unknown_device_reports_failure(manager):
    """A route we cannot honour must say so rather than silently succeeding."""
    source_id = manager.create_input(channels=2)

    assert manager.route_audio(source_id, 999999, volume=1.0) is False


def test_unroute_stops_audio(manager):
    source_id = manager.create_input(channels=2)
    dest_id = manager.create_output(channels=2)
    manager.route_audio(source_id, dest_id)

    assert manager.unroute_audio(source_id, dest_id) is True
    assert manager.get_device(source_id).connected_devices == {}


def test_update_channels_does_not_raise(manager):
    """Regression: referenced device.is_running and device.buffer, neither of which existed."""
    device_id = manager.create_input(channels=2)

    assert manager.update_device_channels(device_id, 1) is True

    device = manager.get_device(device_id)
    assert device.channels == 1
    assert device.current_frame.shape == (BUFFER_SIZE, 1)
    assert device.is_running is True  # restarted after the change


def test_update_sample_rate_does_not_raise(manager):
    """Regression: same missing is_running attribute."""
    device_id = manager.create_input(channels=2)

    assert manager.update_device_sample_rate(device_id, 44100) is True
    assert manager.get_device(device_id).sample_rate == 44100


def test_update_channels_rejects_zero(manager):
    device_id = manager.create_input(channels=2)

    assert manager.update_device_channels(device_id, 0) is False
    assert manager.get_device(device_id).channels == 2


def test_channel_mismatch_does_not_corrupt_destination(manager):
    """A mono source must not be broadcast into a stereo bus as a shape error."""
    source_id = manager.create_input(channels=1)
    dest_id = manager.create_output(channels=2)
    manager.route_audio(source_id, dest_id)

    source = manager.get_device(source_id)
    source.write_audio(np.full((BUFFER_SIZE, 1), 0.5, dtype=np.float32))
    wait_for_frame(source)
    manager.process_routing()

    dest = manager.get_device(dest_id)
    assert dest.current_frame.shape == (BUFFER_SIZE, 2)


def test_device_limits_are_enforced(manager):
    for _ in range(4):
        assert manager.create_input(channels=2) is not None

    assert manager.can_create_input() is False
    assert manager.create_input(channels=2) is None


def test_overrun_drops_oldest_frame_and_counts_it():
    device = NativeVirtualDevice(1, "test", 2, SAMPLE_RATE, BUFFER_SIZE, is_input=True)

    # Not started, so nothing drains the queue (maxsize 10).
    for _ in range(15):
        device.write_audio(np.zeros((BUFFER_SIZE, 2), dtype=np.float32))

    assert device.buffer_overruns > 0
    assert device.frame_queue.qsize() <= 10


def test_stop_drains_queued_frames():
    device = NativeVirtualDevice(1, "test", 2, SAMPLE_RATE, BUFFER_SIZE, is_input=True)
    device.start()
    device.write_audio(np.zeros((BUFFER_SIZE, 2), dtype=np.float32))
    device.stop()

    assert device.frame_queue.qsize() == 0
    assert device.is_running is False
