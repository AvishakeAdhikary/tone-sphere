"""
The native ASIO host against whatever ASIO drivers this machine has.

With no driver registered, every test here skips with
"ASIO HARDWARE VERIFICATION: NOT AVAILABLE ON THIS MACHINE" — the host is built and its
boundary tested (tests/native/test_asio.py), but nothing claims it has driven a driver.

Output content can be verified only when the driver renders through WASAPI inside this
process (FlexASIO does): then this process's own loopback captures what the ASIO host
played. A hardware interface's driver bypasses WASAPI entirely, so for it the output is
proven only up to the driver's buffers, and hearing it needs a loopback cable (the M6
round-trip tool).
"""

import os
import time

import numpy as np
import pytest

from tests.signals import dominant_frequency, rms, sine
from tonesphere.native import NativeEngine, NativeError, Node, Route, asio
from tonesphere.native import available as native_available
from tonesphere.native.wasapi import StreamSpec, default_endpoint

NOT_AVAILABLE = "ASIO HARDWARE VERIFICATION: NOT AVAILABLE ON THIS MACHINE"

pytestmark = [
    pytest.mark.hardware,
    pytest.mark.skipif(not native_available() or not asio.available(), reason="needs tonesphere_asio.dll"),
]

TONE, ASIO_IN, ASIO_OUT, CAPTURED, LOOP, BACK, CLOCK = 1, 10, 20, 30, 40, 50, 60


def working_drivers():
    registered = asio.drivers()
    if not registered:
        pytest.skip(f"{NOT_AVAILABLE} (no ASIO driver is registered under HKLM\\SOFTWARE\\ASIO)")
    present = [d for d in registered if d.dll_present]
    if not present:
        pytest.skip(f"{NOT_AVAILABLE} (drivers are registered but none has its DLL installed)")
    return present


@pytest.fixture(params=([d.name for d in asio.drivers() if d.dll_present] if asio.available() else []) or [None])
def driver(request):
    if request.param is None:
        working_drivers()
    return request.param


def test_drivers_are_registered_and_load():
    for d in working_drivers():
        info = asio.query(d.name)
        assert len(info.inputs) + len(info.outputs) > 0, f"{d.name} reports no channels"
        assert info.min_buffer <= info.preferred_buffer <= info.max_buffer
        assert info.sample_rates, f"{d.name} accepts none of {asio.RATES}"
        print(f"\n{d.name}: '{info.name}' v{info.version}, {len(info.inputs)} in / {len(info.outputs)} out, "
              f"buffer {info.min_buffer}-{info.max_buffer} (preferred {info.preferred_buffer}, granularity "
              f"{info.granularity}), rates {info.sample_rates}, reported latency {info.input_latency_frames}/"
              f"{info.output_latency_frames} frames, outputReady={info.post_output}, "
              f"types in {[c.sample_type for c in info.inputs[:2]]} out {[c.sample_type for c in info.outputs[:2]]}")


def test_the_buffer_switch_runs_the_engine(driver):
    info = asio.query(driver)
    rate = 48000 if 48000 in info.sample_rates else info.sample_rates[0]
    ins = tuple(range(min(2, len(info.inputs))))
    outs = tuple(range(min(2, len(info.outputs))))
    block = max(info.preferred_buffer, 64)
    tone = sine(rate * 3, 1000.0, rate=rate, amplitude=0.05, channels=max(1, len(outs)))

    with NativeEngine(rate, block) as engine:
        nodes, routes = [], []
        if outs:
            nodes += [Node.source(TONE, len(outs), ring_frames=rate * 4), Node.sink(ASIO_OUT, len(outs))]
            routes.append(Route(TONE, ASIO_OUT))
        if ins:
            nodes += [Node.source(ASIO_IN, len(ins)), Node.sink(CAPTURED, len(ins), ring_frames=rate * 4)]
            routes.append(Route(ASIO_IN, CAPTURED))
        engine.apply_plan(nodes, routes)
        if outs:
            engine.port_write(TONE, tone)
        asio.start(engine, driver, input_node=ASIO_IN if ins else 0, inputs=ins,
                   output_node=ASIO_OUT if outs else 0, outputs=outs, buffer_frames=info.preferred_buffer)
        time.sleep(1.5)
        status = engine.stream_status()
        stats = engine.stats()
        captured = engine.port_read(CAPTURED, rate * 4) if ins else None
        engine.stop_backend()

    assert all(s['state'] == 'running' for s in status), status
    assert stats['blocks'] > 0.8 * 1.5 * rate / block, f"the buffer switch ran only {stats['blocks']} blocks"
    assert stats['rt_allocations'] == 0
    if ins:
        assert len(captured) > 0.8 * 1.5 * rate, "inputs delivered no frames"
    print(f"\n{driver}: {stats['blocks']} buffer switches of {status[0]['buffer_frames']} frames at {rate} Hz, "
          f"callback mean {stats['callback_ns_mean'] / 1000:.1f} us max {stats['callback_ns_max'] / 1000:.1f} us, "
          f"load {stats['processing_load']:.1%}, xruns {stats['xruns']}, reported latency "
          f"{[s['reported_latency_ms'] for s in status]} ms, {status[0]['message']}")
    if ins:
        # What the inputs heard, for the record: an instrument's noise floor or a microphone's
        # room, never asserted, since neither is ToneSphere's to guarantee.
        body = captured[len(captured) // 3:].astype(np.float64)
        for c in range(body.shape[1]):
            level = rms(body[:, c])
            print(f"  input {c + 1}: {20 * np.log10(max(level, 1e-12)):.1f} dBFS rms, dominant "
                  f"{dominant_frequency(body[:, c], rate):.2f} Hz")


def test_output_content_where_the_driver_renders_through_wasapi(driver):
    """
    Two engines: one plays a tone through the ASIO driver, the other captures this
    process's audio with process loopback. If the driver renders through WASAPI in this
    process, the tone must come back at the level sent; if not, there is nothing to capture
    and the test says so instead of passing.
    """
    info = asio.query(driver)
    if len(info.outputs) < 2:
        pytest.skip("needs a stereo output")
    rate = 48000 if 48000 in info.sample_rates else info.sample_rates[0]
    clock = default_endpoint('render')
    tone = sine(rate * 3, 1000.0, rate=rate, amplitude=0.05)
    with NativeEngine(rate, max(info.preferred_buffer, 64)) as player, NativeEngine(rate, 480) as listener:
        player.apply_plan([Node.source(TONE, 2, ring_frames=rate * 4), Node.sink(ASIO_OUT, 2)], [Route(TONE, ASIO_OUT)])
        player.port_write(TONE, tone)
        listener.apply_plan([Node.sink(CLOCK, 2), Node.source(LOOP, 2), Node.sink(BACK, 2, ring_frames=rate * 6)],
                            [Route(LOOP, BACK)])
        asio.start(player, driver, output_node=ASIO_OUT, outputs=(0, 1), buffer_frames=info.preferred_buffer)
        try:
            listener.start_wasapi([StreamSpec(CLOCK, 'render', 2, clock.id),
                                   StreamSpec(LOOP, 'process_loopback', 2, process_id=os.getpid())])
        except NativeError as e:
            player.stop_backend()
            if 'exclusive mode' not in str(e):
                raise
            # A driver holding the device exclusively plays past the Windows mixer, where no
            # loopback can hear it.
            pytest.skip(f"{driver} holds the default output exclusively; its output needs a loopback cable")
        time.sleep(2.0)
        back = listener.port_read(BACK, rate * 6)
        listener.stop_backend()
        player.stop_backend()
    loud = np.nonzero(np.abs(back[:, 0]) > 1e-5)[0]
    if len(loud) < rate // 2:
        pytest.skip(f"{driver} does not render through WASAPI in this process; its output needs a loopback cable")
    heard = back[loud[0] + rate // 10: loud[0] + rate // 10 + rate // 2]
    assert dominant_frequency(heard) == pytest.approx(1000.0, abs=2.0)
    print(f"\n{driver} output through process loopback: rms {rms(heard):.5f} (sent {rms(tone):.5f})")
