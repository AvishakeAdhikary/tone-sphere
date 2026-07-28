"""
Pan, polarity, channel strips and the limiter as they run inside the real callback.

The unit tests in test_dsp.py cover each processor on its own. These drive `_mix_into`,
which is verbatim the code the audio thread executes, so they cover the wiring too.
"""

import math

import numpy as np
import pytest

from tonesphere.engine.devices import DeviceInfo, HostApi
from tonesphere.engine.graph import Connection, RoutingGraph, bus_node
from tonesphere.engine.host import AudioHost, StreamConfig, _DeviceStream

BLOCK = 256
RATE = 48000


def sine(frames, freq=1000.0, amplitude=0.5, channels=2, phase=0.0):
    t = (np.arange(frames, dtype=np.float64) + phase) / RATE
    wave = (amplitude * np.sin(2.0 * math.pi * freq * t)).astype(np.float32)
    return np.repeat(wave.reshape(-1, 1), channels, axis=1)


def fake_device(name="test", inputs=0, outputs=2):
    return DeviceInfo(
        index=0, name=name, host_api=HostApi.UNKNOWN, host_api_name='test',
        max_input_channels=inputs, max_output_channels=outputs,
        default_samplerate=RATE,
        default_low_input_latency_ms=0.0, default_low_output_latency_ms=0.0,
        default_high_input_latency_ms=0.0, default_high_output_latency_ms=0.0,
    )


def sink_stream(out_channels=2, node=None):
    stream = _DeviceStream(StreamConfig(
        device=fake_device(outputs=out_channels),
        samplerate=RATE, blocksize=BLOCK, output_channels=out_channels,
    ))
    stream.node = node if node is not None else bus_node('sink')
    return stream


class TestPanInTheMixer:
    def make_host(self):
        host = AudioHost(samplerate=RATE, blocksize=BLOCK)
        host.create_bus('mono', channels=1)
        host.create_bus('stereo', channels=2)
        return host

    def render(self, host, graph, sources, blocks=3, out_channels=2):
        """Run several blocks so smoothed parameters have settled before asserting."""
        host.graph_holder.commit(graph)
        host._rebuild_routes()

        stream = sink_stream(out_channels)
        out = np.zeros((BLOCK, out_channels), dtype=np.float32)

        for _ in range(blocks):
            for name, block in sources.items():
                host.write_bus(name, block)
            host._mix_into(stream, out, BLOCK)

        return out

    def test_mono_source_is_centred_with_constant_power(self):
        """
        A centred mono source sits at -3 dB per side so its total power matches a
        hard-panned one. Duplicating at unity would make centre 3 dB louder.
        """
        host = self.make_host()
        out = self.render(
            host,
            RoutingGraph(connections=(Connection(bus_node('mono'), bus_node('sink')),)),
            {'mono': np.ones((BLOCK, 1), dtype=np.float32)},
        )

        expected = 1.0 / math.sqrt(2.0)
        assert float(out[BLOCK // 2, 0]) == pytest.approx(expected, abs=1e-3)
        assert float(out[BLOCK // 2, 1]) == pytest.approx(expected, abs=1e-3)

    def test_hard_left_route_puts_nothing_in_the_right(self):
        host = self.make_host()
        out = self.render(
            host,
            RoutingGraph(connections=(
                Connection(bus_node('mono'), bus_node('sink'), pan=-1.0),
            )),
            {'mono': np.ones((BLOCK, 1), dtype=np.float32)},
        )

        assert float(out[BLOCK // 2, 0]) == pytest.approx(1.0, abs=1e-2)
        assert abs(float(out[BLOCK // 2, 1])) < 1e-2

    def test_two_routes_from_one_source_pan_independently(self):
        """
        Why pan lives on the route: the same guitar centre in the headphone mix and hard
        left in the recording feed.
        """
        host = self.make_host()
        host.graph_holder.commit(RoutingGraph(connections=(
            Connection(bus_node('mono'), bus_node('phones'), pan=0.0),
            Connection(bus_node('mono'), bus_node('rec'), pan=-1.0),
        )))
        host._rebuild_routes()

        phones = sink_stream(2, node=bus_node('phones'))
        rec = sink_stream(2, node=bus_node('rec'))
        out_phones = np.zeros((BLOCK, 2), dtype=np.float32)
        out_rec = np.zeros((BLOCK, 2), dtype=np.float32)

        for _ in range(3):
            host.write_bus('mono', np.ones((BLOCK, 1), dtype=np.float32))
            host._mix_into(phones, out_phones, BLOCK)
            host._mix_into(rec, out_rec, BLOCK)

        assert float(out_phones[BLOCK // 2, 1]) == pytest.approx(1 / math.sqrt(2), abs=1e-2)
        assert abs(float(out_rec[BLOCK // 2, 1])) < 1e-2

    def test_polarity_inversion_cancels_against_the_original(self):
        host = self.make_host()
        host.create_bus('a', channels=2)
        host.create_bus('b', channels=2)

        tone = sine(BLOCK, amplitude=0.5)
        out = self.render(
            host,
            RoutingGraph(connections=(
                Connection(bus_node('a'), bus_node('sink')),
                Connection(bus_node('b'), bus_node('sink'), invert=True),
            )),
            {'a': tone, 'b': tone},
        )

        assert float(np.max(np.abs(out))) < 1e-4


class TestLimiterInTheMixer:
    def test_hot_sum_is_kept_inside_full_scale(self):
        """
        Four sources at 0.9 sum to 3.6. Harmless in float, but the device's integer
        converter would clip it into broadband distortion.
        """
        host = AudioHost(samplerate=RATE, blocksize=BLOCK)
        names = ('s1', 's2', 's3', 's4')
        for name in names:
            host.create_bus(name, channels=2)

        host.graph_holder.commit(RoutingGraph(connections=tuple(
            Connection(bus_node(name), bus_node('sink')) for name in names
        )))
        host._rebuild_routes()

        stream = sink_stream(2)
        out = np.zeros((BLOCK, 2), dtype=np.float32)
        tone = sine(BLOCK, amplitude=0.9)

        for _ in range(40):
            for name in names:
                host.write_bus(name, tone)
            host._mix_into(stream, out, BLOCK)

        assert float(np.max(np.abs(out))) <= 1.001, "must prevent converter clipping"

    def test_quiet_mix_is_left_untouched(self):
        host = AudioHost(samplerate=RATE, blocksize=BLOCK)
        host.create_bus('stereo', channels=2)
        host.graph_holder.commit(RoutingGraph(connections=(
            Connection(bus_node('stereo'), bus_node('sink')),
        )))
        host._rebuild_routes()

        stream = sink_stream(2)
        out = np.zeros((BLOCK, 2), dtype=np.float32)
        tone = sine(BLOCK, amplitude=0.2)

        for _ in range(3):
            host.write_bus('stereo', tone)
            host._mix_into(stream, out, BLOCK)

        np.testing.assert_allclose(out, tone, atol=1e-5)


class TestChannelStripsInTheHost:
    def test_device_streams_are_given_strips(self):
        stream = _DeviceStream(StreamConfig(
            device=fake_device(inputs=2, outputs=2),
            samplerate=RATE, blocksize=BLOCK, input_channels=2, output_channels=2,
        ))

        assert stream.input_strip is not None
        assert stream.output_strip is not None
        assert stream.input_strip.is_transparent()

    def test_capture_does_not_write_to_the_driver_buffer(self):
        """
        PortAudio's input buffer is the driver's memory and must be treated as read-only.
        Trim has to be applied in our own scratch.
        """
        host = AudioHost(samplerate=RATE, blocksize=BLOCK)
        stream = _DeviceStream(StreamConfig(
            device=fake_device(inputs=2, outputs=0),
            samplerate=RATE, blocksize=BLOCK, input_channels=2,
        ))
        stream.input_strip.set_master_gain(0.5)

        indata = np.ones((BLOCK, 2), dtype=np.float32)
        original = indata.copy()

        host._publish_capture(str(stream.node), indata, stream)

        np.testing.assert_array_equal(indata, original, "must not modify the input buffer")

    def test_input_trim_reaches_the_routed_audio(self):
        """
        The point of the whole channel-control wiring: a trim set on a device must change
        what downstream destinations receive.
        """
        host = AudioHost(samplerate=RATE, blocksize=BLOCK)
        device = fake_device(name='iface', inputs=2, outputs=0)
        stream = _DeviceStream(StreamConfig(
            device=device, samplerate=RATE, blocksize=BLOCK, input_channels=2,
        ))
        host._streams[device.key] = stream

        from tonesphere.engine.graph import device_node

        host.graph_holder.commit(RoutingGraph(connections=(
            Connection(device_node(device.key), bus_node('sink')),
        )))
        host._rebuild_routes()

        stream.input_strip.set_master_gain(0.25)

        sink = sink_stream(2)
        out = np.zeros((BLOCK, 2), dtype=np.float32)

        for _ in range(4):
            host._publish_capture(str(stream.node), np.ones((BLOCK, 2), dtype=np.float32), stream)
            host._mix_into(sink, out, BLOCK)

        assert float(out[BLOCK // 2, 0]) == pytest.approx(0.25, abs=1e-2)


class TestDriftResamplingIsWiredUp:
    def test_cross_device_routes_get_a_resampler(self):
        """Two devices never share a clock, so their route needs drift correction."""
        host = AudioHost(samplerate=RATE, blocksize=BLOCK)

        source = fake_device(name='in', inputs=2, outputs=0)
        dest = fake_device(name='out', inputs=0, outputs=2)

        host._streams[source.key] = _DeviceStream(StreamConfig(
            device=source, samplerate=RATE, blocksize=BLOCK, input_channels=2))
        dest_stream = _DeviceStream(StreamConfig(
            device=dest, samplerate=RATE, blocksize=BLOCK, output_channels=2))
        host._streams[dest.key] = dest_stream

        from tonesphere.engine.graph import device_node

        host.graph_holder.commit(RoutingGraph(connections=(
            Connection(device_node(source.key), device_node(dest.key)),
        )))
        host._rebuild_routes()

        assert host._routes.resamplers, "cross-device route should be drift corrected"

    def test_same_device_loopback_gets_no_resampler(self):
        """
        A duplex stream has one clock, so input and output cannot drift. Resampling it
        would add interpolation error where there is no error to correct.
        """
        host = AudioHost(samplerate=RATE, blocksize=BLOCK)
        device = fake_device(name='iface', inputs=2, outputs=2)

        stream = _DeviceStream(StreamConfig(
            device=device, samplerate=RATE, blocksize=BLOCK,
            input_channels=2, output_channels=2,
        ))
        host._streams[device.key] = stream

        from tonesphere.engine.graph import device_node

        node = device_node(device.key)
        host.graph_holder.commit(RoutingGraph(connections=(Connection(node, node),)))
        host._rebuild_routes()

        assert not host._routes.resamplers, "a single clock needs no drift correction"
