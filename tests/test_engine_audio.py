"""
The acceptance tests for Phase 1.

The single question this project failed to answer for its whole existence is "does a
sample get from A to B?" These tests answer it by measurement: put a known signal in,
assert the expected signal comes out at the expected amplitude.

Tests needing real hardware are marked `hardware` and excluded on CI, which has no audio
devices. Everything else — ring buffers, the graph, channel mapping, the mixer — runs
everywhere, because that is where the bugs were.
"""

import math

import numpy as np
import pytest

from tonesphere.engine.graph import (
    Connection, GraphHolder, RoutingGraph, bus_node, db_to_linear, device_node, linear_to_db,
)
from tonesphere.engine.host import AudioHost
from tonesphere.engine.ringbuffer import AudioRingBuffer

BLOCK = 256
RATE = 48000


def sine(frames: int, freq: float = 1000.0, rate: int = RATE,
         amplitude: float = 0.5, channels: int = 2, phase: float = 0.0) -> np.ndarray:
    """A test tone. 1 kHz because it sits mid-band where nothing rolls it off."""
    t = (np.arange(frames, dtype=np.float64) + phase) / rate
    wave = (amplitude * np.sin(2.0 * math.pi * freq * t)).astype(np.float32)
    return np.repeat(wave.reshape(-1, 1), channels, axis=1)


def dominant_frequency(block: np.ndarray, rate: int = RATE) -> float:
    """Peak bin of the spectrum, used to prove the tone survived unshifted."""
    mono = block[:, 0] if block.ndim > 1 else block
    spectrum = np.abs(np.fft.rfft(mono * np.hanning(len(mono))))
    return float(np.fft.rfftfreq(len(mono), 1.0 / rate)[int(np.argmax(spectrum))])


class TestRingBuffer:
    def test_write_then_read_returns_identical_samples(self):
        ring = AudioRingBuffer(BLOCK * 4, channels=2)
        written = sine(BLOCK)

        assert ring.write(written) == BLOCK

        out = np.zeros((BLOCK, 2), dtype=np.float32)
        assert ring.read_into(out) == BLOCK
        np.testing.assert_array_equal(out, written)

    def test_wraps_around_end_without_corruption(self):
        """The wrap is where ring buffers are usually wrong, so exercise it repeatedly."""
        ring = AudioRingBuffer(BLOCK * 2, channels=2)
        out = np.zeros((BLOCK, 2), dtype=np.float32)

        for i in range(20):
            block = sine(BLOCK, phase=i * BLOCK)
            ring.write(block)
            ring.read_into(out)
            np.testing.assert_allclose(out, block, atol=1e-7)

    def test_underflow_yields_silence_not_stale_audio(self):
        """
        Repeating the previous block on starvation is audible as a buzz. Silence is the
        correct sound for "no data".
        """
        ring = AudioRingBuffer(BLOCK * 2, channels=2)
        ring.write(sine(BLOCK // 2, amplitude=0.9))

        out = np.full((BLOCK, 2), 7.0, dtype=np.float32)   # poison
        got = ring.read_into(out)

        assert got == BLOCK // 2
        assert np.all(out[got:] == 0.0)
        assert ring.underflow_count == 1

    def test_overflow_is_counted_and_does_not_corrupt(self):
        ring = AudioRingBuffer(BLOCK, channels=2)

        assert ring.write(sine(BLOCK)) == BLOCK
        assert ring.write(sine(BLOCK)) == 0
        assert ring.overflow_count == 1
        assert ring.available == BLOCK

    def test_available_and_space_are_consistent(self):
        ring = AudioRingBuffer(1000, channels=2)
        ring.write(np.zeros((400, 2), dtype=np.float32))

        assert ring.available == 400
        assert ring.space == 600
        assert ring.available + ring.space == ring.capacity

    def test_discard_drops_oldest(self):
        ring = AudioRingBuffer(BLOCK * 4, channels=1)
        ring.write(np.arange(100, dtype=np.float32).reshape(-1, 1))

        assert ring.discard(40) == 40

        out = np.zeros((60, 1), dtype=np.float32)
        ring.read_into(out)
        assert out[0, 0] == 40.0

    def test_peek_does_not_consume(self):
        ring = AudioRingBuffer(BLOCK * 4, channels=2)
        ring.write(sine(BLOCK))

        out = np.zeros((BLOCK, 2), dtype=np.float32)
        ring.peek_latest(out)

        assert ring.available == BLOCK, "metering must not steal audio from the mixer"

    def test_survives_concurrent_producer_and_consumer(self):
        """
        The SPSC claim is the basis for having no lock in the audio path, so exercise it
        under real threads rather than trusting the reasoning.
        """
        import threading

        ring = AudioRingBuffer(BLOCK * 8, channels=1)
        total = 500
        produced = np.arange(total * BLOCK, dtype=np.float32).reshape(-1, 1)
        consumed = []
        errors = []

        def produce():
            try:
                offset = 0
                while offset < total * BLOCK:
                    n = ring.write(produced[offset:offset + BLOCK])
                    offset += n
            except Exception as e:
                errors.append(e)

        def consume():
            try:
                out = np.zeros((BLOCK, 1), dtype=np.float32)
                received = 0
                while received < total * BLOCK:
                    got = ring.read_into(out)
                    if got:
                        consumed.append(out[:got].copy())
                        received += got
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=produce), threading.Thread(target=consume)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=30)

        assert not errors, f"SPSC violation: {errors}"
        result = np.concatenate(consumed)[:total * BLOCK]
        np.testing.assert_array_equal(result, produced)


class TestGraph:
    def test_gain_conversions_round_trip(self):
        for db in (-40.0, -12.0, -6.0, 0.0, 6.0):
            assert linear_to_db(db_to_linear(db)) == pytest.approx(db, abs=1e-6)

    def test_unity_gain_is_one(self):
        assert db_to_linear(0.0) == pytest.approx(1.0)

    def test_minus_six_db_is_half_amplitude(self):
        assert db_to_linear(-6.0206) == pytest.approx(0.5, abs=1e-4)

    def test_floor_is_true_silence(self):
        """A fader at the bottom must be silent, not -60 dB of audible hiss."""
        assert db_to_linear(-60.0) == 0.0
        assert db_to_linear(-90.0) == 0.0

    def test_muted_connection_is_inaudible(self):
        connection = Connection(bus_node('a'), bus_node('b'), gain=1.0, muted=True)

        assert connection.effective_gain == 0.0
        assert connection.is_audible is False

    def test_muted_route_is_excluded_from_mix_lookup(self):
        graph = RoutingGraph(connections=(
            Connection(bus_node('a'), bus_node('out'), muted=True),
            Connection(bus_node('b'), bus_node('out')),
        ))

        sources = graph.sources_for(bus_node('out'))
        assert len(sources) == 1
        assert sources[0].source == bus_node('b')

    def test_solo_suppresses_other_sources(self):
        graph = RoutingGraph(
            connections=(
                Connection(bus_node('gtr'), bus_node('out')),
                Connection(bus_node('mic'), bus_node('out')),
            ),
            soloed=frozenset({bus_node('gtr')}),
        )

        sources = graph.sources_for(bus_node('out'))
        assert len(sources) == 1
        assert sources[0].source == bus_node('gtr')

    def test_clearing_solo_restores_all_sources(self):
        graph = RoutingGraph(connections=(
            Connection(bus_node('gtr'), bus_node('out')),
            Connection(bus_node('mic'), bus_node('out')),
        ))
        graph = graph.with_solo(bus_node('gtr'), True)
        assert len(graph.sources_for(bus_node('out'))) == 1

        graph = graph.with_solo(bus_node('gtr'), False)
        assert len(graph.sources_for(bus_node('out'))) == 2

    def test_graph_is_immutable(self):
        """Immutability is what makes the lock-free handoff safe."""
        original = RoutingGraph(connections=(Connection(bus_node('a'), bus_node('b')),))
        modified = original.with_connection(Connection(bus_node('c'), bus_node('d')))

        assert len(original.connections) == 1
        assert len(modified.connections) == 2
        assert original is not modified

    def test_direct_feedback_is_detected(self):
        assert RoutingGraph().would_feedback(bus_node('a'), bus_node('a')) is True

    def test_indirect_feedback_is_detected(self):
        """a -> b -> c already exists; adding c -> a would howl."""
        graph = RoutingGraph(connections=(
            Connection(bus_node('a'), bus_node('b')),
            Connection(bus_node('b'), bus_node('c')),
        ))

        assert graph.would_feedback(bus_node('c'), bus_node('a')) is True

    def test_non_cyclic_route_is_allowed(self):
        graph = RoutingGraph(connections=(Connection(bus_node('a'), bus_node('b')),))

        assert graph.would_feedback(bus_node('a'), bus_node('c')) is False

    def test_removing_a_node_removes_its_routes(self):
        graph = RoutingGraph(connections=(
            Connection(device_node('gone'), bus_node('out')),
            Connection(bus_node('keep'), bus_node('out')),
        ))

        pruned = graph.without_node(device_node('gone'))
        assert len(pruned.connections) == 1

    def test_round_trips_through_dict(self):
        graph = RoutingGraph(
            connections=(
                Connection(device_node('iface'), bus_node('main'), gain=0.5),
                Connection(bus_node('main'), device_node('phones'), muted=True),
            ),
            soloed=frozenset({device_node('iface')}),
            master_gain=0.8,
        )

        restored = RoutingGraph.from_dict(graph.to_dict())

        assert len(restored.connections) == 2
        assert restored.master_gain == pytest.approx(0.8)
        assert restored.soloed == graph.soloed


class TestGraphHolder:
    def test_commit_publishes_new_graph(self):
        holder = GraphHolder()
        assert holder.current().connections == ()

        holder.commit(RoutingGraph(connections=(Connection(bus_node('a'), bus_node('b')),)))
        assert len(holder.current().connections) == 1

    def test_generation_advances_on_each_commit(self):
        holder = GraphHolder()
        first = holder.generation
        holder.commit(RoutingGraph())

        assert holder.generation == first + 1

    def test_reader_holding_old_graph_is_unaffected_by_a_commit(self):
        """
        The callback reads once per block and keeps that reference. A concurrent edit must
        not change what it is working from.
        """
        holder = GraphHolder(RoutingGraph(connections=(
            Connection(bus_node('a'), bus_node('out')),
        )))

        observed = holder.current()
        holder.commit(RoutingGraph())

        assert len(observed.sources_for(bus_node('out'))) == 1


def make_sink_stream(out_channels=2, node=None):
    """
    A `_DeviceStream` wired to a bus node so the real mixer can run with no hardware.

    `_mix_into` is verbatim the code the audio callback executes, so driving it directly
    tests the real path rather than a reimplementation of it.
    """
    from tonesphere.engine.devices import DeviceInfo, HostApi
    from tonesphere.engine.host import StreamConfig, _DeviceStream

    device = DeviceInfo(
        index=0, name='sink', host_api=HostApi.UNKNOWN, host_api_name='test',
        max_input_channels=0, max_output_channels=out_channels,
        default_samplerate=RATE,
        default_low_input_latency_ms=0.0, default_low_output_latency_ms=0.0,
        default_high_input_latency_ms=0.0, default_high_output_latency_ms=0.0,
    )
    stream = _DeviceStream(StreamConfig(
        device=device, samplerate=RATE, blocksize=BLOCK, output_channels=out_channels,
    ))
    stream.node = node if node is not None else bus_node('sink')
    return stream


class TestBusMixing:
    """The mixer, exercised through buses so it runs with no hardware."""

    def make_host(self):
        host = AudioHost(samplerate=RATE, blocksize=BLOCK)
        host.create_bus('src_a', channels=2)
        host.create_bus('src_b', channels=2)
        host.create_bus('sink', channels=2)
        return host

    def mix(self, host, graph, sources=None, out_channels=2, frames=BLOCK, stream=None):
        """
        Publish a graph, feed the sources, run one block of the real callback mixer.

        The order matters and mirrors reality: routes must exist before a producer can
        fan audio out to them.
        """
        host.graph_holder.commit(graph)
        host._rebuild_routes()

        for name, block in (sources or {}).items():
            host.write_bus(name, block)

        if stream is None:
            stream = make_sink_stream(out_channels)

        out = np.zeros((frames, out_channels), dtype=np.float32)
        host._mix_into(stream, out, frames)
        return out

    def test_single_source_passes_through_at_unity(self):
        host = self.make_host()
        tone = sine(BLOCK, amplitude=0.5)

        out = self.mix(
            host,
            RoutingGraph(connections=(
                Connection(bus_node('src_a'), bus_node('sink'), gain=1.0),
            )),
            {'src_a': tone},
        )

        np.testing.assert_allclose(out, tone, atol=1e-6)

    def test_tone_frequency_is_preserved(self):
        """Guards against off-by-one indexing that would resample and shift pitch."""
        host = self.make_host()

        out = self.mix(
            host,
            RoutingGraph(connections=(Connection(bus_node('src_a'), bus_node('sink')),)),
            {'src_a': sine(BLOCK, freq=1000.0)},
        )

        assert dominant_frequency(out) == pytest.approx(1000.0, abs=RATE / BLOCK)

    def test_gain_scales_amplitude(self):
        host = self.make_host()

        out = self.mix(
            host,
            RoutingGraph(connections=(
                Connection(bus_node('src_a'), bus_node('sink'), gain=0.5),
            )),
            {'src_a': sine(BLOCK, amplitude=0.8)},
        )

        assert float(np.max(np.abs(out))) == pytest.approx(0.4, abs=1e-3)

    def test_two_sources_sum(self):
        host = self.make_host()

        out = self.mix(
            host,
            RoutingGraph(connections=(
                Connection(bus_node('src_a'), bus_node('sink')),
                Connection(bus_node('src_b'), bus_node('sink')),
            )),
            {
                'src_a': np.full((BLOCK, 2), 0.3, dtype=np.float32),
                'src_b': np.full((BLOCK, 2), 0.2, dtype=np.float32),
            },
        )

        np.testing.assert_allclose(out, 0.5, atol=1e-6)

    def test_opposite_phase_sources_cancel(self):
        """Proves the mixer sums rather than picking a winner."""
        host = self.make_host()
        tone = sine(BLOCK, amplitude=0.5)

        out = self.mix(
            host,
            RoutingGraph(connections=(
                Connection(bus_node('src_a'), bus_node('sink')),
                Connection(bus_node('src_b'), bus_node('sink')),
            )),
            {'src_a': tone, 'src_b': -tone},
        )

        assert float(np.max(np.abs(out))) < 1e-6

    def test_muted_source_contributes_nothing(self):
        host = self.make_host()

        out = self.mix(
            host,
            RoutingGraph(connections=(
                Connection(bus_node('src_a'), bus_node('sink'), muted=True),
            )),
            {'src_a': sine(BLOCK, amplitude=0.9)},
        )

        assert float(np.max(np.abs(out))) == 0.0

    def test_master_gain_applies(self):
        host = self.make_host()

        out = self.mix(
            host,
            RoutingGraph(
                connections=(Connection(bus_node('src_a'), bus_node('sink')),),
                master_gain=0.5,
            ),
            {'src_a': np.full((BLOCK, 2), 0.8, dtype=np.float32)},
        )

        np.testing.assert_allclose(out, 0.4, atol=1e-6)

    def test_no_route_produces_silence_not_noise(self):
        host = self.make_host()

        out = self.mix(host, RoutingGraph(), {})

        assert float(np.max(np.abs(out))) == 0.0

    def test_one_source_feeding_two_destinations_reaches_both(self):
        """
        Regression: rings were shared per source, and `read_into` consumes. Whichever
        output callback ran first took the audio and the second got silence — fan-out
        was broken, and non-deterministically so. Each route now has its own ring.
        """
        host = AudioHost(samplerate=RATE, blocksize=BLOCK)
        host.create_bus('guitar', channels=2)

        graph = RoutingGraph(connections=(
            Connection(bus_node('guitar'), bus_node('phones')),
            Connection(bus_node('guitar'), bus_node('recorder')),
        ))
        host.graph_holder.commit(graph)
        host._rebuild_routes()

        tone = sine(BLOCK, amplitude=0.5)
        host.write_bus('guitar', tone)

        phones = make_sink_stream(2, node=bus_node('phones'))
        recorder = make_sink_stream(2, node=bus_node('recorder'))

        out_phones = np.zeros((BLOCK, 2), dtype=np.float32)
        out_recorder = np.zeros((BLOCK, 2), dtype=np.float32)

        host._mix_into(phones, out_phones, BLOCK)
        host._mix_into(recorder, out_recorder, BLOCK)

        np.testing.assert_allclose(out_phones, tone, atol=1e-6)
        np.testing.assert_allclose(out_recorder, tone, atol=1e-6)

    def test_fan_out_order_does_not_matter(self):
        """Neither destination may depend on which callback happens to run first."""
        host = AudioHost(samplerate=RATE, blocksize=BLOCK)
        host.create_bus('guitar', channels=2)
        host.graph_holder.commit(RoutingGraph(connections=(
            Connection(bus_node('guitar'), bus_node('a')),
            Connection(bus_node('guitar'), bus_node('b')),
        )))
        host._rebuild_routes()

        tone = sine(BLOCK, amplitude=0.5)
        host.write_bus('guitar', tone)

        # Reverse order from the previous test.
        out_b = np.zeros((BLOCK, 2), dtype=np.float32)
        out_a = np.zeros((BLOCK, 2), dtype=np.float32)
        host._mix_into(make_sink_stream(2, node=bus_node('b')), out_b, BLOCK)
        host._mix_into(make_sink_stream(2, node=bus_node('a')), out_a, BLOCK)

        np.testing.assert_allclose(out_a, tone, atol=1e-6)
        np.testing.assert_allclose(out_b, tone, atol=1e-6)

    def test_mono_source_fills_both_output_channels(self):
        host = AudioHost(samplerate=RATE, blocksize=BLOCK)
        host.create_bus('mono', channels=1)

        out = self.mix(
            host,
            RoutingGraph(connections=(Connection(bus_node('mono'), bus_node('sink')),)),
            {'mono': np.full((BLOCK, 1), 0.5, dtype=np.float32)},
            out_channels=2,
        )

        np.testing.assert_allclose(out[:, 0], 0.5, atol=1e-6)
        np.testing.assert_allclose(out[:, 1], 0.5, atol=1e-6)

    def test_stereo_to_mono_averages_rather_than_dropping(self):
        """
        Taking only the left channel would silence a hard-right-panned guitar. Averaging
        keeps it.
        """
        host = AudioHost(samplerate=RATE, blocksize=BLOCK)
        host.create_bus('stereo', channels=2)

        hard_right = np.zeros((BLOCK, 2), dtype=np.float32)
        hard_right[:, 1] = 0.8

        out = self.mix(
            host,
            RoutingGraph(connections=(Connection(bus_node('stereo'), bus_node('sink')),)),
            {'stereo': hard_right},
            out_channels=1,
        )

        assert float(np.max(np.abs(out))) == pytest.approx(0.4, abs=1e-3)

    def test_mixer_allocates_nothing_per_block(self):
        """
        Allocation in a callback risks a GC pause, and a GC pause is a dropout. Assert
        the steady state is allocation-free.
        """
        import gc
        import tracemalloc

        host = self.make_host()
        host.graph_holder.commit(RoutingGraph(connections=(
            Connection(bus_node('src_a'), bus_node('sink')),
            Connection(bus_node('src_b'), bus_node('sink')),
        )))
        host._rebuild_routes()

        stream = make_sink_stream(2)
        out = np.zeros((BLOCK, 2), dtype=np.float32)

        # Warm up so any lazily-created scratch already exists.
        for _ in range(10):
            host.write_bus('src_a', sine(BLOCK))
            host.write_bus('src_b', sine(BLOCK))
            host._mix_into(stream, out, BLOCK)

        gc.collect()
        tracemalloc.start()
        before = tracemalloc.take_snapshot()

        for _ in range(200):
            host._mix_into(stream, out, BLOCK)

        after = tracemalloc.take_snapshot()
        tracemalloc.stop()

        growth = sum(s.size_diff for s in after.compare_to(before, 'filename'))
        # tracemalloc's own bookkeeping is unavoidable; per-block buffer allocation would
        # be orders of magnitude larger than this.
        assert growth < 100_000, f"mixer allocated {growth} bytes over 200 blocks"

    def test_gain_change_is_ramped_not_stepped(self):
        """
        An instant gain change is a step discontinuity in the waveform, and a step
        discontinuity is a click. The block must slide between the two gains.
        """
        host = self.make_host()
        steady = np.full((BLOCK, 2), 0.5, dtype=np.float32)
        stream = make_sink_stream(2)
        out = np.zeros((BLOCK, 2), dtype=np.float32)

        host.graph_holder.commit(RoutingGraph(connections=(
            Connection(bus_node('src_a'), bus_node('sink'), gain=1.0),
        )))
        host._rebuild_routes()

        host.write_bus('src_a', steady)
        host._mix_into(stream, out, BLOCK)
        assert out[-1, 0] == pytest.approx(0.5, abs=1e-4)

        # Drop hard to silence.
        host.graph_holder.commit(RoutingGraph(connections=(
            Connection(bus_node('src_a'), bus_node('sink'), gain=0.0),
        )))
        host._rebuild_routes()
        host.write_bus('src_a', steady)
        host._mix_into(stream, out, BLOCK)

        assert out[0, 0] == pytest.approx(0.5, abs=1e-3), "ramp should start at the old gain"
        assert abs(float(out[-1, 0])) < 1e-3, "ramp should finish at the new gain"

        steps = np.abs(np.diff(out[:, 0]))
        assert float(np.max(steps)) < 0.05, "gain ramp must be smooth"

    def test_muting_ramps_down_rather_than_cutting(self):
        """
        Regression: a muted route was excluded from the mix outright, so the audio
        stopped mid-waveform — a click. It must be rendered once more, ramping to zero.
        """
        host = self.make_host()
        steady = np.full((BLOCK, 2), 0.5, dtype=np.float32)
        stream = make_sink_stream(2)
        out = np.zeros((BLOCK, 2), dtype=np.float32)

        host.graph_holder.commit(RoutingGraph(connections=(
            Connection(bus_node('src_a'), bus_node('sink')),
        )))
        host._rebuild_routes()
        host.write_bus('src_a', steady)
        host._mix_into(stream, out, BLOCK)

        host.graph_holder.commit(RoutingGraph(connections=(
            Connection(bus_node('src_a'), bus_node('sink'), muted=True),
        )))
        host._rebuild_routes()
        host.write_bus('src_a', steady)
        host._mix_into(stream, out, BLOCK)

        assert out[0, 0] == pytest.approx(0.5, abs=1e-3), "mute must ramp, not cut"
        assert abs(float(out[-1, 0])) < 1e-3

    def test_deleting_a_route_ramps_down_rather_than_cutting(self):
        host = self.make_host()
        steady = np.full((BLOCK, 2), 0.5, dtype=np.float32)
        stream = make_sink_stream(2)
        out = np.zeros((BLOCK, 2), dtype=np.float32)

        host.graph_holder.commit(RoutingGraph(connections=(
            Connection(bus_node('src_a'), bus_node('sink')),
        )))
        host._rebuild_routes()
        host.write_bus('src_a', steady)
        host._mix_into(stream, out, BLOCK)

        # Route gone entirely, but its ring is retained for the fade-out block.
        host.write_bus('src_a', steady)
        host.graph_holder.commit(RoutingGraph())
        host._mix_into(stream, out, BLOCK)

        assert out[0, 0] == pytest.approx(0.5, abs=1e-3), "deletion must ramp, not cut"
        assert abs(float(out[-1, 0])) < 1e-3

    def test_route_stays_silent_after_fade_out_completes(self):
        host = self.make_host()
        stream = make_sink_stream(2)
        out = np.zeros((BLOCK, 2), dtype=np.float32)

        host.graph_holder.commit(RoutingGraph(connections=(
            Connection(bus_node('src_a'), bus_node('sink')),
        )))
        host._rebuild_routes()
        host.write_bus('src_a', np.full((BLOCK, 2), 0.5, dtype=np.float32))
        host._mix_into(stream, out, BLOCK)

        host.graph_holder.commit(RoutingGraph())
        host._mix_into(stream, out, BLOCK)   # fade-out block
        host._mix_into(stream, out, BLOCK)   # must now be fully silent

        assert float(np.max(np.abs(out))) == 0.0

    def test_zero_gain_route_still_drains_its_ring(self):
        """
        A silent route that stops consuming would let its ring fill until drift
        correction fired on a route nobody can hear.
        """
        host = self.make_host()
        stream = make_sink_stream(2)
        out = np.zeros((BLOCK, 2), dtype=np.float32)

        host.graph_holder.commit(RoutingGraph(connections=(
            Connection(bus_node('src_a'), bus_node('sink'), muted=True),
        )))
        host._rebuild_routes()

        ring = host._routes.ring_for(str(bus_node('src_a')), str(bus_node('sink')))

        for _ in range(3):
            host.write_bus('src_a', sine(BLOCK))
            host._mix_into(stream, out, BLOCK)

        assert ring.available == 0, "muted route must still consume its input"


class TestDriftCorrection:
    def make_route(self, blocks_buffered: int):
        """A host with one route whose ring already holds `blocks_buffered` blocks."""
        host = AudioHost(samplerate=RATE, blocksize=BLOCK)
        host.create_bus('src', channels=2)
        host.graph_holder.commit(RoutingGraph(connections=(
            Connection(bus_node('src'), bus_node('sink')),
        )))
        host._rebuild_routes()

        ring = host._routes.ring_for(str(bus_node('src')), str(bus_node('sink')))
        for _ in range(blocks_buffered):
            ring.write(sine(BLOCK))

        return host, ring, make_sink_stream(2)

    def test_backlog_is_trimmed_to_bound_latency(self):
        """
        A faster source clock fills its ring forever, and latency grows with it. Trimming
        costs one small glitch and keeps latency bounded.
        """
        host, ring, stream = self.make_route(blocks_buffered=4)

        backlog_before = ring.available
        host._drift_correct(ring, stream)

        assert ring.available < backlog_before
        assert ring.available == BLOCK
        assert stream.drift_corrections == 1

    def test_healthy_backlog_is_left_alone(self):
        host, ring, stream = self.make_route(blocks_buffered=1)

        host._drift_correct(ring, stream)

        assert ring.available == BLOCK
        assert stream.drift_corrections == 0


class TestHostStatistics:
    def test_unmeasured_values_are_none_before_running(self):
        stats = AudioHost().statistics()

        assert stats.running is False
        assert stats.cpu_load is None
        assert stats.measured_latency_ms is None

    def test_nominal_latency_is_computed_and_labelled_separately(self):
        stats = AudioHost(samplerate=48000, blocksize=256).statistics()

        assert stats.nominal_latency_ms == pytest.approx(5.333, abs=0.01)
        assert stats.measured_latency_ms is None, "must not pass arithmetic off as measured"

    def test_audio_path_active_tracks_running_state(self):
        assert AudioHost().statistics().as_dict()['audio_path_active'] is False

    def test_configure_reports_unknown_devices_instead_of_failing_silently(self):
        host = AudioHost()
        problems = host.configure(RoutingGraph(connections=(
            Connection(device_node('nope::does not exist'), bus_node('out')),
        )))

        assert any('not found' in p.lower() for p in problems)

    def test_start_without_configuration_reports_why(self):
        problems = AudioHost().start()

        assert problems
        assert not AudioHost().is_running


@pytest.mark.hardware
class TestRealHardware:
    """
    The tests that prove audio genuinely reaches a device.

    Excluded from CI, which has no audio hardware. Run locally with:
        uv run pytest -m hardware
    """

    def test_devices_are_enumerated(self):
        from tonesphere.engine.devices import enumerate_devices

        devices = enumerate_devices()
        assert devices, "no audio devices found"
        assert any(d.can_output for d in devices)

    def test_a_host_api_is_available(self):
        from tonesphere.engine.devices import available_host_apis, preferred_host_api

        apis = available_host_apis()
        assert apis
        assert preferred_host_api(apis) is not None

    def test_output_stream_runs_without_xruns(self):
        """
        Open a real output, play a tone for two seconds, and require a clean run. This is
        the test the project never had: proof that the audio path works end to end.
        """
        from tonesphere.engine.devices import enumerate_devices, preferred_host_api

        devices = enumerate_devices()
        api = preferred_host_api()
        candidates = [d for d in devices if d.can_output and d.host_api == api]
        if not candidates:
            pytest.skip(f"no output device on {api}")

        device = next((d for d in candidates if d.is_default_output), candidates[0])

        host = AudioHost(samplerate=RATE, blocksize=BLOCK, host_api=api)
        host.create_bus('tone', channels=2)

        graph = RoutingGraph(connections=(
            Connection(bus_node('tone'), device_node(device.key), gain=0.1),
        ))

        problems = host.configure(graph)
        assert not problems, f"configure failed: {problems}"

        problems = host.start()
        assert not problems, f"start failed: {problems}"

        try:
            import time
            phase = 0
            deadline = time.monotonic() + 2.0
            while time.monotonic() < deadline:
                host.write_bus('tone', sine(BLOCK, amplitude=0.2, phase=phase))
                phase += BLOCK
                time.sleep(BLOCK / RATE / 2)

            stats = host.statistics()

            assert stats.callback_count > 100, f"callback barely ran: {stats.callback_count}"
            assert stats.callback_errors == 0, f"callback raised: {host.last_callback_error}"
            assert stats.measured_latency_ms is not None
            assert stats.cpu_load is not None
            assert stats.xruns == 0, f"{stats.xruns} xruns"
        finally:
            host.stop()
