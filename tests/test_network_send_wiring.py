"""
Network send and receive as they are actually wired into the engine.

Two things are proved here, and neither can be proved by a mock.

**A known signal survives.** A 1 kHz sine is written into a bus on one engine, sent over
real localhost UDP, and read back out of a bus on a second engine. The assertion is not
"a packet arrived" but "these are the same samples", which is only possible because the
default quality is float32 and therefore lossless — so the comparison is exact rather
than approximate, on top of the dominant-frequency and RMS checks the rest of the suite
uses.

**The route survives the patchbay.** `AudioEngine._build_graph` reconstructs the entire
host graph from `routing_matrix.connections`, and every routing mutation republishes it.
A network send target registered any other way is erased the next time the user touches
anything. `TestNetworkSendSurvivesPatchbayEdits` asserts the real route survives such an
edit, and — as the control that makes that assertion mean something — that a connection
pushed straight into the host graph genuinely does not.

No hardware anywhere: buses, ring buffers and UDP sockets only, so this runs on all three
platforms. Nothing calls `initialize()`, so PortAudio is never touched at all.
"""

import math
import time

import numpy as np
import pytest

from tonesphere.engine.graph import Connection, network_node
from tonesphere.network.udp_transport import (
    CODEC_PCM_FLOAT32,
    UdpPacket,
    frames_per_packet,
)

from tonesphere.core.engine import NETWORK_ID_BASE, AudioEngine  # isort: skip

RATE = 48000
BLOCK = 256
PACKET_FRAMES = frames_per_packet(CODEC_PCM_FLOAT32, 2)


def sine(frames: int, freq: float = 1000.0, rate: int = RATE,
         amplitude: float = 0.5, channels: int = 2, phase: float = 0.0) -> np.ndarray:
    t = (np.arange(frames, dtype=np.float64) + phase) / rate
    wave = (amplitude * np.sin(2.0 * math.pi * freq * t)).astype(np.float32)
    return np.repeat(wave.reshape(-1, 1), channels, axis=1)


def dominant_frequency(block: np.ndarray, rate: int = RATE) -> float:
    mono = block[:, 0] if block.ndim > 1 else block
    spectrum = np.abs(np.fft.rfft(mono * np.hanning(len(mono))))
    return float(np.fft.rfftfreq(len(mono), 1.0 / rate)[int(np.argmax(spectrum))])


def rms(block: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(block, dtype=np.float64))))


@pytest.fixture
def engine():
    e = AudioEngine(sample_rate=RATE, buffer_size=BLOCK)
    yield e
    e.cleanup()


@pytest.fixture
def sender():
    e = AudioEngine(sample_rate=RATE, buffer_size=BLOCK)
    yield e
    e.cleanup()


@pytest.fixture
def receiver():
    e = AudioEngine(sample_rate=RATE, buffer_size=BLOCK)
    yield e
    e.cleanup()


def node_key(engine: AudioEngine, device_id: int) -> str:
    return str(engine._node_for(device_id))


def drain(engine: AudioEngine, source_id: int, dest_id: int) -> np.ndarray | None:
    """
    Take exactly the frames a route currently holds, and no more.

    Reading a fixed block instead would zero-fill whatever had not arrived yet, splicing
    silence into the middle of the captured signal and turning an exact comparison into a
    spectral smear.
    """
    source, dest = node_key(engine, source_id), node_key(engine, dest_id)
    stats = engine.host.route_statistics(source, dest)
    if not stats or not stats['available']:
        return None
    return engine.host.read_route(source, dest, stats['available'])


def link(sender: AudioEngine, receiver: AudioEngine, source_id: int,
         dest_id: int, target_latency_ms: float = 20.0) -> None:
    """Point one engine's send at the other's receive, over loopback."""
    # Port 0 on both: an explicit port would collide with a parallel test run or with
    # anything already holding 9002, and neither engine needs a predictable one here.
    started, message = receiver.start_udp_transport('127.0.0.1', 0)
    assert started, message

    ok, message = receiver.register_network_receive(
        dest_id, transport='udp', target_latency_ms=target_latency_ms
    )
    assert ok, message

    started, message = sender.start_udp_transport('127.0.0.1', 0)
    assert started, message

    ok, message = sender.send_device_audio_to_network(source_id, transport='udp')
    assert ok, message

    # The worker's own thread is stopped so the test can drive `tick()` itself. Pacing
    # from the test is the difference between proving packetisation and proving the
    # machine was not busy at the time. Stopped before the peer is registered, so nothing
    # can leave while the test is still setting up.
    sender._send_worker.stop()

    # The worker waits one interval before its first read, so a route stopped this
    # promptly has read nothing. Asserted rather than assumed: a stray read would put a
    # block of silence at the head of the stream and every alignment below would be off
    # by it, which is a confusing way to find out.
    sink_id = sender.list_network_sends()[0]['sink_id']
    assert sender._send_worker.route(sink_id).reads == 0

    sender.add_udp_peer('receiver', *receiver.udp_transport.bound_address)


class TestKnownSignalSurvivesUdp:
    def test_a_sine_written_into_a_bus_arrives_sample_for_sample(self, sender, receiver):
        """
        The concrete end-to-end proof: same samples out as in.

        Feeding is a burst rather than realtime-paced, so the two engines' independent
        pacing cannot inject a gap and the comparison stays exact. `TestPlayoutLatency`
        covers the paced case, where timing is the thing under test.
        """
        source = sender.create_virtual_input('guitar', channels=2)
        destination = receiver.create_virtual_input('from-network', channels=2)
        monitor = receiver.create_virtual_output('monitor', channels=2)
        assert receiver.create_routing(destination, monitor)[0] is True

        link(sender, receiver, source, destination)

        blocks = 20
        signal = sine(blocks * BLOCK, freq=1000.0, amplitude=0.5)

        for index in range(blocks):
            chunk = signal[index * BLOCK:(index + 1) * BLOCK]
            assert sender.write_to_bus(source, chunk) == BLOCK
            sender._send_worker.tick()

        # Whole packets only: the tail frames that did not fill one are still in the
        # accumulator, exactly as they should be.
        packets = (blocks * BLOCK) // PACKET_FRAMES
        expected = signal[:packets * PACKET_FRAMES]

        captured = self._collect(receiver, destination, monitor, len(expected))

        assert len(captured) == len(expected), (
            f"{len(captured)} of {len(expected)} frames arrived"
        )

        # Printed unconditionally rather than only asserted below: if the exact comparison
        # below fails, these counts are what actually tells apart "packets were lost/evicted
        # on this run" from "the same samples landed in the wrong slot" -- two different
        # bugs that look identical as a bare shape/value mismatch.
        print("receive stats:", receiver.get_network_statistics()['udp']['receive'][str(destination)])

        # float32 end to end, so this is exact rather than approximate.
        np.testing.assert_allclose(captured, expected, atol=1e-6)

        assert dominant_frequency(captured) == pytest.approx(
            dominant_frequency(expected), abs=1.0
        )
        assert rms(captured) == pytest.approx(rms(expected), rel=0.01)

        stats = receiver.get_network_statistics()['udp']['receive'][str(destination)]
        buffer = stats['jitter_buffer']

        assert stats['packets_received'] == packets
        assert stats['packets_rejected'] == 0

        # Every packet sent was played in its own slot. This, plus the exact sample
        # comparison above, is what rules out a concealment having landed inside the
        # signal — a concealed slot would have spliced 146 frames of silence into it.
        assert buffer['packets_played_on_time'] == packets
        assert buffer['packets_late_dropped'] == 0
        assert buffer['packets_duplicate'] == 0
        assert buffer['packets_evicted'] == 0

        # `packets_lost` is deliberately *not* asserted to be zero. The sender stopped
        # and the playout thread did not, so it is still pulling slots nothing will ever
        # fill and reporting each as a loss — which is exactly right. A buffer that went
        # quiet without saying so is the failure mode being avoided here.

    def test_the_send_side_reports_what_it_actually_sent(self, sender, receiver):
        source = sender.create_virtual_input('guitar', channels=2)
        destination = receiver.create_virtual_input('from-network', channels=2)
        monitor = receiver.create_virtual_output('monitor', channels=2)
        receiver.create_routing(destination, monitor)

        link(sender, receiver, source, destination)

        for index in range(10):
            sender.write_to_bus(source, sine(BLOCK, phase=index * BLOCK))
            sender._send_worker.tick()

        route = next(iter(
            sender.get_network_statistics()['udp']['send']['routes'].values()
        ))

        assert route['channels'] == 2
        assert route['frames_per_packet'] == PACKET_FRAMES
        assert route['frames_read'] == 10 * BLOCK
        assert route['packets_sent'] == (10 * BLOCK) // PACKET_FRAMES
        assert route['packets_undeliverable'] == 0
        assert route['encode_errors'] == 0
        assert route['last_error'] is None
        # The ring's own count of reads the producer had not filled — the free, honest
        # answer to "is the sender starving?".
        assert route['ring']['underflow_count'] == 0

    def test_a_mono_source_survives_as_mono(self, sender, receiver):
        """
        Channel count comes from the audio, not from a default. A mono bus must not
        arrive as a silent second channel or a reshape error.
        """
        source = sender.create_virtual_input('mic', channels=1)
        destination = receiver.create_virtual_input('from-network', channels=1)
        monitor = receiver.create_virtual_output('monitor', channels=1)
        receiver.create_routing(destination, monitor)

        link(sender, receiver, source, destination)

        mono_packet_frames = frames_per_packet(CODEC_PCM_FLOAT32, 1)
        blocks = 12
        signal = sine(blocks * BLOCK, freq=440.0, channels=1)

        for index in range(blocks):
            sender.write_to_bus(source, signal[index * BLOCK:(index + 1) * BLOCK])
            sender._send_worker.tick()

        packets = (blocks * BLOCK) // mono_packet_frames
        expected = signal[:packets * mono_packet_frames]

        captured = self._collect(receiver, destination, monitor, len(expected))

        # See the stereo test above for why this is printed unconditionally.
        print("receive stats:", receiver.get_network_statistics()['udp']['receive'][str(destination)])

        assert captured.shape[1] == 1
        np.testing.assert_allclose(captured, expected, atol=1e-6)

    @staticmethod
    def _collect(receiver: AudioEngine, destination: int, monitor: int,
                 wanted: int, timeout: float = 5.0) -> np.ndarray:
        """
        Read the destination route as the paced playout thread fills it.

        Draining continuously is not a convenience: the route's ring is four blocks deep,
        so leaving it alone would make playout's writes start returning 0 and the audio
        would be lost rather than merely delayed.
        """
        collected: list[np.ndarray] = []
        total = 0
        deadline = time.monotonic() + timeout

        while total < wanted and time.monotonic() < deadline:
            piece = drain(receiver, destination, monitor)
            if piece is None:
                time.sleep(0.002)
                continue
            collected.append(piece)
            total += piece.shape[0]

        if not collected:
            return np.zeros((0, 2), dtype=np.float32)

        return np.concatenate(collected)[:wanted]


class TestPlayoutLatency:
    def test_playout_adds_a_bounded_latency_around_the_configured_target(
        self, sender, receiver
    ):
        """
        The latency claim, measured rather than asserted.

        Fed at realtime pace, the first frame cannot leave the receiver until the buffer
        has primed, and priming needs `target_packets` packets — which is a real quantity
        of audio the sender has to spend real time producing. So the lower bound proves
        the buffer is actually buffering rather than passing through, and the upper bound
        proves it is not accumulating unboundedly.

        The ceiling is loose on purpose. This runs on shared CI runners where a 10 ms
        thread wake-up is unremarkable, and a tight bound here would fail for reasons
        that have nothing to do with the transport.
        """
        target_latency_ms = 20.0

        source = sender.create_virtual_input('guitar', channels=2)
        destination = receiver.create_virtual_input('from-network', channels=2)
        monitor = receiver.create_virtual_output('monitor', channels=2)
        receiver.create_routing(destination, monitor)

        link(sender, receiver, source, destination, target_latency_ms)

        interval = BLOCK / RATE
        first_out: float | None = None
        started = time.monotonic()
        deadline = started

        for index in range(60):
            sender.write_to_bus(source, sine(BLOCK, phase=index * BLOCK))
            sender._send_worker.tick()

            piece = drain(receiver, destination, monitor)
            if piece is not None and first_out is None and rms(piece) > 1e-4:
                first_out = time.monotonic()

            deadline += interval
            remaining = deadline - time.monotonic()
            if remaining > 0:
                time.sleep(remaining)

        # Keep draining briefly in case the tone landed just after the feed ended.
        poll_until = time.monotonic() + 0.5
        while first_out is None and time.monotonic() < poll_until:
            piece = drain(receiver, destination, monitor)
            if piece is not None and rms(piece) > 1e-4:
                first_out = time.monotonic()
            else:
                time.sleep(0.002)

        assert first_out is not None, "no audio ever reached the destination bus"

        latency_ms = (first_out - started) * 1000.0
        buffer_stats = (
            receiver.get_network_statistics()['udp']['receive'][str(destination)]
        )['jitter_buffer']

        # The buffer genuinely primes: a pass-through would show up here as a couple of
        # milliseconds, not as the target depth.
        assert latency_ms >= target_latency_ms * 0.6, (
            f"first frame out after only {latency_ms:.1f} ms — the buffer did not prime"
        )
        assert latency_ms <= target_latency_ms + 250.0, (
            f"first frame out after {latency_ms:.1f} ms"
        )
        assert buffer_stats['target_packets'] * buffer_stats['packet_duration_ms'] >= (
            target_latency_ms
        )
        assert buffer_stats['packets_played_on_time'] > 0

        # The regression guard for the reason this transport did not work at first.
        # An MTU-sized packet is about 3 ms and `Event.wait` resolves to the system tick
        # — 8-15 ms on Windows — so a playout loop pulling one packet per wake-up runs at
        # roughly a third of the arrival rate. The backlog then grows until the buffer
        # hits its cap and evicts, which reports a perfectly healthy link as packet loss.
        # Eviction here means playout has stopped keeping up.
        assert buffer_stats['packets_evicted'] == 0, (
            f"the jitter buffer overflowed: playout is not keeping up with arrivals "
            f"({buffer_stats['buffered_latency_ms']:.1f} ms backed up)"
        )
        assert buffer_stats['buffered_latency_ms'] <= target_latency_ms * 4.0, (
            f"buffer is drifting upwards: {buffer_stats['buffered_latency_ms']:.1f} ms "
            f"held against a {target_latency_ms:.0f} ms target"
        )

        # A rate, not zero. The sender is paced by this test thread, so one scheduling
        # stall longer than the buffer depth legitimately starves playout — on a shared
        # CI runner that is a real possibility and not a transport defect. A systemic
        # failure would be far above this.
        pushed = buffer_stats['packets_pushed']
        assert buffer_stats['packets_lost'] <= pushed * 0.1, (
            f"{buffer_stats['packets_lost']} of {pushed} packets concealed"
        )


class TestNetworkSendSurvivesPatchbayEdits:
    """
    The gotcha that drives the whole send-side design.

    `_build_graph` rebuilds the host graph from the routing matrix on every mutation, so
    anything not in the matrix is transient. These tests assert both halves: that a
    matrix-registered network route survives, and that a side-channel one does not.
    """

    def _network_edges(self, engine: AudioEngine) -> list[str]:
        return [
            str(c.dest) for c in engine.host.graph_holder.current().connections
            if c.dest.kind == 'network'
        ]

    def test_a_network_send_is_a_real_routing_matrix_connection(self, engine):
        source = engine.create_virtual_input('guitar')
        engine.start_udp_transport('127.0.0.1', 0)

        ok, message = engine.send_device_audio_to_network(source, transport='udp')
        assert ok, message

        sink_id = engine.list_network_sends()[0]['sink_id']

        assert sink_id >= NETWORK_ID_BASE, "network ids must not overlap the bus range"
        assert (source, sink_id) in engine.routing_matrix.connections
        assert self._network_edges(engine) == [f"network:send_{sink_id}"]

    def test_it_survives_an_unrelated_route_being_added(self, engine):
        source = engine.create_virtual_input('guitar')
        other_in = engine.create_virtual_input('keys')
        other_out = engine.create_virtual_output('headphones')
        engine.start_udp_transport('127.0.0.1', 0)
        engine.send_device_audio_to_network(source, transport='udp')

        before = self._network_edges(engine)
        assert before

        assert engine.create_routing(other_in, other_out)[0] is True

        assert self._network_edges(engine) == before, (
            "adding an unrelated route wiped the network send"
        )

    def test_it_survives_a_route_being_changed_and_removed(self, engine):
        source = engine.create_virtual_input('guitar')
        other_in = engine.create_virtual_input('keys')
        other_out = engine.create_virtual_output('headphones')
        engine.start_udp_transport('127.0.0.1', 0)
        engine.send_device_audio_to_network(source, transport='udp')
        engine.create_routing(other_in, other_out)

        before = self._network_edges(engine)

        engine.set_routing_volume(other_in, other_out, 0.5)
        assert self._network_edges(engine) == before, "a gain change wiped it"

        engine.set_routing_mute(other_in, other_out, True)
        assert self._network_edges(engine) == before, "a mute wiped it"

        engine.set_routing_pan(other_in, other_out, -1.0)
        assert self._network_edges(engine) == before, "a pan wiped it"

        assert engine.remove_routing(other_in, other_out) is True
        assert self._network_edges(engine) == before, "removing a route wiped it"

    def test_audio_still_flows_after_a_patchbay_edit(self, sender, receiver):
        """
        Surviving in the graph is necessary but not sufficient — the route's ring has to
        survive too, or the send worker would read a route that exists and carries nothing.
        """
        source = sender.create_virtual_input('guitar')
        other_in = sender.create_virtual_input('keys')
        other_out = sender.create_virtual_output('headphones')
        destination = receiver.create_virtual_input('from-network')
        monitor = receiver.create_virtual_output('monitor')
        receiver.create_routing(destination, monitor)

        link(sender, receiver, source, destination)

        for index in range(10):
            sender.write_to_bus(source, sine(BLOCK, phase=index * BLOCK))
            sender._send_worker.tick()

        sent_before = sender.udp_transport.stats['packets_sent']
        assert sent_before > 0

        assert sender.create_routing(other_in, other_out)[0] is True

        for index in range(10, 20):
            sender.write_to_bus(source, sine(BLOCK, phase=index * BLOCK))
            sender._send_worker.tick()

        assert sender.udp_transport.stats['packets_sent'] > sent_before, (
            "the send stopped carrying audio after an unrelated patchbay edit"
        )

        route = next(iter(
            sender.get_network_statistics()['udp']['send']['routes'].values()
        ))
        assert route['ring'] is not None, "the route's ring did not survive the rebuild"
        assert route['empty_reads'] == 0

    def test_a_graph_edge_outside_the_matrix_does_not_survive(self, engine):
        """
        The control that makes the tests above mean something.

        This is the mistake the design avoids, demonstrated: pushing a connection straight
        into the host graph works right up until anything else changes, which is the worst
        possible failure mode — it passes a smoke test and breaks when a user drags a cable.
        """
        source = engine.create_virtual_input('guitar')
        other_in = engine.create_virtual_input('keys')
        other_out = engine.create_virtual_output('headphones')

        side_channel = network_node('side_channel')
        engine.host.apply_graph(
            engine.host.graph_holder.current().with_connection(
                Connection(source=engine._node_for(source), dest=side_channel)
            )
        )
        assert self._network_edges(engine) == ['network:side_channel']

        engine.create_routing(other_in, other_out)

        assert self._network_edges(engine) == [], (
            "the premise of this design is wrong: an out-of-band edge survived a rebuild"
        )


class TestSendTeardown:
    def test_disable_removes_the_route_and_the_matrix_entry(self, engine):
        source = engine.create_virtual_input('guitar')
        engine.start_udp_transport('127.0.0.1', 0)
        engine.send_device_audio_to_network(source, transport='udp')
        sink_id = engine.list_network_sends()[0]['sink_id']

        assert engine.disable_network_send(source) is True

        assert engine.list_network_sends() == []
        assert (source, sink_id) not in engine.routing_matrix.connections
        assert not [
            c for c in engine.host.graph_holder.current().connections
            if c.dest.kind == 'network'
        ]
        assert engine._send_worker.route_count == 0
        assert engine._send_worker.is_running is False

    def test_disabling_something_that_was_not_sending_reports_failure(self, engine):
        bus = engine.create_virtual_input('guitar')

        assert engine.disable_network_send(bus) is False

    def test_removing_the_bus_takes_its_network_send_with_it(self, engine):
        """
        Left behind, the worker would keep reading a route whose source is gone and keep
        reporting healthy counts for audio going nowhere.
        """
        source = engine.create_virtual_input('guitar')
        engine.start_udp_transport('127.0.0.1', 0)
        engine.send_device_audio_to_network(source, transport='udp')

        assert engine.remove_virtual_device(source) is True

        assert engine.list_network_sends() == []
        assert engine._send_worker.route_count == 0

    def test_clearing_the_patchbay_clears_network_sends_too(self, engine):
        source = engine.create_virtual_input('guitar')
        engine.start_udp_transport('127.0.0.1', 0)
        engine.send_device_audio_to_network(source, transport='udp')

        engine.clear_all_routing()

        assert engine.list_network_sends() == []
        assert engine.routing_matrix.connections == {}

    def test_stopping_the_transport_stops_every_thread(self, sender, receiver):
        source = sender.create_virtual_input('guitar')
        destination = receiver.create_virtual_input('from-network')
        monitor = receiver.create_virtual_output('monitor')
        receiver.create_routing(destination, monitor)

        link(sender, receiver, source, destination)
        sender._send_worker.start()

        receiver.stop_udp_transport()
        sender.stop_udp_transport()

        assert sender._send_worker.is_running is False
        assert sender.udp_transport.is_running is False
        assert receiver.udp_transport.is_running is False
        assert receiver.get_network_statistics()['udp']['receive'] == {}


class TestRefusalsAreHonest:
    def test_tcp_send_says_it_is_unwired_rather_than_reporting_success(self, engine):
        """
        This method used to log a warning and return None, so a REST caller got a success
        response for audio that never moved. Refusing is the honest answer.
        """
        source = engine.create_virtual_input('guitar')

        ok, message = engine.send_device_audio_to_network(source, transport='tcp')

        assert ok is False
        assert 'not wired' in message
        assert engine.list_network_sends() == []

    def test_an_unknown_source_is_refused(self, engine):
        ok, message = engine.send_device_audio_to_network(999_999, transport='udp')

        assert ok is False
        assert 'unknown' in message.lower()

    def test_sending_twice_from_one_device_is_refused(self, engine):
        source = engine.create_virtual_input('guitar')
        engine.start_udp_transport('127.0.0.1', 0)

        assert engine.send_device_audio_to_network(source, transport='udp')[0] is True

        ok, message = engine.send_device_audio_to_network(source, transport='udp')

        assert ok is False
        assert 'already sending' in message

    def test_a_send_with_no_peers_says_nothing_is_leaving_the_machine(self, engine):
        """
        The difference between "streaming" and "encoding audio into a void". Reporting a
        healthy send here would be a packet count for audio nobody receives.
        """
        source = engine.create_virtual_input('guitar')
        engine.start_udp_transport('127.0.0.1', 0)

        ok, message = engine.send_device_audio_to_network(source, transport='udp')

        assert ok is True
        assert 'no peers registered' in message

        for _ in range(3):
            engine.write_to_bus(source, sine(BLOCK))
            engine._send_worker.tick()

        route = next(iter(
            engine.get_network_statistics()['udp']['send']['routes'].values()
        ))
        assert route['packets_sent'] == 0
        assert route['packets_undeliverable'] > 0

    def test_receiving_into_a_hardware_device_is_refused(self, engine):
        ok, message = engine.register_network_receive(0, transport='udp')

        assert ok is False
        assert 'not a bus' in message

    def test_an_unknown_transport_is_refused(self, engine):
        bus = engine.create_virtual_input('from-network')

        ok, message = engine.register_network_receive(bus, transport='quic')

        assert ok is False
        assert 'quic' in message

    def test_receiving_twice_on_one_bus_is_refused(self, engine):
        bus = engine.create_virtual_input('from-network')
        engine.start_udp_transport('127.0.0.1', 0)

        assert engine.register_network_receive(bus, transport='udp')[0] is True

        ok, message = engine.register_network_receive(bus, transport='udp')

        assert ok is False
        assert 'already receiving' in message

    def test_tcp_receive_still_works_the_way_it_always_did(self, engine):
        """The existing behaviour is extended, not replaced."""
        bus = engine.create_virtual_input('from-network')

        ok, _message = engine.register_network_receive(bus)

        assert ok is True
        assert bus in engine.network_router.receive_callbacks
        assert engine._udp_receives == {}


class TestStatistics:
    def test_tcp_keys_stay_at_the_top_level(self, engine):
        """Existing callers read these where they are; moving them would break them."""
        stats = engine.get_network_statistics()

        for key in ('packets_sent', 'packets_received', 'bytes_sent', 'quality'):
            assert key in stats

    def test_udp_is_additive_and_reports_nothing_before_it_runs(self, engine):
        stats = engine.get_network_statistics()['udp']

        assert stats['transport']['running'] is False
        assert stats['transport']['bound_port'] is None
        assert stats['transport']['opus_available'] is False
        assert stats['send'] == {'running': False, 'routes': {}}
        assert stats['receive'] == {}

    def test_a_jitter_buffer_is_null_until_a_packet_has_arrived(self, engine):
        """
        Not an empty stat block. Zeros here would read as a healthy link delivering
        silence, when in fact nothing has been measured at all.
        """
        bus = engine.create_virtual_input('from-network')
        engine.start_udp_transport('127.0.0.1', 0)
        engine.register_network_receive(bus, transport='udp')

        entry = engine.get_network_statistics()['udp']['receive'][str(bus)]

        assert entry['jitter_buffer'] is None
        assert entry['packets_received'] == 0
        assert entry['playout_alive'] is True

    def test_a_malformed_datagram_is_counted_against_the_device(self, engine):
        import socket

        bus = engine.create_virtual_input('from-network')
        engine.start_udp_transport('127.0.0.1', 0)
        engine.register_network_receive(bus, transport='udp')

        host, port = engine.udp_transport.bound_address
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            # A well-formed header whose payload cannot possibly hold the audio it
            # claims: rejected at decode, so it reaches the per-device counter.
            packet = UdpPacket(
                codec=CODEC_PCM_FLOAT32, channels=2, flags=0, sample_rate=RATE,
                device_id=bus, sequence=0, timestamp_us=0, frame_count=64,
                payload=b"\x00" * 8,
            )
            sock.sendto(packet.pack(), (host, port))

            deadline = time.monotonic() + 3.0
            while time.monotonic() < deadline:
                entry = engine.get_network_statistics()['udp']['receive'][str(bus)]
                if entry['packets_rejected']:
                    break
                time.sleep(0.01)
        finally:
            sock.close()

        entry = engine.get_network_statistics()['udp']['receive'][str(bus)]
        assert entry['packets_rejected'] == 1
        assert entry['jitter_buffer'] is None, "a rejected packet must not create a buffer"
        assert entry['error'] is not None

    def test_quality_applies_to_both_transports(self, engine):
        ok, message = engine.set_network_quality('lossless')

        assert ok, message
        assert engine.network_router.quality.value == 'lossless'
        assert engine.udp_transport.quality.value == 'lossless'

    def test_an_unknown_quality_is_refused_with_the_real_options(self, engine):
        ok, message = engine.set_network_quality('excellent')

        assert ok is False
        assert 'lossless' in message
