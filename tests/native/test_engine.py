"""
The native engine run offline through `ts_engine_process` — the same `run_block` a device
backend calls, with no device. Every test feeds a known signal and asserts what comes out.
"""

import math
import threading

import numpy as np
import pytest

from tests.signals import (
    RATE,
    assert_finite,
    channel_impulses,
    dominant_frequency,
    peak,
    sine,
)
from tonesphere.native import NativeEngine, NativeError, Node, Route, _abi

BLOCK = 256
SRC, SRC2, BUS, BUS2, OUT = 1, 2, 10, 11, 20


@pytest.fixture
def engine():
    with NativeEngine(RATE, BLOCK) as e:
        yield e


def run(engine, blocks, inputs, outputs):
    """Feed `blocks` consecutive blocks of each input signal; return the concatenated outputs."""
    collected = {node: [] for node in outputs}
    for b in range(blocks):
        feed = {node: signal[b * BLOCK:(b + 1) * BLOCK] for node, signal in inputs.items()}
        for node, block in engine.process(feed, outputs).items():
            collected[node].append(block)
    return {node: np.concatenate(parts) for node, parts in collected.items()}


def settled(signal):
    """Drop the first block: a new route fades in from silence over one block, by design."""
    return signal[BLOCK:]


class TestAudioMovesThroughTheGraph:
    def test_source_to_sink_is_sample_exact_after_the_fade_in(self, engine):
        engine.apply_plan([Node.source(SRC, 2), Node.sink(OUT, 2)], [Route(SRC, OUT)])
        tone = sine(BLOCK * 8)
        out = run(engine, 8, {SRC: tone}, {OUT: 2})[OUT]
        assert np.array_equal(settled(out), settled(tone))

    def test_a_new_route_fades_in_rather_than_clicking(self, engine):
        engine.apply_plan([Node.source(SRC, 1), Node.sink(OUT, 1)], [Route(SRC, OUT)])
        ones = np.ones((BLOCK, 1), dtype=np.float32)
        first = engine.process({SRC: ones}, {OUT: 1})[OUT][:, 0]
        assert first[0] < 0.01 and first[-1] == pytest.approx(1.0)
        assert np.all(np.diff(first) > 0), "fade-in must be monotonic"

    def test_device_to_bus_to_device_carries_the_tone(self, engine):
        """The legacy host's defect (a bus that never forwards) must not exist here."""
        engine.apply_plan(
            [Node.source(SRC, 2), Node.bus(BUS, 2), Node.sink(OUT, 2)],
            [Route(SRC, BUS), Route(BUS, OUT)],
        )
        tone = sine(BLOCK * 16)
        out = run(engine, 16, {SRC: tone}, {OUT: 2})[OUT]
        assert dominant_frequency(settled(out)) == pytest.approx(1000.0, abs=100.0)
        assert np.array_equal(settled(out), settled(tone))

    def test_a_chain_of_buses_is_ordered_whatever_order_the_plan_lists_them(self, engine):
        engine.apply_plan(
            [Node.sink(OUT, 2), Node.bus(BUS2, 2), Node.bus(BUS, 2), Node.source(SRC, 2)],
            [Route(BUS2, OUT), Route(BUS, BUS2), Route(SRC, BUS)],
        )
        tone = sine(BLOCK * 8)
        out = run(engine, 8, {SRC: tone}, {OUT: 2})[OUT]
        # Three new routes each fade in over the same first block; after it, exact.
        assert np.array_equal(out[BLOCK:], tone[BLOCK:])

    def test_two_sources_sum(self, engine):
        engine.apply_plan(
            [Node.source(SRC, 1), Node.source(SRC2, 1), Node.sink(OUT, 1)],
            [Route(SRC, OUT), Route(SRC2, OUT)],
        )
        a = sine(BLOCK * 4, 440.0, amplitude=0.3, channels=1)
        b = sine(BLOCK * 4, 1000.0, amplitude=0.2, channels=1)
        out = run(engine, 4, {SRC: a, SRC2: b}, {OUT: 1})[OUT]
        assert np.allclose(settled(out), settled(a + b), atol=1e-6)

    def test_fan_out_reaches_every_destination(self, engine):
        engine.apply_plan(
            [Node.source(SRC, 2), Node.sink(OUT, 2), Node.sink(OUT + 1, 2), Node.sink(OUT + 2, 2)],
            [Route(SRC, OUT), Route(SRC, OUT + 1), Route(SRC, OUT + 2)],
        )
        tone = sine(BLOCK * 4)
        outs = run(engine, 4, {SRC: tone}, {OUT: 2, OUT + 1: 2, OUT + 2: 2})
        for node, out in outs.items():
            assert np.array_equal(settled(out), settled(tone)), f"sink {node} lost the signal"

    def test_unclaimed_outputs_and_unrouted_sinks_are_silence_not_garbage(self, engine):
        engine.apply_plan([Node.source(SRC, 2), Node.sink(OUT, 2)], [])
        out = engine.process({SRC: sine(BLOCK)}, {OUT: 2, 99: 2})
        assert np.array_equal(out[OUT], np.zeros((BLOCK, 2), np.float32))
        assert np.array_equal(out[99], np.zeros((BLOCK, 2), np.float32))

    def test_no_plan_is_silence(self, engine):
        out = engine.process({SRC: sine(BLOCK)}, {OUT: 2})[OUT]
        assert np.array_equal(out, np.zeros((BLOCK, 2), np.float32))


class TestGainMuteInvert:
    def test_gain_scales_the_signal(self, engine):
        engine.apply_plan([Node.source(SRC, 2), Node.sink(OUT, 2)], [Route(SRC, OUT, gain=0.5)])
        tone = sine(BLOCK * 4)
        out = run(engine, 4, {SRC: tone}, {OUT: 2})[OUT]
        assert np.allclose(settled(out), settled(tone) * 0.5, atol=1e-7)

    def test_gain_change_ramps_linearly_over_one_block(self, engine):
        engine.apply_plan([Node.source(SRC, 1), Node.sink(OUT, 1)], [Route(SRC, OUT)])
        ones = np.ones((BLOCK, 1), dtype=np.float32)
        engine.process({SRC: ones}, {OUT: 1})
        engine.set_route_gain(SRC, OUT, 0.0)
        ramp = engine.process({SRC: ones}, {OUT: 1})[OUT][:, 0]
        after = engine.process({SRC: ones}, {OUT: 1})[OUT][:, 0]
        assert ramp[0] == pytest.approx(1.0 - 1.0 / BLOCK)
        assert ramp[-1] == pytest.approx(0.0, abs=1e-7)
        assert np.allclose(np.diff(ramp), -1.0 / BLOCK, atol=1e-6)
        assert np.array_equal(after, np.zeros(BLOCK, np.float32))

    def test_mute_silences_after_a_fade(self, engine):
        engine.apply_plan([Node.source(SRC, 2), Node.sink(OUT, 2)], [Route(SRC, OUT)])
        tone = sine(BLOCK * 4)
        run(engine, 2, {SRC: tone}, {OUT: 2})
        engine.set_route_muted(SRC, OUT, True)
        out = run(engine, 2, {SRC: tone}, {OUT: 2})[OUT]
        assert peak(out[BLOCK:]) == 0.0
        engine.set_route_muted(SRC, OUT, False)
        back = run(engine, 2, {SRC: tone}, {OUT: 2})[OUT]
        assert np.array_equal(back[BLOCK:], tone[BLOCK:2 * BLOCK])

    def test_a_route_muted_in_the_plan_carries_nothing(self, engine):
        engine.apply_plan([Node.source(SRC, 2), Node.sink(OUT, 2)], [Route(SRC, OUT, muted=True)])
        assert peak(run(engine, 3, {SRC: sine(BLOCK * 3)}, {OUT: 2})[OUT]) == 0.0

    def test_invert_flips_polarity(self, engine):
        engine.apply_plan([Node.source(SRC, 2), Node.sink(OUT, 2)], [Route(SRC, OUT, invert=True)])
        tone = sine(BLOCK * 3)
        out = run(engine, 3, {SRC: tone}, {OUT: 2})[OUT]
        assert np.array_equal(settled(out), -settled(tone))

    def test_inverted_and_straight_copies_cancel(self, engine):
        engine.apply_plan(
            [Node.source(SRC, 1), Node.bus(BUS, 1), Node.sink(OUT, 1)],
            [Route(SRC, BUS), Route(BUS, OUT), Route(SRC, OUT, invert=True)],
        )
        out = run(engine, 4, {SRC: sine(BLOCK * 4, channels=1)}, {OUT: 1})[OUT]
        assert peak(out[BLOCK:]) < 1e-6

    def test_master_gain_scales_every_sink_once(self, engine):
        engine.apply_plan(
            [Node.source(SRC, 2), Node.bus(BUS, 2), Node.sink(OUT, 2)],
            [Route(SRC, BUS), Route(BUS, OUT)],
        )
        engine.set_master_gain(0.25)
        tone = sine(BLOCK * 4)
        out = run(engine, 4, {SRC: tone}, {OUT: 2})[OUT]
        assert np.allclose(out[2 * BLOCK:], tone[2 * BLOCK:] * 0.25, atol=1e-7), \
            "master gain must apply at the sink, not once per hop"


class TestChannelMapping:
    def test_mono_to_stereo_centre_is_constant_power(self, engine):
        engine.apply_plan([Node.source(SRC, 1), Node.sink(OUT, 2)], [Route(SRC, OUT)])
        out = run(engine, 2, {SRC: np.ones((BLOCK * 2, 1), np.float32)}, {OUT: 2})[OUT]
        assert out[-1, 0] == pytest.approx(1 / math.sqrt(2), abs=1e-6)
        assert out[-1, 1] == pytest.approx(1 / math.sqrt(2), abs=1e-6)

    def test_mono_hard_left(self, engine):
        engine.apply_plan([Node.source(SRC, 1), Node.sink(OUT, 2)], [Route(SRC, OUT, pan=-1.0)])
        out = run(engine, 2, {SRC: np.ones((BLOCK * 2, 1), np.float32)}, {OUT: 2})[OUT]
        assert out[-1, 0] == pytest.approx(1.0, abs=1e-6)
        assert abs(out[-1, 1]) < 1e-6

    def test_stereo_balance_is_continuous_at_centre(self, engine):
        """The legacy host stepped 3 dB the moment a stereo pan left centre."""
        levels = []
        for pan in (0.0, 0.001, -0.001):
            engine.apply_plan([Node.source(SRC, 2), Node.sink(OUT, 2)], [Route(SRC, OUT, pan=pan)])
            out = run(engine, 3, {SRC: np.ones((BLOCK * 3, 2), np.float32)}, {OUT: 2})[OUT]
            levels.append(out[-1])
        for left, right in levels:
            assert left == pytest.approx(1.0, abs=1e-4) and right == pytest.approx(1.0, abs=1e-4)

    def test_stereo_balance_hard_right_removes_left_only(self, engine):
        engine.apply_plan([Node.source(SRC, 2), Node.sink(OUT, 2)], [Route(SRC, OUT, pan=1.0)])
        out = run(engine, 2, {SRC: np.ones((BLOCK * 2, 2), np.float32)}, {OUT: 2})[OUT]
        assert abs(out[-1, 0]) < 1e-6 and out[-1, 1] == pytest.approx(1.0)

    def test_stereo_to_mono_averages(self, engine):
        engine.apply_plan([Node.source(SRC, 2), Node.sink(OUT, 1)], [Route(SRC, OUT)])
        block = np.column_stack([np.full(BLOCK * 2, 0.6), np.full(BLOCK * 2, 0.2)]).astype(np.float32)
        out = run(engine, 2, {SRC: block}, {OUT: 1})[OUT]
        assert out[-1, 0] == pytest.approx(0.4, abs=1e-6)

    def test_equal_width_multichannel_keeps_every_channel_in_place(self, engine):
        engine.apply_plan([Node.source(SRC, 8), Node.sink(OUT, 8)], [Route(SRC, OUT)])
        silence = np.zeros((BLOCK, 8), np.float32)
        impulses = channel_impulses(BLOCK, 8, spacing=16)
        engine.process({SRC: silence}, {OUT: 8})
        out = engine.process({SRC: impulses}, {OUT: 8})[OUT]
        assert np.array_equal(out, impulses)

    def test_wider_source_truncates_and_narrower_source_zero_pads(self, engine):
        engine.apply_plan(
            [Node.source(SRC, 4), Node.sink(OUT, 2), Node.source(SRC2, 2), Node.sink(OUT + 1, 4)],
            [Route(SRC, OUT), Route(SRC2, OUT + 1)],
        )
        four = channel_impulses(BLOCK, 4, spacing=16)
        two = channel_impulses(BLOCK, 2, spacing=16)
        zeros4 = np.zeros((BLOCK, 4), np.float32)
        zeros2 = np.zeros((BLOCK, 2), np.float32)
        engine.process({SRC: zeros4, SRC2: zeros2}, {OUT: 2, OUT + 1: 4})
        out = engine.process({SRC: four, SRC2: two}, {OUT: 2, OUT + 1: 4})
        assert np.array_equal(out[OUT], four[:, :2])
        assert np.array_equal(out[OUT + 1][:, :2], two)
        assert np.array_equal(out[OUT + 1][:, 2:], zeros2)


class TestPlanValidation:
    def working_plan(self, engine):
        engine.apply_plan([Node.source(SRC, 2), Node.bus(BUS, 2), Node.bus(BUS2, 2), Node.sink(OUT, 2)],
                          [Route(SRC, BUS), Route(BUS, OUT)])

    def test_a_feedback_loop_is_refused_and_the_running_plan_survives(self, engine):
        self.working_plan(engine)
        with pytest.raises(NativeError) as refused:
            engine.apply_plan(
                [Node.source(SRC, 2), Node.bus(BUS, 2), Node.bus(BUS2, 2), Node.sink(OUT, 2)],
                [Route(SRC, BUS), Route(BUS, BUS2), Route(BUS2, BUS), Route(BUS, OUT)],
            )
        assert refused.value.code == _abi.ERR_CYCLE
        assert "feedback" in str(refused.value) and str(BUS) in str(refused.value)

        tone = sine(BLOCK * 4)
        out = run(engine, 4, {SRC: tone}, {OUT: 2})[OUT]
        assert np.array_equal(settled(out), settled(tone))

    def test_a_self_route_is_a_cycle(self, engine):
        with pytest.raises(NativeError) as refused:
            engine.apply_plan([Node.bus(BUS, 2)], [Route(BUS, BUS)])
        assert refused.value.code == _abi.ERR_CYCLE

    @pytest.mark.parametrize("nodes, routes, fragment", [
        ([Node.source(SRC, 2), Node.sink(OUT, 2)], [Route(OUT, SRC)], "sink cannot feed"),
        ([Node.source(SRC, 2), Node.source(SRC2, 2)], [Route(SRC, SRC2)], "source cannot be fed"),
        ([Node.source(SRC, 2)], [Route(SRC, 77)], "unknown node"),
        ([Node.source(SRC, 2), Node.source(SRC, 2)], [], "duplicate node"),
        ([Node.source(SRC, 0)], [], "channel count"),
        ([Node.source(SRC, 2), Node.sink(OUT, 2)], [Route(SRC, OUT), Route(SRC, OUT)], "duplicate route"),
        ([Node.source(SRC, 2), Node.sink(OUT, 2)], [Route(SRC, OUT, gain=float('nan'))], "non-finite"),
        ([Node.source(SRC, 2, ring_frames=16), Node.sink(OUT, 2)], [], "ring must hold"),
    ])
    def test_malformed_plans_are_refused_with_a_reason(self, engine, nodes, routes, fragment):
        with pytest.raises(NativeError) as refused:
            engine.apply_plan(nodes, routes)
        assert refused.value.code == _abi.ERR_INVALID
        assert fragment in str(refused.value)

    def test_process_refuses_an_oversized_block(self, engine):
        with pytest.raises(NativeError):
            engine.process({SRC: np.zeros((BLOCK + 1, 2), np.float32)}, {OUT: 2})


class TestPlanSwaps:
    def test_a_route_kept_across_a_swap_does_not_fade_in_again(self, engine):
        engine.apply_plan([Node.source(SRC, 2), Node.sink(OUT, 2)], [Route(SRC, OUT)])
        tone = sine(BLOCK * 6)
        run(engine, 2, {SRC: tone}, {OUT: 2})
        engine.apply_plan([Node.source(SRC, 2), Node.source(SRC2, 2), Node.sink(OUT, 2)],
                          [Route(SRC, OUT), Route(SRC2, OUT)])
        out = engine.process({SRC: tone[2 * BLOCK:3 * BLOCK], SRC2: np.zeros((BLOCK, 2), np.float32)}, {OUT: 2})
        assert np.array_equal(out[OUT], tone[2 * BLOCK:3 * BLOCK]), "an unchanged route must not dip on a swap"

    def test_plans_swap_while_another_thread_processes(self, engine):
        """
        The real situation: an audio thread runs blocks continuously while the control
        thread publishes plan after plan. Every block must stay finite and bounded, and the
        audio thread must end up running the last plan published.
        """
        nodes = [Node.source(SRC, 2), Node.bus(BUS, 2), Node.sink(OUT, 2)]
        engine.apply_plan(nodes, [Route(SRC, BUS), Route(BUS, OUT)])
        stop = threading.Event()
        problems = []
        tone = sine(BLOCK)

        def audio_thread():
            while not stop.is_set():
                out = engine.process({SRC: tone}, {OUT: 2})[OUT]
                if not np.all(np.isfinite(out)) or peak(out) > 0.5 + 1e-5:
                    problems.append(peak(out))

        worker = threading.Thread(target=audio_thread)
        worker.start()
        try:
            for i in range(300):
                gain = 0.25 + 0.75 * (i % 2)
                engine.apply_plan(nodes, [Route(SRC, BUS, gain=gain), Route(BUS, OUT)])
        finally:
            stop.set()
            worker.join(timeout=30)

        assert not problems, f"{len(problems)} bad blocks during swaps"
        final = engine.stats()['plan_generation']
        engine.process({SRC: tone}, {OUT: 2})
        assert engine.stats()['plan_generation'] >= final
        assert engine.stats()['plan_generation'] == 301


class TestRingPorts:
    def test_audio_written_to_a_source_ring_comes_out_of_the_sink(self, engine):
        engine.apply_plan([Node.source(SRC, 2, ring_frames=BLOCK * 8), Node.sink(OUT, 2)], [Route(SRC, OUT)])
        tone = sine(BLOCK * 6)
        assert engine.port_write(SRC, tone) == len(tone)
        silence = np.zeros((BLOCK, 2), np.float32)
        outs = [engine.process({99: silence}, {OUT: 2})[OUT] for _ in range(6)]
        assert np.array_equal(np.concatenate(outs)[BLOCK:], tone[BLOCK:])

    def test_a_starved_source_ring_is_counted_and_reported_once(self, engine):
        engine.apply_plan([Node.source(SRC, 2, ring_frames=BLOCK * 4), Node.sink(OUT, 2)], [Route(SRC, OUT)])
        engine.port_write(SRC, sine(BLOCK // 2))
        silence = np.zeros((BLOCK, 2), np.float32)
        for _ in range(5):
            engine.process({99: silence}, {OUT: 2})
        stats = engine.stats()
        assert stats['ring_underruns'] == BLOCK // 2 + 4 * BLOCK
        underruns = [e for e in engine.events() if e['code'] == _abi.EVENT_RING_UNDERRUN]
        assert len(underruns) == 1 and underruns[0]['arg0'] == SRC

    def test_audio_reaching_a_sink_ring_can_be_read_back(self, engine):
        engine.apply_plan([Node.source(SRC, 2), Node.sink(OUT, 2, ring_frames=BLOCK * 8)], [Route(SRC, OUT)])
        tone = sine(BLOCK * 4)
        run(engine, 4, {SRC: tone}, {})
        assert engine.port_available(OUT) == BLOCK * 4
        back = engine.port_read(OUT, BLOCK * 4)
        assert np.array_equal(back[BLOCK:], tone[BLOCK:])

    def test_a_full_sink_ring_counts_what_it_dropped(self, engine):
        engine.apply_plan([Node.source(SRC, 2), Node.sink(OUT, 2, ring_frames=BLOCK * 2)], [Route(SRC, OUT)])
        run(engine, 5, {SRC: sine(BLOCK * 5)}, {})
        assert engine.stats()['ring_overruns'] == 3 * BLOCK
        assert [e['code'] for e in engine.events()] == [_abi.EVENT_RING_OVERRUN]

    def test_queued_audio_survives_an_unrelated_plan_change(self, engine):
        ring_source = Node.source(SRC, 2, ring_frames=BLOCK * 8)
        engine.apply_plan([ring_source, Node.sink(OUT, 2)], [Route(SRC, OUT)])
        tone = sine(BLOCK * 4)
        engine.port_write(SRC, tone)
        engine.apply_plan([ring_source, Node.bus(BUS, 2), Node.sink(OUT, 2)], [Route(SRC, OUT)])
        assert engine.port_available(SRC) == BLOCK * 4

    def test_writing_to_a_node_that_is_not_a_ring_is_refused(self, engine):
        engine.apply_plan([Node.source(SRC, 2), Node.sink(OUT, 2)], [Route(SRC, OUT)])
        with pytest.raises(NativeError):
            engine.port_write(SRC, sine(BLOCK))


class TestSafetyAndMeasurement:
    def test_a_nan_never_reaches_an_output(self, engine):
        engine.apply_plan([Node.source(SRC, 2), Node.sink(OUT, 2)], [Route(SRC, OUT)])
        poisoned = sine(BLOCK)
        poisoned[10, 0] = np.nan
        out = engine.process({SRC: poisoned}, {OUT: 2})[OUT]
        assert_finite(out)
        assert peak(out) == 0.0
        assert [e['code'] for e in engine.events()] == [_abi.EVENT_NONFINITE]

    def test_meters_report_what_was_processed(self, engine):
        engine.apply_plan([Node.source(SRC, 2), Node.sink(OUT, 2)], [Route(SRC, OUT)])
        run(engine, 4, {SRC: sine(BLOCK * 4, amplitude=0.5)}, {OUT: 2})
        meter = engine.meter(OUT)
        assert meter['peak'] == pytest.approx(0.5, abs=1e-3)
        assert meter['rms'] == pytest.approx(0.5 / math.sqrt(2), rel=1e-2)
        assert meter['clipped'] is False

    def test_clipping_latches_until_reset(self, engine):
        engine.apply_plan([Node.source(SRC, 1), Node.sink(OUT, 1)], [Route(SRC, OUT)])
        run(engine, 2, {SRC: np.full((BLOCK * 2, 1), 1.2, np.float32)}, {OUT: 1})
        run(engine, 2, {SRC: np.full((BLOCK * 2, 1), 0.1, np.float32)}, {OUT: 1})
        assert engine.meter(OUT)['clipped'] is True
        engine.reset_meters()
        run(engine, 1, {SRC: np.full((BLOCK, 1), 0.1, np.float32)}, {OUT: 1})
        meter = engine.meter(OUT)
        assert meter['clipped'] is False and meter['peak'] == pytest.approx(0.1, abs=1e-6)

    def test_load_is_judged_per_block_not_against_the_last_block(self, engine):
        """
        A full block then a short remainder, as a device loop produces: the load must not
        divide the full block's time by the remainder's tiny period.
        """
        engine.apply_plan([Node.source(SRC, 2), Node.sink(OUT, 2)], [Route(SRC, OUT)])
        for _ in range(20):
            engine.process({SRC: sine(BLOCK)}, {OUT: 2})
            engine.process({SRC: sine(8)}, {OUT: 2})
        stats = engine.stats()
        assert stats['period_ns'] == 8 * 1_000_000_000 // RATE
        assert stats['processing_load'] < stats['callback_ns_max'] / stats['period_ns']

    def test_statistics_are_unknown_until_a_block_runs(self, engine):
        stats = engine.stats()
        assert stats['blocks'] == 0
        for key in ('callback_ns_min', 'callback_ns_max', 'callback_ns_mean', 'callback_ns_p99', 'processing_load'):
            assert stats[key] is None, f"{key} must be None, not a number nobody measured"

    def test_statistics_measure_real_callback_time(self, engine):
        engine.apply_plan([Node.source(SRC, 2), Node.bus(BUS, 2), Node.sink(OUT, 2)],
                          [Route(SRC, BUS), Route(BUS, OUT)])
        run(engine, 50, {SRC: sine(BLOCK * 50)}, {OUT: 2})
        stats = engine.stats()
        assert stats['blocks'] == 50
        assert 0 < stats['callback_ns_min'] <= stats['callback_ns_mean'] <= stats['callback_ns_max']
        assert stats['callback_ns_min'] <= stats['callback_ns_p99'] <= stats['callback_ns_max']
        assert stats['period_ns'] == BLOCK * 1_000_000_000 // RATE
        # Every block here is the same size, so the worst per-block load is exactly the
        # worst time over the one period, and the mean load is the mean time over it.
        assert stats['processing_load'] == pytest.approx(stats['callback_ns_max'] / stats['period_ns'], rel=1e-3)
        assert stats['mean_load'] == pytest.approx(stats['callback_ns_mean'] / stats['period_ns'], rel=1e-3)
        assert sum(stats['histogram']) == 50
        engine.reset_stats()
        run(engine, 3, {SRC: sine(BLOCK * 3)}, {OUT: 2})
        assert engine.stats()['blocks'] == 3

    def test_the_audio_thread_allocates_nothing(self, engine):
        """
        `rt_allocations` counts heap allocations this DLL makes on the audio thread. After
        the plan is published, running blocks — rings, buses, fan-out, gain ramps, meters,
        plan swaps observed mid-stream — must not allocate at all.
        """
        nodes = [Node.source(SRC, 2), Node.source(SRC2, 1, ring_frames=BLOCK * 4), Node.bus(BUS, 2),
                 Node.sink(OUT, 2), Node.sink(OUT + 1, 1, ring_frames=BLOCK * 4)]
        routes = [Route(SRC, BUS), Route(SRC2, BUS, pan=0.3), Route(BUS, OUT), Route(BUS, OUT + 1)]
        engine.apply_plan(nodes, routes)
        engine.reset_stats()
        tone = sine(BLOCK)
        for i in range(500):
            if i % 50 == 0:
                engine.apply_plan(nodes, routes)
            engine.set_route_gain(SRC, BUS, 0.5 + 0.5 * (i % 2))
            engine.port_write(SRC2, tone[:, :1])
            engine.process({SRC: tone}, {OUT: 2})
            engine.port_read(OUT + 1, BLOCK)
        stats = engine.stats()
        assert stats['blocks'] == 500
        assert stats['rt_allocations'] == 0
