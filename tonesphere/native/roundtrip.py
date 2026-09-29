"""
Measure round-trip latency: send a signal out, capture it back, time the difference.

This is the only thing in ToneSphere allowed to produce a latency labelled "measured".

The engine tees the stimulus: in the same block it goes to the output device and to a
monitor ring, while the input device's audio goes to a capture ring. Both rings are
filled by the same `run_block`, so they share one frame clock, and the offset that best
aligns capture with monitor is the whole round trip — engine output, driver, DAC, the air
or cable in between, ADC, driver, and any clock-boundary cushion on the input side, which
is real latency too.

The offset is found by GCC-PHAT (cross-correlation with phase transform), which gives a
sharp peak even through a coloured path such as a laptop speaker and microphone. A peak
that does not stand well clear of everything else is not a measurement: the result is
then None, rendered `--`, and the note says why.
"""

import time
from dataclasses import dataclass

import numpy as np

from tonesphere.native import NativeEngine, Node, Route
from tonesphere.native.wasapi import StreamSpec, endpoints

STIMULUS, OUTPUT, MONITOR, INPUT, CAPTURED = 1, 2, 3, 4, 5

# The peak must be this many times the largest correlation value outside it. Measured
# acoustic paths through a laptop speaker and microphone give 10-50; no path at all gives
# under 2.
CONFIDENCE_THRESHOLD = 4.0


@dataclass(frozen=True)
class RoundTrip:
    measured_ms: float | None
    measured_frames: int | None
    confidence: float
    nominal_ms: float           # engine block / rate: arithmetic about ToneSphere's own share
    reported_ms: float | None   # what the drivers report for the two streams, summed
    sample_rate: int
    block: int
    output: str
    input: str
    note: str


def sweep(rate: int, seconds: float, low: float, high: float, level_db: float) -> np.ndarray:
    """An exponential sine sweep with 5 ms raised-cosine edges, so it starts and stops without a click."""
    t = np.arange(int(rate * seconds)) / rate
    k = np.log(high / low)
    wave = np.sin(2 * np.pi * low * seconds / k * (np.exp(t / seconds * k) - 1))
    edge = int(rate * 0.005)
    ramp = 0.5 - 0.5 * np.cos(np.linspace(0, np.pi, edge))
    wave[:edge] *= ramp
    wave[-edge:] *= ramp[::-1]
    return (wave * 10 ** (level_db / 20)).astype(np.float32)


def gcc_phat(captured: np.ndarray, reference: np.ndarray, max_lag: int) -> tuple[int, float]:
    """Lag (in frames, capture after reference) and confidence of the best alignment."""
    n = 1 << int(np.ceil(np.log2(len(captured) + len(reference))))
    cross = np.fft.rfft(captured, n) * np.conj(np.fft.rfft(reference, n))
    cross /= np.maximum(np.abs(cross), 1e-12)
    corr = np.fft.irfft(cross, n)[:max_lag]
    lag = int(np.argmax(corr))
    guard = max(8, max_lag // 200)
    outside = np.concatenate([corr[:max(0, lag - guard)], corr[lag + guard:]])
    rival = float(np.max(np.abs(outside))) if len(outside) else 0.0
    return lag, float(corr[lag] / rival) if rival > 0 else float('inf')


def measure(output_id: str, input_id: str, *, input_kind: str = 'capture', sample_rate: int = 48000,
            block: int = 480, level_db: float = -24.0, exclusive: bool = False,
            max_latency_ms: float = 500.0) -> RoundTrip:
    """
    Play a sweep on `output_id` and listen on `input_id` (`input_kind` 'capture' for a
    microphone or line input, 'loopback' to listen to the output endpoint itself — a digital
    path, useful to check the method). The output stream is the master clock; the input
    stream crosses into it through the engine's drift-corrected ring, and that cushion is
    part of what is measured.
    """
    by_id = {e.id: e for e in endpoints()}
    out_ep, in_ep = by_id.get(output_id), by_id.get(input_id)
    if out_ep is None or in_ep is None:
        raise ValueError("unknown endpoint id")
    in_channels = out_ep.mix_channels if input_kind == 'loopback' else in_ep.mix_channels

    stimulus = sweep(sample_rate, 0.5, 150.0, min(12000.0, sample_rate * 0.45), level_db)
    lead = np.zeros(int(sample_rate * 0.4), np.float32)
    signal = np.concatenate([lead, stimulus, np.zeros(int(sample_rate * max_latency_ms / 1000) + sample_rate // 4,
                                                       np.float32)])
    stereo = np.repeat(signal.reshape(-1, 1), out_ep.mix_channels, axis=1)
    frames = len(signal)

    with NativeEngine(sample_rate, block) as engine:
        engine.apply_plan(
            [Node.source(STIMULUS, out_ep.mix_channels, ring_frames=frames + sample_rate),
             Node.sink(OUTPUT, out_ep.mix_channels),
             Node.sink(MONITOR, 1, ring_frames=frames + sample_rate * 2),
             Node.source(INPUT, in_channels),
             Node.sink(CAPTURED, 1, ring_frames=frames + sample_rate * 2)],
            [Route(STIMULUS, OUTPUT), Route(STIMULUS, MONITOR), Route(INPUT, CAPTURED)],
        )
        # The monitor tee and the capture both mix to mono: averaging the channels changes
        # nothing about where the stimulus is in time.
        engine.port_write(STIMULUS, stereo)
        engine.start_wasapi([
            StreamSpec(OUTPUT, 'render', out_ep.mix_channels, output_id, exclusive=exclusive),
            StreamSpec(INPUT, input_kind, in_channels, input_id if input_kind == 'capture' else output_id),
        ], master=0)
        monitor, captured = [], []
        deadline = time.time() + frames / sample_rate + 0.5
        while time.time() < deadline:
            monitor.append(engine.port_read(MONITOR, sample_rate))
            captured.append(engine.port_read(CAPTURED, sample_rate))
            time.sleep(0.02)
        status = engine.stream_status()
        engine.stop_backend()
        monitor.append(engine.port_read(MONITOR, sample_rate * 4))
        captured.append(engine.port_read(CAPTURED, sample_rate * 4))

    mon = np.concatenate(monitor)[:, 0].astype(np.float64)
    cap = np.concatenate(captured)[:, 0].astype(np.float64)
    length = min(len(mon), len(cap))
    mon, cap = mon[:length], cap[:length]

    reported = [s['reported_latency_ms'] for s in status]
    reported_ms = sum(reported) if all(r is not None for r in reported) else None
    listened = in_ep.name if input_kind == 'capture' else f"loopback of {out_ep.name}"
    common = dict(nominal_ms=block / sample_rate * 1000, reported_ms=reported_ms, sample_rate=sample_rate,
                  block=block, output=out_ep.name, input=listened)

    failed = [s for s in status if s['state'] != 'running']
    if failed:
        return RoundTrip(None, None, 0.0, note=f"a stream failed: {failed[0]['message']}", **common)
    if np.max(np.abs(cap)) < 1e-6:
        return RoundTrip(None, None, 0.0, note="nothing was captured: the input heard silence", **common)

    lag, confidence = gcc_phat(cap, mon, int(sample_rate * max_latency_ms / 1000))
    if confidence < CONFIDENCE_THRESHOLD:
        return RoundTrip(None, None, confidence,
                         note=f"no clear path from output to input (confidence {confidence:.1f} < "
                              f"{CONFIDENCE_THRESHOLD}); a loopback cable, or a microphone that can hear the speaker, "
                              f"is needed", **common)
    return RoundTrip(lag / sample_rate * 1000, lag, confidence, note="measured by GCC-PHAT on an exponential sweep",
                     **common)
