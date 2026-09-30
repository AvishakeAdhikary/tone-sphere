"""
Opus, end to end: libopus through ctypes, the UDP transport's per-stream encoder, and the
receiver's jitter buffer decoding in sequence with Opus's own loss concealment.

Opus is lossy, so nothing here is sample-exact. What is asserted is what a listener would
check: the tone comes back at its frequency and level, the error after aligning for the
codec's own delay is well below the signal, the packets are a small fraction of PCM's, and
a lost packet is concealed rather than dropped to silence.

Where libopus is absent the module skips with the reason; CI installs it on all three
platforms (built from source on Windows, the system package on Linux and macOS).
"""

import time

import numpy as np
import pytest

from tests.signals import dominant_frequency, rms, sine
from tonesphere.network import opus

pytestmark = pytest.mark.skipif(not opus.available(), reason=f"libopus unavailable: {opus.unavailable_reason()}")

RATE = 48000
FRAME = RATE // 100


def codec_round_trip(signal: np.ndarray) -> tuple[np.ndarray, list[int]]:
    encoder, decoder = opus.Encoder(RATE, signal.shape[1]), opus.Decoder(RATE, signal.shape[1])
    out, sizes = [], []
    for i in range(0, len(signal) - FRAME + 1, FRAME):
        packet = encoder.encode(signal[i:i + FRAME])
        sizes.append(len(packet))
        out.append(decoder.decode(packet))
    return np.concatenate(out), sizes


def align(reference: np.ndarray, decoded: np.ndarray) -> tuple[np.ndarray, np.ndarray, int]:
    """Undo the codec's delay (its lookahead) by cross-correlating on a noise-like signal."""
    n = 1 << int(np.ceil(np.log2(len(reference) + len(decoded))))
    corr = np.fft.irfft(np.fft.rfft(decoded, n) * np.conj(np.fft.rfft(reference, n)), n)
    lag = int(np.argmax(corr[:RATE // 10]))
    length = min(len(reference), len(decoded) - lag)
    return reference[:length], decoded[lag:lag + length], lag


def test_a_tone_comes_back_at_its_frequency_and_level_with_little_error():
    """
    The codec's delay is found on a noise-like signal, where the alignment is unambiguous,
    and must be Opus's documented 6.5 ms lookahead; the error is then measured on a pure
    tone at that delay. (On the noise itself a waveform SNR says little: Opus codes noise by
    its spectrum, not sample by sample, which is the point of a perceptual codec.)
    """
    noise = np.random.default_rng(4).standard_normal((RATE, 2)).astype(np.float32) * 0.2
    decoded, _ = codec_round_trip(noise)
    _, _, lag = align(noise[:, 0], decoded[:, 0])

    tone = sine(RATE * 2, 1000.0, amplitude=0.5)
    decoded, sizes = codec_round_trip(tone)
    length = len(decoded) - lag
    reference, heard = tone[:length, 0], decoded[lag:, 0]
    body = slice(RATE // 4, None)
    snr = 20 * np.log10(rms(reference[body]) / rms(heard[body] - reference[body]))
    level = 20 * np.log10(rms(heard[body]) / rms(reference[body]))
    pcm_bytes = FRAME * 2 * 4
    print(f"\nOpus {opus.version()}: delay {lag} frames ({lag / RATE * 1000:.2f} ms), 1 kHz SNR {snr:.1f} dB, "
          f"level {level:+.2f} dB, {np.mean(sizes):.0f} bytes per 10 ms against {pcm_bytes} of float32 PCM")
    assert lag == 312, "Opus's lookahead at 48 kHz is 2.5 ms + 4 ms = 312 frames"
    assert dominant_frequency(heard[body]) == pytest.approx(1000.0, abs=2.0)
    assert abs(level) < 1.0
    assert snr >= 30.0
    assert np.mean(sizes) < pcm_bytes / 10


def test_a_lost_packet_is_concealed_by_opus_not_dropped_to_silence():
    """Packets arrive one per slot, as a paced sender's do; one in ten goes missing."""
    from tonesphere.network.jitter_buffer import JitterBuffer

    tone = sine(RATE * 2, 440.0, amplitude=0.5)
    encoder = opus.Encoder(RATE, 2)
    buffer = JitterBuffer(frames_per_packet=FRAME, channels=2, target_latency_ms=30.0,
                          decoder=opus.Decoder(RATE, 2))
    rng = np.random.default_rng(9)
    lost, blocks = 0, []
    for k, i in enumerate(range(0, len(tone), FRAME)):
        packet = encoder.encode(tone[i:i + FRAME])
        if k > 10 and rng.random() < 0.10:
            lost += 1
        else:
            buffer.push(k, packet)
        block = buffer.pull()
        if block is not None:
            blocks.append(block)
    stats = buffer.statistics()
    played = np.concatenate(blocks[5:])[:, 0]
    silent = sum(1 for b in blocks[5:] if not np.any(b))
    level = 20 * np.log10(rms(played) / rms(tone[:, 0]))
    print(f"\n10% loss: {lost} packets withheld, {stats['codec_concealments']} concealed by Opus, "
          f"{silent} silent blocks, level {level:+.1f} dB")
    assert stats['codec_concealments'] == lost
    assert silent == 0
    assert stats['decode_errors'] == 0
    assert abs(level) < 3.0
    assert dominant_frequency(played) == pytest.approx(440.0, abs=3.0)


def test_opus_crosses_between_two_engines_over_udp():
    from tests.test_network_send_wiring import drain
    from tonesphere.core.engine import AudioEngine

    sender = AudioEngine(sample_rate=RATE, buffer_size=256, host_backend='portaudio')
    receiver = AudioEngine(sample_rate=RATE, buffer_size=4096, host_backend='portaudio')
    try:
        source = sender.create_virtual_input('guitar', channels=2)
        destination = receiver.create_virtual_input('from-network', channels=2)
        monitor = receiver.create_virtual_output('monitor', channels=2)
        assert receiver.create_routing(destination, monitor)[0]
        assert receiver.start_udp_transport('127.0.0.1', 0)[0]
        assert receiver.register_network_receive(destination, transport='udp', target_latency_ms=30.0,
                                                 jitter_mode='fixed')[0]
        ok, message = sender.set_network_quality('opus')
        assert ok, message
        assert sender.start_udp_transport('127.0.0.1', 0)[0]
        assert sender.send_device_audio_to_network(source, transport='udp')[0]
        sender.add_udp_peer('receiver', *receiver.udp_transport.bound_address)

        tone = sine(RATE * 2, 1000.0, amplitude=0.5)
        written, collected, started = 0, [], time.monotonic()
        while time.monotonic() - started < 2.5:
            due = min(len(tone), int((time.monotonic() - started) * RATE))
            if written < due:
                written += sender.write_to_bus(source, tone[written:due])
            piece = drain(receiver, destination, monitor)
            if piece is not None:
                collected.append(piece)
            time.sleep(0.001)
        stats = receiver.get_network_statistics()['udp']['receive'][str(destination)]
        sent = sender.get_network_statistics()['udp']['transport']
    finally:
        sender.cleanup()
        receiver.cleanup()

    heard = np.concatenate(collected)[:, 0]
    body = heard[np.nonzero(np.abs(heard) > 1e-3)[0][0] + RATE // 10:][:RATE]
    level = 20 * np.log10(rms(body) / rms(tone[:, 0]))
    per_packet = sent['bytes_sent'] / max(1, sent['packets_sent'])
    print(f"\nOpus over UDP: {stats['packets_received']} packets, {per_packet:.0f} bytes each, "
          f"level {level:+.2f} dB, lost {stats['jitter_buffer']['packets_lost']}")
    assert stats['packets_received'] > 150 and stats['packets_wrong_rate'] == 0
    assert dominant_frequency(body) == pytest.approx(1000.0, abs=2.0)
    assert abs(level) < 1.0
    assert sent['bytes_sent'] / sent['packets_sent'] < 400


def test_opus_is_refused_with_the_reason_where_it_cannot_run(monkeypatch):
    from tonesphere.core.engine import AudioEngine

    with pytest.raises(opus.OpusUnavailable, match="44100"):
        opus.Encoder(44100, 2)
    engine = AudioEngine(host_backend='portaudio')
    try:
        monkeypatch.setattr(opus, 'available', lambda: False)
        monkeypatch.setattr(opus, 'unavailable_reason', lambda: 'libopus not found (test)')
        ok, message = engine.set_network_quality('opus')
        assert not ok and 'libopus not found' in message
        assert engine.udp_transport.quality.value == 'high', "a refusal leaves PCM, and says so"
    finally:
        engine.cleanup()
