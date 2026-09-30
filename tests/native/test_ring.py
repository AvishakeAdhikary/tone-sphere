"""
The native SPSC ring, fed with audio rather than counters: every test compares the
samples that come out against the samples that went in, exactly.
"""

import threading

import numpy as np
import pytest

from tests.signals import sine, white_noise
from tonesphere.native import NativeRing


def test_capacity_rounds_up_to_a_power_of_two():
    assert NativeRing(1000, 2).capacity == 1024
    assert NativeRing(1024, 2).capacity == 1024


def test_empty_ring_reads_nothing():
    ring = NativeRing(256, 2)
    assert ring.available() == 0
    assert ring.read(64).shape == (0, 2)


def test_wraparound_preserves_every_sample():
    """Chunk sizes that do not divide the capacity force every write and read to wrap."""
    ring = NativeRing(1024, 2)
    signal = sine(48000, 997.0, amplitude=0.8)
    out = []
    position = 0
    while position < len(signal):
        chunk = signal[position:position + 300]
        assert ring.write(chunk) == len(chunk)
        position += len(chunk)
        out.append(ring.read(300))
    assert np.array_equal(np.concatenate(out), signal)


def test_full_ring_drops_the_newest_frames_and_says_how_many_it_took():
    ring = NativeRing(512, 1)
    signal = white_noise(800, channels=1)
    assert ring.write(signal) == 512
    assert ring.write(signal[:10]) == 0, "a full ring must refuse, not overwrite unread audio"
    assert np.array_equal(ring.read(1000), signal[:512])


def test_multichannel_frames_stay_together():
    ring = NativeRing(64, 8)
    block = np.arange(40 * 8, dtype=np.float32).reshape(40, 8)
    ring.write(block[:30])
    first = ring.read(20)
    ring.write(block[30:])
    rest = ring.read(64)
    assert np.array_equal(np.concatenate([first, rest]), block)


@pytest.mark.parametrize("chunk", [1, 37, 256])
def test_concurrent_producer_and_consumer_deliver_sample_exact_audio(chunk):
    """
    A real producer thread and a real consumer thread: ctypes releases the GIL for the
    duration of each native call, so the two ends genuinely overlap. The ring is small
    relative to the signal so it fills and empties thousands of times.
    """
    ring = NativeRing(256, 2)
    signal = sine(200_000, 1000.0, amplitude=0.9) + white_noise(200_000, amplitude=0.05)
    received = []
    done = threading.Event()

    def produce():
        position = 0
        while position < len(signal):
            written = ring.write(signal[position:position + chunk])
            position += written
        done.set()

    def consume():
        total = 0
        while total < len(signal):
            block = ring.read(chunk)
            if len(block):
                received.append(block)
                total += len(block)
            elif done.is_set() and ring.available() == 0:
                break

    producer = threading.Thread(target=produce)
    consumer = threading.Thread(target=consume)
    producer.start()
    consumer.start()
    producer.join(timeout=60)
    consumer.join(timeout=60)

    out = np.concatenate(received)
    assert len(out) == len(signal)
    assert np.array_equal(out, signal), "a frame was lost, duplicated or reordered"
