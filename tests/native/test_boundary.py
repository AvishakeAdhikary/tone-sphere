"""
The device boundary without a device: sample-format conversion and the drift resampler
that carries audio between two device clocks.
"""

import numpy as np
import pytest

from tests.signals import RATE, dominant_frequency, rms, sine
from tonesphere.native import NativeResampler, _abi, convert, unconvert

# Worst round-trip error allowed: half a quantisation step, except that int32 carries
# more precision than float32 itself, so there the bound is float32's half-ulp near 1.0.
FORMATS = {
    'int16': (_abi.FORMAT_INT16, 2 ** -16),
    'int24': (_abi.FORMAT_INT24, 2 ** -24),
    'int32': (_abi.FORMAT_INT32, 2 ** -24),
    'float32': (_abi.FORMAT_FLOAT32, 0.0),
    'float64': (_abi.FORMAT_FLOAT64, 0.0),
}


class TestConversion:
    @pytest.mark.parametrize("name", FORMATS)
    def test_round_trip_is_within_half_a_quantisation_step(self, name):
        fmt, step = FORMATS[name]
        tone = sine(4800, 997.0, amplitude=0.9, channels=2).ravel()
        back = unconvert(fmt, convert(fmt, tone), tone.size)
        assert np.max(np.abs(back - tone)) <= step * 1.01

    def test_int16_has_the_right_scale_and_byte_order(self):
        raw = convert(_abi.FORMAT_INT16, np.array([0.5, -0.5, 1.0, -1.0], np.float32))
        assert np.frombuffer(raw, '<i2').tolist() == [16384, -16384, 32767, -32768]

    def test_int24_is_packed_little_endian(self):
        raw = convert(_abi.FORMAT_INT24, np.array([0.5], np.float32))
        assert int.from_bytes(raw, 'little', signed=True) == 4194304
        assert len(raw) == 3

    @pytest.mark.parametrize("name", ['int16', 'int24', 'int32'])
    def test_over_range_clips_instead_of_wrapping(self, name):
        """A plain cast wraps +1.2 to a large negative number: a full-scale click."""
        fmt, _ = FORMATS[name]
        back = unconvert(fmt, convert(fmt, np.array([1.2, -1.7], np.float32)), 2)
        assert back[0] == pytest.approx(1.0, abs=1e-4) and back[1] == pytest.approx(-1.0, abs=1e-4)

    def test_int24_reads_negative_values_with_their_sign(self):
        data = (-4194304).to_bytes(3, 'little', signed=True)  # -0.5 in 24-bit
        assert unconvert(_abi.FORMAT_INT24, data, 1)[0] == pytest.approx(-0.5)


class TestDriftResampler:
    BLOCK = 256

    def run(self, produce_per_block: float, blocks: int = 3000, target: int = 512):
        """
        A producer delivering `produce_per_block` frames per consumer block on average — a
        device clock that is fast or slow against the consumer's — through the resampler.
        """
        resampler = NativeResampler(2, self.BLOCK, target, 8192)
        tone = sine(int(produce_per_block * blocks) + 4096, 1000.0, amplitude=0.5)
        produced = 0.0
        position = 0
        out, fills, missing = [], [], 0
        for _ in range(blocks):
            produced += produce_per_block
            n = int(produced) - position
            resampler.push(tone[position:position + n])
            position += n
            block, lost = resampler.pull(self.BLOCK)
            missing += lost
            out.append(block)
            fills.append(resampler.fill)
        return np.concatenate(out), np.array(fills), missing, resampler.ratio

    def test_matched_clocks_pass_the_tone_unchanged(self):
        out, fills, missing, ratio = self.run(self.BLOCK)
        settled = out[self.BLOCK * 200:]
        assert missing == 0
        assert ratio == pytest.approx(1.0, abs=2e-4)
        assert dominant_frequency(settled) == pytest.approx(1000.0, abs=5.0)
        assert rms(settled) == pytest.approx(0.5 / np.sqrt(2), rel=2e-3)

    @pytest.mark.parametrize("ppm", [-3000, -500, -100, 100, 500, 3000])
    def test_a_drifting_clock_is_absorbed_without_dropouts(self, ppm):
        """
        Two crystals disagree by tens to hundreds of ppm. Uncorrected, 500 ppm fills or
        drains a 512-frame cushion in about 20 seconds; corrected, the fill level must stay
        bounded and the tone must come through whole, with no gap and no jump. 3000 ppm is
        beyond the worst device measured (a USB interface 0.26 % slow at 44.1 kHz).
        """
        out, fills, missing, ratio = self.run(self.BLOCK * (1 + ppm * 1e-6), blocks=6000)
        assert missing == 0, "the consumer ran dry"
        late = fills[3000:]
        assert late.max() - late.min() < 2 * self.BLOCK, "the fill level must settle, not wander"
        assert ratio == pytest.approx(1 + ppm * 1e-6, abs=1.5e-4), "the ratio must track the drift"
        settled = out[self.BLOCK * 3000:]
        assert dominant_frequency(settled) == pytest.approx(1000.0 * (1 + ppm * 1e-6), abs=5.0)
        step = np.max(np.abs(np.diff(settled[:, 0])))
        assert step < 2 * np.pi * 1000 / RATE * 0.5 * 1.05, "a discontinuity is a click"

    def test_waits_for_a_cushion_before_consuming(self):
        resampler = NativeResampler(2, self.BLOCK, 512, 4096)
        resampler.push(sine(300))
        block, missing = resampler.pull(self.BLOCK)
        assert missing == 0 and np.all(block == 0) and resampler.fill == 300
