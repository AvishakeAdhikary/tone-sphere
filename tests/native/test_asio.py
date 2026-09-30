"""
The ASIO host without an ASIO driver: what can be proven on any machine, including CI.

Loading and running a real driver is in tests/hardware/test_asio.py. Nothing here makes
ASIO "supported" — only a driver initialised with audio moving through its buffer switch
does.
"""

import numpy as np
import pytest

from tests.signals import sine
from tonesphere.native import NativeEngine, NativeError, Node, Route, asio

pytestmark = pytest.mark.skipif(not asio.available(), reason="tonesphere_asio.dll is not built (no ASIO SDK)")

LSB = {'int16': (16, 2, 16), 'int24': (17, 3, 24), 'int32': (18, 4, 32), 'float32': (19, 4, None),
       'float64': (20, 8, None), 'int32 lsb16': (24, 4, 16), 'int32 lsb18': (25, 4, 18),
       'int32 lsb20': (26, 4, 20), 'int32 lsb24': (27, 4, 24)}


@pytest.mark.parametrize("name", LSB)
def test_every_windows_sample_type_round_trips(name):
    sample_type, width, bits = LSB[name]
    tone = sine(4800, 997.0, amplitude=0.9, channels=1).ravel()
    back = asio.convert_in(sample_type, asio.convert_out(sample_type, tone, width), len(tone))
    tolerance = 0.0 if bits is None else max(2.0 ** -bits, 2.0 ** -24)
    assert np.max(np.abs(back - tone)) <= tolerance


@pytest.mark.parametrize("name", ['int32 lsb16', 'int32 lsb24'])
def test_aligned_types_put_the_sample_in_the_low_bits(name):
    """Int32LSB24 is a 24-bit value in a 32-bit word: 0.5 is 2^22, not 2^30."""
    sample_type, _, bits = LSB[name]
    raw = asio.convert_out(sample_type, np.array([0.5, -1.0], np.float32), 4)
    assert np.frombuffer(raw, '<i4').tolist() == [2 ** (bits - 2), -(2 ** (bits - 1))]


def test_over_range_clips():
    raw = asio.convert_out(18, np.array([1.5, -3.0], np.float32), 4)
    assert np.frombuffer(raw, '<i4').tolist() == [2 ** 31 - 1, -(2 ** 31)]


def test_big_endian_and_dsd_types_are_refused_not_misread():
    with pytest.raises(NativeError):
        asio.convert_in(2, b"\0" * 4, 1)  # ASIOSTInt32MSB


def test_listing_works_with_or_without_drivers():
    for d in asio.drivers():
        assert d.name and d.clsid.startswith('{')


def test_starting_a_driver_that_is_not_registered_fails_with_the_reason():
    with NativeEngine(48000, 256) as engine:
        engine.apply_plan([Node.source(1, 2), Node.sink(2, 2)], [Route(1, 2)])
        with pytest.raises(NativeError) as refused:
            asio.start(engine, "No Such ASIO Driver", input_node=1, inputs=(0, 1), output_node=2, outputs=(0, 1))
        assert "no ASIO driver named" in str(refused.value)
        # The engine is untouched: it still runs offline, which it would refuse with a backend attached.
        assert engine.process({1: sine(256)}, {2: 2})[2].shape == (256, 2)


def test_channels_without_a_node_are_refused():
    with NativeEngine(48000, 256) as engine:
        with pytest.raises(NativeError) as refused:
            asio.start(engine, "anything", inputs=(0,))
        assert "without an engine node" in str(refused.value)
