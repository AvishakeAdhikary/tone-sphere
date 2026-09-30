"""
Whole-system loopback as a routable source: the graph's rules, without hardware.

A loopback of output X is everything the system plays on X. Routing it anywhere is
capture; routing it back into X, directly or through buses, is a howl, and the graph refuses
it in either order of building the path. The audio itself is proved on real hardware in
`tests/hardware/test_loopback_source.py`.
"""

import pytest

from tonesphere.engine.graph import Connection, RoutingGraph, bus_node, device_node, loopback_node

X, Y = device_node('speakers'), device_node('headset')
LX = loopback_node('speakers')
A, B = bus_node('a'), bus_node('b')


def graph(*edges) -> RoutingGraph:
    return RoutingGraph(connections=tuple(Connection(source=s, dest=d) for s, d in edges))


@pytest.mark.parametrize('existing, source, dest', [
    ((), LX, X),
    (((LX, A),), A, X),
    (((A, X),), LX, A),
    (((LX, A), (B, X)), A, B),
    (((A, B), (B, X)), LX, A),
], ids=['direct', 'through-a-bus', 'bus-built-first', 'joining-two-halves', 'into-a-chain'])
def test_a_loopback_can_never_reach_its_own_output(existing, source, dest):
    assert graph(*existing).would_feedback(source, dest)


@pytest.mark.parametrize('existing, source, dest', [
    ((), LX, Y),
    (((LX, A),), A, Y),
    (((A, Y),), LX, A),
    ((), device_node('mic'), X),
], ids=['to-another-output', 'bus-to-another-output', 'into-a-bus-to-another-output', 'mic-to-speakers'])
def test_a_loopback_can_go_anywhere_else(existing, source, dest):
    assert not graph(*existing).would_feedback(source, dest)


def test_the_engine_refuses_a_loopback_as_a_destination():
    from tonesphere.core.engine import LOOPBACK_ID_BASE, AudioEngine

    engine = AudioEngine()
    engine.initialize()
    try:
        loops = [d for d in engine.get_devices() if d['origin'] == 'loopback']
        if not loops:
            pytest.skip("no WASAPI output to take a loopback of")
        bus = engine.create_virtual_input('x')
        ok, message = engine.create_routing(bus, loops[0]['id'])
        assert not ok and 'source' in message
        assert loops[0]['id'] >= LOOPBACK_ID_BASE and loops[0]['direction'] == 'input'
    finally:
        engine.cleanup()
