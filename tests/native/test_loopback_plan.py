"""
The native host's plan for a loopback source, with no device opened: a source node the
width of its output, and — when nothing else in the plan has a clock — a silent render sink
on the same output to be one, since Windows sends a loopback nothing during silence.
"""

import sys

import pytest

from tonesphere.engine.devices import DeviceInfo, HostApi
from tonesphere.engine.graph import Connection, RoutingGraph, bus_node, device_node, loopback_node

pytestmark = pytest.mark.skipif(sys.platform != 'win32', reason="the native host is Windows-only")


def device(name: str, ins: int, outs: int) -> DeviceInfo:
    return DeviceInfo(index=0, name=name, host_api=HostApi.WASAPI, host_api_name='Windows WASAPI',
                      max_input_channels=ins, max_output_channels=outs, default_samplerate=48000,
                      default_low_input_latency_ms=0.0, default_low_output_latency_ms=0.0,
                      default_high_input_latency_ms=0.0, default_high_output_latency_ms=0.0,
                      endpoint_id=f'{{0.0.0.00000000}}.{name}')


@pytest.fixture
def host():
    from tonesphere.engine.native_host import NativeHost

    h = NativeHost(48000, 256)
    h._devices = [device('speakers', 0, 2), device('mic', 1, 0)]
    h.create_bus('b', 2)
    yield h
    h.cleanup()


def keys(host):
    return {d.name: d.key for d in host._devices}


def test_a_loopback_alone_gets_a_silent_render_clock_on_its_own_output(host):
    speakers = keys(host)['speakers']
    problems = host.configure(RoutingGraph(connections=(Connection(loopback_node(speakers), bus_node('b')),)))
    assert problems == []
    assert host._loopback_sides == {speakers: host._compiler.native_id(loopback_node(speakers), 'in')}
    clock, ref = host._loopback_clock
    assert ref == speakers
    sinks = {n.id: n for n in host._plan.nodes}
    assert clock in sinks and not any(r.dest == clock for r in host._plan.routes), "the clock sink plays silence"


def test_a_loopback_beside_a_device_needs_no_clock_of_its_own(host):
    k = keys(host)
    graph = RoutingGraph(connections=(Connection(loopback_node(k['speakers']), bus_node('b')),
                                      Connection(device_node(k['mic']), device_node(k['speakers']))))
    assert host.configure(graph) == []
    assert host._loopback_clock is None
    assert (k['speakers'], 'loopback', host._loopback_sides[k['speakers']]) in host._wanted_streams()


def test_a_loopback_of_an_input_device_is_refused_as_not_available(host):
    mic = keys(host)['mic']
    problems = host.configure(RoutingGraph(connections=(Connection(loopback_node(mic), bus_node('b')),)))
    assert any('loopback' in p for p in problems)
