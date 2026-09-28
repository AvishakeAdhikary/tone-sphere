"""
The control plane's RoutingGraph compiled to a native plan, then run: the same graph the
UI and API edit, proven to carry a signal through the native engine.
"""

import numpy as np
import pytest

from tests.signals import RATE, peak, sine
from tonesphere.engine.graph import Connection, RoutingGraph, bus_node, device_node, network_node
from tonesphere.native import EQ, NativeEngine
from tonesphere.native.plan import INPUT, OUTPUT, Endpoint, PlanCompiler

BLOCK = 256
IFACE = device_node('ASIO::Interface')
HEADPHONES = device_node('WASAPI::Headphones')
MIX = bus_node('mix')


def run_plan(plan, feeds, blocks=4):
    with NativeEngine(RATE, BLOCK) as engine:
        engine.apply_plan(plan.nodes, plan.routes, plan.inserts)
        engine.set_master_gain(plan.master_gain)
        outputs = {native: 2 for (node, role), native in plan.native_ids.items()
                   if role == OUTPUT and node.kind != 'bus'}
        collected = {k: [] for k in outputs}
        for b in range(blocks):
            feed = {native: signal[b * BLOCK:(b + 1) * BLOCK] for native, signal in feeds.items()}
            for k, v in engine.process(feed, outputs).items():
                collected[k].append(v)
        return {k: np.concatenate(v) for k, v in collected.items()}


def endpoints():
    return {
        IFACE: Endpoint(input_channels=2, output_channels=2),
        HEADPHONES: Endpoint(output_channels=2),
        MIX: Endpoint(input_channels=2, output_channels=2),
    }


def test_a_device_routed_to_itself_is_the_monitoring_path():
    """The guitar path: interface input straight to the same interface's output."""
    compiler = PlanCompiler()
    graph = RoutingGraph(connections=(Connection(IFACE, IFACE),))
    plan = compiler.compile(graph, endpoints())
    source, sink = plan.native_ids[(IFACE, INPUT)], plan.native_ids[(IFACE, OUTPUT)]
    assert source != sink

    tone = sine(BLOCK * 4, amplitude=0.5)
    out = run_plan(plan, {source: tone})[sink]
    assert np.array_equal(out[BLOCK:], tone[BLOCK:])


def test_device_through_a_bus_to_another_device():
    compiler = PlanCompiler()
    graph = RoutingGraph(connections=(Connection(IFACE, MIX), Connection(MIX, HEADPHONES, gain=0.5)))
    plan = compiler.compile(graph, endpoints())
    tone = sine(BLOCK * 4)
    out = run_plan(plan, {plan.native_ids[(IFACE, INPUT)]: tone})[plan.native_ids[(HEADPHONES, OUTPUT)]]
    assert np.allclose(out[BLOCK:], tone[BLOCK:] * 0.5, atol=1e-7)


def test_solo_silences_every_other_source():
    other = device_node('WASAPI::Mic')
    eps = endpoints() | {other: Endpoint(input_channels=2)}
    graph = RoutingGraph(
        connections=(Connection(IFACE, HEADPHONES), Connection(other, HEADPHONES)),
        soloed=frozenset({IFACE}),
    )
    plan = PlanCompiler().compile(graph, eps)
    a = sine(BLOCK * 4, 440.0, amplitude=0.3)
    b = sine(BLOCK * 4, 1000.0, amplitude=0.3)
    out = run_plan(plan, {plan.native_ids[(IFACE, INPUT)]: a, plan.native_ids[(other, INPUT)]: b})
    heard = out[plan.native_ids[(HEADPHONES, OUTPUT)]]
    assert np.allclose(heard[BLOCK:], a[BLOCK:], atol=1e-7), "only the soloed source may be heard"


def test_mute_pan_invert_and_master_gain_are_compiled():
    graph = RoutingGraph(connections=(Connection(IFACE, HEADPHONES, invert=True),), master_gain=0.5)
    plan = PlanCompiler().compile(graph, endpoints())
    tone = sine(BLOCK * 4)
    out = run_plan(plan, {plan.native_ids[(IFACE, INPUT)]: tone})[plan.native_ids[(HEADPHONES, OUTPUT)]]
    assert np.allclose(out[2 * BLOCK:], -tone[2 * BLOCK:] * 0.5, atol=1e-7)

    muted = PlanCompiler().compile(RoutingGraph(connections=(Connection(IFACE, HEADPHONES, muted=True),)), endpoints())
    assert peak(run_plan(muted, {muted.native_ids[(IFACE, INPUT)]: tone})[muted.native_ids[(HEADPHONES, OUTPUT)]]) == 0


def test_native_ids_are_stable_across_compiles():
    """State lives on native ids; renumbering would reset every fader and filter."""
    compiler = PlanCompiler()
    first = compiler.compile(RoutingGraph(connections=(Connection(IFACE, MIX),)), endpoints())
    second = compiler.compile(
        RoutingGraph(connections=(Connection(MIX, HEADPHONES), Connection(IFACE, MIX))), endpoints())
    assert first.native_ids[(IFACE, INPUT)] == second.native_ids[(IFACE, INPUT)]
    assert first.native_ids[(MIX, INPUT)] == second.native_ids[(MIX, INPUT)]


def test_network_endpoints_become_rings():
    send = network_node('udp:1')
    eps = endpoints() | {send: Endpoint(output_channels=2, ring_frames=BLOCK * 8, limiter=False)}
    plan = PlanCompiler().compile(RoutingGraph(connections=(Connection(IFACE, send),)), eps)
    sink = next(n for n in plan.nodes if n.id == plan.native_ids[(send, OUTPUT)])
    assert sink.ring_frames == BLOCK * 8 and sink.limiter is False


def test_inserts_attach_to_the_right_side():
    plan = PlanCompiler().compile(
        RoutingGraph(connections=(Connection(IFACE, HEADPHONES),)), endpoints(),
        inserts={(IFACE, INPUT): [(0, EQ, False)]},
    )
    assert [(i.node, i.slot, i.type) for i in plan.inserts] == [(plan.native_ids[(IFACE, INPUT)], 0, EQ)]


def test_undescribed_endpoints_and_impossible_directions_are_errors():
    with pytest.raises(KeyError):
        PlanCompiler().compile(RoutingGraph(connections=(Connection(IFACE, device_node('nowhere')),)), endpoints())
    with pytest.raises(ValueError):
        PlanCompiler().compile(RoutingGraph(connections=(Connection(HEADPHONES, IFACE),)), endpoints())
