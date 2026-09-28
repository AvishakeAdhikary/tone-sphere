"""
Compile the control plane's `RoutingGraph` into a native execution plan.

The graph speaks in endpoints — a device, a bus, a network stream — and a device is one
node whether a route leaves it or enters it. The native plan speaks in directions: audio
enters at a SOURCE and leaves at a SINK. So a device used both ways becomes two native
nodes, its input and its output, and that is exactly why routing an interface's input to
its own output is monitoring rather than a loop.

Native node ids are allocated once per (endpoint, role) and kept for the compiler's
lifetime, because the engine carries state — a fader, a filter's memory, a ring's queued
audio — by node id across plan swaps. Renumbering on every compile would silently reset
all of it.
"""

from dataclasses import dataclass, field

from tonesphere.engine.graph import NodeId, RoutingGraph
from tonesphere.native import Insert, Node, Route

INPUT = 'in'
OUTPUT = 'out'
BUS = 'bus'


@dataclass(frozen=True)
class Endpoint:
    """
    What the compiler needs to know about a graph node: how wide it is on each side, and
    whether its audio crosses a ring (network streams, per-process capture) rather than a
    device callback.
    """
    input_channels: int = 0
    output_channels: int = 0
    ring_frames: int = 0
    limiter: bool = True


@dataclass
class CompiledPlan:
    nodes: list[Node]
    routes: list[Route]
    inserts: list[Insert]
    master_gain: float
    # Which native node carries each graph node's input and output side.
    native_ids: dict[tuple[NodeId, str], int] = field(default_factory=dict)


class PlanCompiler:
    def __init__(self):
        self._ids: dict[tuple[NodeId, str], int] = {}
        self._next = 1

    def native_id(self, node: NodeId, role: str) -> int:
        key = (node, role)
        if key not in self._ids:
            self._ids[key] = self._next
            self._next += 1
        return self._ids[key]

    def compile(self, graph: RoutingGraph, endpoints: dict[NodeId, Endpoint],
                inserts: dict[tuple[NodeId, str], list[tuple[int, int, bool]]] | None = None) -> CompiledPlan:
        """
        `inserts` maps (graph node, role) to its built-in processors as (slot, type,
        bypassed). A route touching an endpoint the caller did not describe is an error
        here, not a silently dropped cable.
        """
        nodes: dict[int, Node] = {}
        routes: list[Route] = []
        native_ids: dict[tuple[NodeId, str], int] = {}

        def side(node: NodeId, role: str) -> int:
            endpoint = endpoints.get(node)
            if endpoint is None:
                raise KeyError(f"route touches {node}, which has no endpoint description")
            if node.kind == 'bus':
                channels = endpoint.output_channels or endpoint.input_channels
                native = self.native_id(node, BUS)
                nodes.setdefault(native, Node.bus(native, channels))
                native_ids[(node, INPUT)] = native_ids[(node, OUTPUT)] = native
                return native

            native = self.native_id(node, role)
            if role == INPUT:
                if endpoint.input_channels < 1:
                    raise ValueError(f"{node} has no input channels to route from")
                nodes.setdefault(native, Node.source(native, endpoint.input_channels, endpoint.ring_frames))
            else:
                if endpoint.output_channels < 1:
                    raise ValueError(f"{node} has no output channels to route to")
                nodes.setdefault(native, Node.sink(native, endpoint.output_channels, endpoint.ring_frames,
                                                   limiter=endpoint.limiter))
            native_ids[(node, role)] = native
            return native

        for connection in graph.connections:
            # Solo is a property of the graph, not the route: with anything soloed, every
            # route out of a non-soloed source is silent. Muting it here, rather than
            # dropping it, keeps its smoothed gain alive so un-soloing does not click.
            silenced = bool(graph.soloed) and connection.source not in graph.soloed
            routes.append(Route(
                source=side(connection.source, INPUT),
                dest=side(connection.dest, OUTPUT),
                gain=connection.gain,
                pan=connection.pan,
                muted=connection.muted or silenced,
                invert=connection.invert,
            ))

        compiled_inserts = []
        for (graph_node, role), processors in (inserts or {}).items():
            native = native_ids.get((graph_node, role))
            if native is None:
                continue  # an endpoint with no routes is not in the plan, so neither are its inserts
            for slot, kind, bypassed in processors:
                compiled_inserts.append(Insert(native, slot, kind, bypassed))

        return CompiledPlan(
            nodes=list(nodes.values()),
            routes=routes,
            inserts=compiled_inserts,
            master_gain=graph.master_gain,
            native_ids=native_ids,
        )
