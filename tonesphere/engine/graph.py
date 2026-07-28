"""
The routing graph, and how it reaches the audio callback safely.

The problem this solves: the UI mutates routing whenever the user drags a cable, while
the audio callback reads routing every few milliseconds on a thread that must never
block. Guarding the graph with a lock would let a UI click stall the callback and cause
an audible dropout.

So the graph is immutable. Edits build a whole new `RoutingGraph`; the callback picks up
the new one by reading a single attribute, which is atomic under the GIL. A callback
already in progress finishes against the old graph, which is consistent because nothing
mutates it. This is a read-copy-update, and it is why there is no lock anywhere near the
audio path.

Gains are stored linear, not in dB. dB is a display unit; converting per sample in the
callback would be arithmetic we can do once, at edit time.
"""

import math
from dataclasses import dataclass, field, replace
from typing import Dict, FrozenSet, Iterable, List, Optional, Tuple

MIN_GAIN_DB = -60.0
MAX_GAIN_DB = 12.0


def db_to_linear(db: float) -> float:
    """Amplitude ratio for a decibel value. At or below MIN_GAIN_DB this is silence."""
    if db <= MIN_GAIN_DB:
        return 0.0
    return float(10.0 ** (db / 20.0))


def linear_to_db(linear: float) -> float:
    """Decibels for an amplitude ratio. Zero maps to MIN_GAIN_DB, not -inf."""
    if linear <= 0.0:
        return MIN_GAIN_DB
    return max(MIN_GAIN_DB, 20.0 * math.log10(linear))


@dataclass(frozen=True)
class NodeId:
    """
    Identifies one endpoint in the graph.

    `kind` distinguishes a physical device from an in-process bus, because they are
    reached differently: a device by PortAudio index, a bus by name.
    """
    kind: str   # 'device' | 'bus'
    ref: str    # device key, or bus name

    def __str__(self) -> str:
        return f"{self.kind}:{self.ref}"


def device_node(key: str) -> NodeId:
    return NodeId('device', key)


def bus_node(name: str) -> NodeId:
    return NodeId('bus', name)


@dataclass(frozen=True)
class Connection:
    """
    One route, with its mix parameters resolved.

    `gain` is linear and already folded together with mute, so the callback multiplies by a
    single float and never branches on state.

    `pan` and `invert` live on the route rather than on the source, because the same source
    can legitimately sit centre in the headphone mix and hard left in a recording feed.
    """
    source: NodeId
    dest: NodeId
    gain: float = 1.0
    muted: bool = False
    pan: float = 0.0
    invert: bool = False
    source_channels: Optional[Tuple[int, ...]] = None
    dest_channels: Optional[Tuple[int, ...]] = None

    @property
    def effective_gain(self) -> float:
        return 0.0 if self.muted else self.gain

    @property
    def is_audible(self) -> bool:
        """Whether this route can contribute anything, so the mixer can skip it."""
        return not self.muted and self.gain > 0.0

    def key(self) -> Tuple[str, str]:
        return (str(self.source), str(self.dest))


@dataclass(frozen=True)
class RoutingGraph:
    """
    An immutable routing snapshot.

    Every mutator returns a new graph. Nothing here is ever edited in place, which is
    the whole basis of the lock-free handoff to the callback.
    """
    connections: Tuple[Connection, ...] = ()
    soloed: FrozenSet[NodeId] = frozenset()
    master_gain: float = 1.0

    # Derived, built once in __post_init__ so the callback never has to search.
    _by_dest: Dict[NodeId, Tuple[Connection, ...]] = field(
        default_factory=dict, compare=False, repr=False
    )

    def __post_init__(self):
        by_dest: Dict[NodeId, List[Connection]] = {}

        solo_active = bool(self.soloed)

        for connection in self.connections:
            # Solo is resolved here, at edit time, so the callback does not have to
            # consult global state to decide whether one route is audible.
            if solo_active and connection.source not in self.soloed:
                continue
            if not connection.is_audible:
                continue
            by_dest.setdefault(connection.dest, []).append(connection)

        object.__setattr__(
            self, '_by_dest',
            {dest: tuple(items) for dest, items in by_dest.items()},
        )

    def sources_for(self, dest: NodeId) -> Tuple[Connection, ...]:
        """Audible routes feeding one destination. The callback's hot lookup."""
        return self._by_dest.get(dest, ())

    @property
    def destinations(self) -> Tuple[NodeId, ...]:
        return tuple(self._by_dest.keys())

    @property
    def active_sources(self) -> Tuple[NodeId, ...]:
        seen = {}
        for connections in self._by_dest.values():
            for connection in connections:
                seen[connection.source] = None
        return tuple(seen)

    def nodes(self) -> Tuple[NodeId, ...]:
        """Every node mentioned by any route, audible or not."""
        seen = {}
        for connection in self.connections:
            seen[connection.source] = None
            seen[connection.dest] = None
        return tuple(seen)

    def find(self, source: NodeId, dest: NodeId) -> Optional[Connection]:
        for connection in self.connections:
            if connection.source == source and connection.dest == dest:
                return connection
        return None

    # --- Mutators. Each returns a new graph. ---

    def with_connection(self, connection: Connection) -> "RoutingGraph":
        """Add or replace a route."""
        remaining = tuple(
            c for c in self.connections
            if not (c.source == connection.source and c.dest == connection.dest)
        )
        return replace(self, connections=remaining + (connection,))

    def without_connection(self, source: NodeId, dest: NodeId) -> "RoutingGraph":
        return replace(self, connections=tuple(
            c for c in self.connections
            if not (c.source == source and c.dest == dest)
        ))

    def without_node(self, node: NodeId) -> "RoutingGraph":
        """Drop every route touching a node — for when a device is unplugged."""
        return replace(
            self,
            connections=tuple(
                c for c in self.connections if c.source != node and c.dest != node
            ),
            soloed=frozenset(n for n in self.soloed if n != node),
        )

    def with_gain(self, source: NodeId, dest: NodeId, gain: float) -> "RoutingGraph":
        existing = self.find(source, dest)
        if existing is None:
            return self
        clamped = min(max(gain, 0.0), db_to_linear(MAX_GAIN_DB))
        return self.with_connection(replace(existing, gain=clamped))

    def with_gain_db(self, source: NodeId, dest: NodeId, gain_db: float) -> "RoutingGraph":
        return self.with_gain(source, dest, db_to_linear(gain_db))

    def with_mute(self, source: NodeId, dest: NodeId, muted: bool) -> "RoutingGraph":
        existing = self.find(source, dest)
        if existing is None:
            return self
        return self.with_connection(replace(existing, muted=muted))

    def with_pan(self, source: NodeId, dest: NodeId, pan: float) -> "RoutingGraph":
        existing = self.find(source, dest)
        if existing is None:
            return self
        return self.with_connection(replace(existing, pan=min(max(pan, -1.0), 1.0)))

    def with_invert(self, source: NodeId, dest: NodeId, invert: bool) -> "RoutingGraph":
        existing = self.find(source, dest)
        if existing is None:
            return self
        return self.with_connection(replace(existing, invert=invert))

    def with_solo(self, node: NodeId, soloed: bool) -> "RoutingGraph":
        current = set(self.soloed)
        if soloed:
            current.add(node)
        else:
            current.discard(node)
        return replace(self, soloed=frozenset(current))

    def with_master_gain(self, gain: float) -> "RoutingGraph":
        return replace(self, master_gain=min(max(gain, 0.0), db_to_linear(MAX_GAIN_DB)))

    def cleared(self) -> "RoutingGraph":
        return replace(self, connections=(), soloed=frozenset())

    # --- Diagnostics ---

    def would_feedback(self, source: NodeId, dest: NodeId) -> bool:
        """
        Whether adding source -> dest closes a loop.

        A cycle in an audio graph is not a subtle bug; it is a runaway howl through the
        user's headphones at whatever volume they had set. Refuse it before it happens.
        """
        if source == dest:
            return True

        # Walk forward from dest: if we can reach source, adding the edge closes a loop.
        stack = [dest]
        seen = set()

        while stack:
            node = stack.pop()
            if node == source:
                return True
            if node in seen:
                continue
            seen.add(node)

            for connection in self.connections:
                if connection.source == node:
                    stack.append(connection.dest)

        return False

    def to_dict(self) -> dict:
        """Serialisable form, for presets and the API."""
        return {
            'master_gain': self.master_gain,
            'soloed': [str(node) for node in sorted(self.soloed, key=str)],
            'connections': [
                {
                    'source': str(c.source),
                    'dest': str(c.dest),
                    'gain': c.gain,
                    'gain_db': round(linear_to_db(c.gain), 2),
                    'muted': c.muted,
                    'pan': c.pan,
                    'invert': c.invert,
                }
                for c in self.connections
            ],
        }

    @classmethod
    def from_dict(cls, data: dict) -> "RoutingGraph":
        def parse(text: str) -> NodeId:
            kind, _, ref = text.partition(':')
            return NodeId(kind, ref)

        connections = tuple(
            Connection(
                source=parse(item['source']),
                dest=parse(item['dest']),
                gain=float(item.get('gain', 1.0)),
                muted=bool(item.get('muted', False)),
                pan=float(item.get('pan', 0.0)),
                invert=bool(item.get('invert', False)),
            )
            for item in data.get('connections', ())
        )

        return cls(
            connections=connections,
            soloed=frozenset(parse(text) for text in data.get('soloed', ())),
            master_gain=float(data.get('master_gain', 1.0)),
        )


class GraphHolder:
    """
    The handoff point between the control thread and the audio callback.

    Writers call `commit`; the callback calls `current` once at the top of each block and
    uses that reference throughout. Because graphs are immutable, a callback holding the
    previous graph stays correct for the rest of its block. `_generation` lets the host
    notice a change and reconcile streams without diffing the graph itself.
    """

    __slots__ = ('_graph', '_generation')

    def __init__(self, graph: Optional[RoutingGraph] = None):
        self._graph = graph if graph is not None else RoutingGraph()
        self._generation = 0

    def current(self) -> RoutingGraph:
        """Read the active graph. Single attribute load — safe from the callback."""
        return self._graph

    @property
    def generation(self) -> int:
        return self._generation

    def commit(self, graph: RoutingGraph) -> int:
        """
        Publish a new graph.

        The generation bump happens before the swap so a reader that sees the new graph
        never sees a stale generation.
        """
        self._generation += 1
        self._graph = graph
        return self._generation

    def update(self, mutator) -> int:
        """Apply `mutator(graph) -> graph` and publish the result."""
        return self.commit(mutator(self._graph))
