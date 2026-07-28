"""
ToneSphere audio engine.

The real audio path: PortAudio streams, a lock-free routing graph, and a mixer that runs
in the driver's callback. This package replaces `tonesphere/drivers/`, whose eight
"backends" each satisfied an interface and returned arrays of zeros.
"""

from tonesphere.engine.devices import (
    AudioBackendUnavailable,
    DeviceInfo,
    HostApi,
    available_host_apis,
    describe_backend,
    enumerate_devices,
    find_device,
    preferred_host_api,
)
from tonesphere.engine.graph import (
    Connection,
    GraphHolder,
    NodeId,
    RoutingGraph,
    bus_node,
    db_to_linear,
    device_node,
    linear_to_db,
)
from tonesphere.engine.host import AudioHost, HostStatistics, StreamConfig
from tonesphere.engine.meters import MeterBank, MeterReading, MeterRegistry
from tonesphere.engine.ringbuffer import AudioRingBuffer

__all__ = [
    "AudioBackendUnavailable",
    "AudioHost",
    "AudioRingBuffer",
    "Connection",
    "DeviceInfo",
    "GraphHolder",
    "HostApi",
    "HostStatistics",
    "MeterBank",
    "MeterReading",
    "MeterRegistry",
    "NodeId",
    "RoutingGraph",
    "StreamConfig",
    "available_host_apis",
    "bus_node",
    "db_to_linear",
    "describe_backend",
    "device_node",
    "enumerate_devices",
    "find_device",
    "linear_to_db",
    "preferred_host_api",
]
