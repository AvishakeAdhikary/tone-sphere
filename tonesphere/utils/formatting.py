"""
Display helpers for engine statistics.

Single place that decides how an unmeasured value is rendered, so the CLI, GUI and
API never disagree — and so a missing measurement can never be formatted as 0.0.
"""

from collections.abc import Mapping
from typing import Any

UNKNOWN = "--"


def format_measurement(value: float | None, unit: str = "", decimals: int = 1) -> str:
    """
    Render a possibly-unmeasured number.

    None becomes "--". This is the whole point: `f"{value:.1f}"` on a value we never
    measured prints "0.0", which reads as a real and excellent result.
    """
    if value is None:
        return UNKNOWN
    return f"{value:.{decimals}f}{unit}"


def format_performance_summary(stats: Mapping[str, Any]) -> str:
    """One-line engine summary suitable for a status bar."""
    cpu = format_measurement(stats.get('cpu_usage'), "%")
    reported = stats.get('reported_latency_ms')

    if reported is None:
        nominal = format_measurement(stats.get('nominal_latency_ms'), " ms")
        latency = f"{UNKNOWN} (nominal {nominal})"
    else:
        # "reported", because it is what the driver says; only a signal we emitted and
        # captured back earns the word "measured", and that is shown separately.
        latency = f"{format_measurement(reported, ' ms')} reported"

    summary = f"CPU: {cpu}  |  Latency: {latency}"

    measured = stats.get('measured_round_trip_ms')
    if measured is not None:
        summary += f"  |  Measured round trip: {format_measurement(measured, ' ms')}"

    if not stats.get('audio_path_active', False):
        summary += "  |  NO AUDIO PATH"

    return summary
