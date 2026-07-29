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
    latency = format_measurement(stats.get('measured_latency_ms'), " ms")

    if stats.get('measured_latency_ms') is None:
        nominal = format_measurement(stats.get('nominal_latency_ms'), " ms")
        latency = f"{latency} (nominal {nominal})"

    summary = f"CPU: {cpu}  |  Latency: {latency}"

    if not stats.get('audio_path_active', False):
        summary += "  |  NO AUDIO PATH"

    return summary
