"""
Guards against the failure mode this project was built on: code that reports success
or health without having done or measured anything.

These tests are deliberately about *reporting*, not audio. They should be deleted only
when the thing they describe genuinely works.
"""

import numpy as np

from tonesphere.core.engine import AudioEngine
from tonesphere.utils.formatting import UNKNOWN, format_measurement, format_performance_summary


class TestUnmeasuredValuesAreNotZero:
    """A measurement we never took must not render as a healthy zero."""

    def test_none_renders_as_unknown(self):
        assert format_measurement(None) == UNKNOWN
        assert format_measurement(None, "%") == UNKNOWN

    def test_real_zero_still_renders_as_zero(self):
        assert format_measurement(0.0, "%") == "0.0%"

    def test_summary_marks_missing_cpu_as_unknown(self):
        summary = format_performance_summary({
            'cpu_usage': None,
            'nominal_latency_ms': 2.67,
            'measured_latency_ms': None,
            'audio_path_active': False,
        })

        assert "CPU: --" in summary
        assert "0.0%" not in summary

    def test_summary_flags_inactive_audio_path(self):
        summary = format_performance_summary({
            'cpu_usage': None,
            'nominal_latency_ms': 2.67,
            'measured_latency_ms': None,
            'audio_path_active': False,
        })

        assert "NO AUDIO PATH" in summary

    def test_summary_distinguishes_nominal_from_measured_latency(self):
        """
        Nominal latency is buffer/rate arithmetic. Presenting it as the real figure is
        how the old UI claimed 2.67 ms while passing no audio at all.
        """
        summary = format_performance_summary({
            'cpu_usage': None,
            'nominal_latency_ms': 2.67,
            'measured_latency_ms': None,
            'audio_path_active': False,
        })

        assert "nominal" in summary.lower()


class TestEngineStatsAreHonest:
    def test_cpu_usage_starts_unmeasured_not_zero(self):
        engine = AudioEngine()
        stats = engine.get_performance_stats()

        assert stats['cpu_usage'] is None, "0.0 would render as a real 0% CPU reading"
        assert stats['measured_latency_ms'] is None

    def test_audio_path_is_not_claimed_active(self):
        """Must stay False until a driver callback genuinely carries audio."""
        engine = AudioEngine()

        assert engine.get_performance_stats()['audio_path_active'] is False


class TestRoutingDoesNotOverclaim:
    def test_route_to_physical_device_says_it_carries_no_audio(self):
        """
        Routing a physical device is recorded in the matrix but moves no audio yet. The
        caller must be told, not handed a bare success.
        """
        engine = AudioEngine()
        engine.virtual_manager.create_input(channels=2)

        physical_id = 424242
        virtual_id = next(iter(engine.virtual_manager.input_devices))

        success, message = engine.create_routing(physical_id, virtual_id)

        assert success is True
        assert "recorded only" in message

        engine.virtual_manager.clear_all()

    def test_virtual_to_virtual_route_reports_plain_success(self):
        engine = AudioEngine()
        source_id = engine.virtual_manager.create_input(channels=2)
        dest_id = engine.virtual_manager.create_output(channels=2)

        success, message = engine.create_routing(source_id, dest_id)

        assert success is True
        assert "recorded only" not in message

        engine.virtual_manager.clear_all()


class TestNoResurrectedFakes:
    """
    Structural guards. If someone re-adds the deleted no-op layers, these fail rather
    than the project quietly regressing to reporting success over silence.
    """

    def test_second_device_registry_is_not_reintroduced(self):
        import tonesphere.devices.native_virtual as native_virtual

        assert not hasattr(native_virtual, 'NativeVirtualDeviceManager'), (
            "Two device registries caused every route to be recorded against an empty "
            "dict. Keep exactly one: VirtualDeviceManager."
        )

    def test_engine_has_a_single_virtual_registry(self):
        engine = AudioEngine()

        assert hasattr(engine, 'virtual_manager')
        assert not hasattr(engine, 'virtual_device_manager')

    def test_stream_manager_has_no_noop_polling_thread(self):
        from tonesphere.core.stream_manager import AudioStreamManager

        assert not hasattr(AudioStreamManager, '_processing_loop'), (
            "This loop woke 1000x/second to assign a timestamp and moved no audio."
        )
