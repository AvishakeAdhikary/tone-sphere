"""
Guards against the failure mode this project was built on: code that reports success or
health without having done or measured anything.

These are about *reporting*, not audio. Delete one only when the thing it describes
genuinely works — that is the point of them.
"""

import importlib

import pytest

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
            'nominal_latency_ms': 5.33,
            'measured_latency_ms': None,
            'audio_path_active': False,
        })

        assert "CPU: --" in summary
        assert "0.0%" not in summary

    def test_summary_flags_inactive_audio_path(self):
        summary = format_performance_summary({
            'cpu_usage': None,
            'nominal_latency_ms': 5.33,
            'measured_latency_ms': None,
            'audio_path_active': False,
        })

        assert "NO AUDIO PATH" in summary

    def test_summary_distinguishes_nominal_from_measured_latency(self):
        """
        Nominal latency is blocksize/samplerate arithmetic and describes only our own
        contribution. Measured here: 5.33 ms nominal against 8.33 ms real on WASAPI
        exclusive and 120 ms on DirectSound. Presenting the arithmetic as the real figure
        is how the old code claimed 2.67 ms while passing no audio at all.
        """
        summary = format_performance_summary({
            'cpu_usage': None,
            'nominal_latency_ms': 5.33,
            'measured_latency_ms': None,
            'audio_path_active': False,
        })

        assert "nominal" in summary.lower()

    def test_measured_latency_is_shown_when_known(self):
        summary = format_performance_summary({
            'cpu_usage': 2.5,
            'nominal_latency_ms': 5.33,
            'measured_latency_ms': 8.33,
            'audio_path_active': True,
        })

        assert "8.3" in summary
        assert "NO AUDIO PATH" not in summary


class TestEngineStatsAreHonest:
    def test_cpu_usage_is_unmeasured_before_running(self):
        stats = AudioEngine().get_performance_stats()

        assert stats['cpu_usage'] is None, "0.0 would render as a real 0% CPU reading"
        assert stats['measured_latency_ms'] is None

    def test_audio_path_is_not_claimed_active_before_start(self):
        assert AudioEngine().get_performance_stats()['audio_path_active'] is False

    def test_nominal_latency_is_labelled_as_nominal(self):
        stats = AudioEngine(sample_rate=48000, buffer_size=256).get_performance_stats()

        assert stats['nominal_latency_ms'] == pytest.approx(5.333, abs=0.01)
        assert 'measured_latency_ms' in stats

    def test_meters_are_empty_rather_than_zero_when_stopped(self):
        """
        A meter reading of 0.0 dBFS while nothing runs would say "silence was measured".
        Nothing was measured, so there is nothing to report.
        """
        assert AudioEngine().get_meters() == {}

    def test_backend_problems_are_surfaced_not_swallowed(self):
        stats = AudioEngine().get_performance_stats()

        assert 'problems' in stats
        assert 'backend_error' in stats


class TestRoutingDoesNotOverclaim:
    def test_unknown_source_is_refused(self):
        engine = AudioEngine()
        success, message = engine.create_routing(999_999, 999_998)

        assert success is False
        assert 'unknown' in message.lower()

    def test_feedback_loop_is_refused(self):
        """
        A cycle is not a subtle bug. It is a runaway howl at whatever volume the user's
        headphones were set to, so it must be refused before it happens.
        """
        engine = AudioEngine()
        a = engine.create_virtual_input("A", channels=2)
        b = engine.create_virtual_output("B", channels=2)

        assert engine.create_routing(a, b)[0] is True

        success, message = engine.create_routing(b, a)
        assert success is False
        assert 'feedback' in message.lower()

    def test_self_route_is_refused(self):
        engine = AudioEngine()
        bus = engine.create_virtual_input("A", channels=2)

        success, _ = engine.create_routing(bus, bus)
        assert success is False

    def test_bus_creation_respects_its_limit(self):
        engine = AudioEngine(max_virtual_inputs=2)

        assert engine.create_virtual_input("one") is not None
        assert engine.create_virtual_input("two") is not None
        assert engine.create_virtual_input("three") is None, "must refuse past the limit"

    def test_bus_sample_rate_change_is_refused_rather_than_ignored(self):
        """
        A bus runs at the engine rate. Returning True while doing nothing would be the
        exact dishonesty this suite exists to prevent.
        """
        engine = AudioEngine()
        bus = engine.create_virtual_input("A")

        assert engine.update_virtual_device_sample_rate(bus, 44100) is False


class TestBusesAreNotSystemDevices:
    def test_buses_are_labelled_as_in_process(self):
        """
        The old code claimed these "appear in system sound settings". They do not, and
        cannot without a signed kernel driver, so the UI must not imply otherwise.
        """
        engine = AudioEngine()
        engine.create_virtual_input("Guitar", channels=2)

        buses = [d for d in engine.get_devices() if 'bus' in d['host_api'].lower()]

        assert buses
        assert all('in-process' in d['host_api'].lower() for d in buses)


class TestNoResurrectedFakes:
    """Structural guards, so the project cannot quietly regress to reporting silence."""

    def test_fake_driver_package_is_gone(self):
        """
        All eight "drivers" allocated `np.zeros()` and called it a stream. PortAudio is
        already the abstraction over ASIO/WASAPI/WDM-KS/CoreAudio/ALSA/JACK;
        reimplementing it in Python was the original mistake.
        """
        with pytest.raises(ImportError):
            importlib.import_module('tonesphere.drivers')

    def test_noop_stream_manager_is_gone(self):
        """
        `core.stream_manager` woke 1000x/second and its entire body was
        `stream.last_callback_time = time.time()`.
        """
        with pytest.raises(ImportError):
            importlib.import_module('tonesphere.core.stream_manager')

    def test_queue_based_virtual_devices_are_gone(self):
        """
        `devices/` moved audio between `queue.Queue` objects on polling threads. Audio now
        moves through lock-free rings inside the driver's callback.
        """
        with pytest.raises(ImportError):
            importlib.import_module('tonesphere.devices.native_virtual')

    def test_engine_runs_on_the_real_audio_host(self):
        from tonesphere.engine import AudioHost

        assert isinstance(AudioEngine().host, AudioHost)

    def test_engine_has_exactly_one_device_registry(self):
        engine = AudioEngine()

        assert not hasattr(engine, 'virtual_manager')
        assert not hasattr(engine, 'virtual_device_manager')

    def test_no_polling_thread_pretends_to_carry_audio(self):
        assert not hasattr(AudioEngine, '_audio_processing_loop')

    def test_host_api_selection_requires_actual_devices(self):
        """
        Regression: ASIO was selected whenever `HKLM\\SOFTWARE\\ASIO` existed — true on
        machines with zero ASIO drivers — after which enumeration returned nothing and no
        fallback was attempted. A backend must have devices to be chosen.
        """
        from tonesphere.engine.devices import preferred_host_api

        assert preferred_host_api([]) is None


class TestPartialFailureIsNotSuccess:
    """
    Regression: the host logged "Audio host running: 1 stream(s)" and reported a healthy
    8.3 ms latency while the input stream had failed to open. The route carried nothing.
    """

    def make_host_with_failed_stream(self):
        from tonesphere.engine.devices import DeviceInfo, HostApi
        from tonesphere.engine.host import AudioHost, StreamConfig, _DeviceStream

        host = AudioHost()
        device = DeviceInfo(
            index=0, name='broken', host_api=HostApi.UNKNOWN, host_api_name='test',
            max_input_channels=2, max_output_channels=0, default_samplerate=48000,
            default_low_input_latency_ms=0.0, default_low_output_latency_ms=0.0,
            default_high_input_latency_ms=0.0, default_high_output_latency_ms=0.0,
        )
        stream = _DeviceStream(StreamConfig(
            device=device, samplerate=48000, blocksize=256, input_channels=2,
        ))
        stream.error = "Invalid device"
        host._streams[device.key] = stream
        host._running = True
        return host, stream

    def test_failed_stream_is_reported(self):
        host, _ = self.make_host_with_failed_stream()

        assert host.failed_streams()
        assert host.dead_nodes()

    def test_running_with_a_failure_is_not_fully_healthy(self):
        host, _ = self.make_host_with_failed_stream()
        stats = host.statistics()

        assert stats.running is True
        assert stats.fully_healthy is False, "partial success is not health"
        assert stats.live_stream_count == 0

    def test_failed_stream_contributes_no_latency_measurement(self):
        """Averaging a broken stream in would make a broken setup look measured and fine."""
        host, _ = self.make_host_with_failed_stream()

        assert host.statistics().measured_latency_ms is None

    def test_dead_stream_is_not_counted_as_live(self):
        host, stream = self.make_host_with_failed_stream()

        assert stream.is_live is False


class TestErrorMessagesAreActionable:
    def test_os_refusal_explains_the_privacy_setting(self):
        """
        "Invalid device [PaErrorCode -9996]" on a device that enumerated fine means the OS
        refused access. Saying which setting to check saves an hour.
        """
        from tonesphere.engine.devices import DeviceInfo, HostApi
        from tonesphere.engine.host import AudioHost, StreamConfig

        device = DeviceInfo(
            index=0, name='mic', host_api=HostApi.WASAPI, host_api_name='Windows WASAPI',
            max_input_channels=2, max_output_channels=0, default_samplerate=48000,
            default_low_input_latency_ms=0.0, default_low_output_latency_ms=0.0,
            default_high_input_latency_ms=0.0, default_high_output_latency_ms=0.0,
        )
        config = StreamConfig(device=device, samplerate=48000, blocksize=256,
                              input_channels=2)

        message = AudioHost()._explain_open_failure(
            config, Exception("Error opening InputStream: Invalid device [PaErrorCode -9996]")
        )

        assert 'refused access' in message
        assert 'Privacy' in message

    def test_bad_sample_rate_names_the_rate(self):
        from tonesphere.engine.devices import DeviceInfo, HostApi
        from tonesphere.engine.host import AudioHost, StreamConfig

        device = DeviceInfo(
            index=0, name='dev', host_api=HostApi.WASAPI, host_api_name='Windows WASAPI',
            max_input_channels=0, max_output_channels=2, default_samplerate=48000,
            default_low_input_latency_ms=0.0, default_low_output_latency_ms=0.0,
            default_high_input_latency_ms=0.0, default_high_output_latency_ms=0.0,
        )
        config = StreamConfig(device=device, samplerate=192000, blocksize=256,
                              output_channels=2)

        message = AudioHost()._explain_open_failure(
            config, Exception("Invalid sample rate [PaErrorCode -9997]")
        )

        assert '192000' in message

    def test_device_in_use_suggests_disabling_exclusive_mode(self):
        from tonesphere.engine.devices import DeviceInfo, HostApi
        from tonesphere.engine.host import AudioHost, StreamConfig

        device = DeviceInfo(
            index=0, name='dev', host_api=HostApi.WASAPI, host_api_name='Windows WASAPI',
            max_input_channels=0, max_output_channels=2, default_samplerate=48000,
            default_low_input_latency_ms=0.0, default_low_output_latency_ms=0.0,
            default_high_input_latency_ms=0.0, default_high_output_latency_ms=0.0,
        )
        config = StreamConfig(device=device, samplerate=48000, blocksize=256,
                              output_channels=2)

        message = AudioHost()._explain_open_failure(
            config, Exception("Device unavailable [PaErrorCode -9985]")
        )

        assert 'exclusive' in message.lower()


class TestEngineStates:
    def test_stopped_before_start(self):
        assert AudioEngine().state == 'stopped'

    def test_idle_when_started_with_nothing_patched(self):
        """
        Regression: this displayed as "Engine: Stopped" while the button read STOP ENGINE.
        Started-but-unpatched is a real third state.
        """
        engine = AudioEngine()
        engine.initialize()
        engine.start_engine()

        assert engine.state == 'idle'
        assert engine.has_routes is False

    def test_clearing_all_routes_returns_to_idle(self):
        """Holding devices open exclusively with nothing patched serves nobody."""
        engine = AudioEngine()
        engine.initialize()
        engine.start_engine()

        engine.create_virtual_input("a")
        engine.clear_all_routing()

        assert engine.state == 'idle'


class TestMonitorPatchIsSafe:
    def test_monitor_patch_is_created_muted(self):
        """
        Default input to default output is a laptop mic into laptop speakers. Unmuted, that
        is acoustic feedback at whatever volume the machine was set to.
        """
        engine = AudioEngine()
        engine.initialize()

        if engine.default_input_id() is None or engine.default_output_id() is None:
            pytest.skip("needs both an input and an output device")

        success, message = engine.create_monitor_patch(muted=True)
        if not success:
            pytest.skip(f"could not patch on this machine: {message}")

        routes = engine.get_routing_matrix()
        assert routes
        assert all(r['muted'] for r in routes.values())
        assert 'muted' in message
