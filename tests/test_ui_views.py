"""
The M10 views, offscreen: that each one shows what the engine measured, and `--` for what it
did not. The per-side meter test drives the window with a hand-built meter dictionary,
because what is under test is which strip a reading lands on, not how it was measured
(`tests/native/test_engine.py` covers that).
"""

import pytest

from tonesphere.core.engine import AudioEngine
from tonesphere.utils.config import ConfigManager
from tonesphere.utils.formatting import UNKNOWN

pytest.importorskip("PySide6")


@pytest.fixture
def qt_app(monkeypatch):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication

    return QApplication.instance() or QApplication([])


@pytest.fixture
def engine():
    e = AudioEngine()
    e.initialize()
    yield e
    e.cleanup()


def _side(peak_db, channels):
    return {'peak_db': peak_db, 'rms_db': peak_db - 3, 'peak_hold_db': peak_db, 'clipped': False,
            'channel_peak_db': channels, 'channel_rms_db': [c - 3 for c in channels],
            'channel_peak_hold_db': channels}


class TestMetersLandOnTheirOwnSide:
    def test_a_duplex_devices_input_and_output_are_metered_apart(self, qt_app, tmp_path, monkeypatch):
        from tonesphere.ui.main_window import MainWindow

        window = MainWindow(ConfigManager(db_path=tmp_path / "settings.db"))
        try:
            duplex = {'id': 900, 'name': 'Interface', 'channels': 2, 'origin': 'hardware', 'host_api': 'ASIO'}
            window._add_strip({**duplex, 'direction': 'input'})
            window._add_strip({**duplex, 'direction': 'output'})
            meters = {900: {'peak_db': -6.0, 'rms_db': -9.0, 'peak_hold_db': -6.0, 'clipped': False,
                            'sides': {'input': _side(-6.0, [-6.0, -12.0]), 'output': _side(-20.0, [-20.0, -30.0])}}}
            monkeypatch.setattr(type(window.engine), 'is_running', property(lambda self: True))
            monkeypatch.setattr(window.engine, 'get_meters', lambda: meters)

            window._update_meters()

            assert window._strips[(900, 'input')].meter._peaks == [-6.0, -12.0]
            assert window._strips[(900, 'output')].meter._peaks == [-20.0, -30.0], \
                "the output side must not show the input's reading"
        finally:
            window.close()

    def test_a_side_with_no_reading_is_inactive_not_silent(self, qt_app, tmp_path, monkeypatch):
        from tonesphere.ui.main_window import MainWindow

        window = MainWindow(ConfigManager(db_path=tmp_path / "settings.db"))
        try:
            duplex = {'id': 901, 'name': 'Interface', 'channels': 2, 'origin': 'hardware', 'host_api': 'ASIO'}
            window._add_strip({**duplex, 'direction': 'input'})
            window._add_strip({**duplex, 'direction': 'output'})
            meters = {901: {'peak_db': -6.0, 'rms_db': -9.0, 'peak_hold_db': -6.0, 'clipped': False,
                            'sides': {'input': _side(-6.0, [-6.0, -6.0])}}}
            monkeypatch.setattr(type(window.engine), 'is_running', property(lambda self: True))
            monkeypatch.setattr(window.engine, 'get_meters', lambda: meters)

            window._update_meters()

            assert window._strips[(901, 'output')].meter._active is False
        finally:
            window.close()


class TestStrip:
    def test_balance_is_offered_on_stereo_strips_only(self, qt_app):
        from tonesphere.ui.strip import ChannelStripWidget

        assert ChannelStripWidget(1, "st", "", channels=2).pan_knob.isHidden() is False
        assert ChannelStripWidget(2, "mono", "", channels=1).pan_knob.isHidden() is True

    def test_inserts_are_disabled_where_plugins_cannot_be_hosted(self, qt_app):
        from tonesphere.ui.strip import ChannelStripWidget

        assert ChannelStripWidget(1, "bus", "", hosts_plugins=False).inserts_button.isEnabled() is False
        strip = ChannelStripWidget(1, "dev", "", hosts_plugins=True)
        received = []
        strip.inserts_requested.connect(lambda device, is_input: received.append((device, is_input)))
        strip.inserts_button.click()
        assert received == [(1, True)]


class TestDiagnostics:
    def test_nothing_measured_reads_as_unknown(self, qt_app, engine):
        from tonesphere.ui.diagnostics_view import DiagnosticsDialog

        dialog = DiagnosticsDialog(engine)
        try:
            latency = dialog.latency_section.values
            assert latency['measured'].text() == UNKNOWN
            assert latency['driver_in'].text() == UNKNOWN
            timing = dialog.timing_section.values
            assert timing['load_worst'].text() == UNKNOWN, "a stopped engine has measured no load"
            assert UNKNOWN in timing['callback'].text()
        finally:
            dialog.done(0)

    def test_a_measured_round_trip_is_shown_only_for_the_current_configuration(self, qt_app, engine):
        from tonesphere.ui.diagnostics_view import DiagnosticsDialog

        trip = {'path': 'capture', 'measured_ms': 12.34, 'measured_frames': 592, 'confidence': 20.0,
                'nominal_ms': 5.3, 'reported_ms': 10.0, 'sample_rate': engine.sample_rate,
                'block': engine.buffer_size, 'output': 'Out', 'input': 'In', 'note': 'measured'}
        engine._round_trip = trip
        dialog = DiagnosticsDialog(engine)
        try:
            assert '12.34' in dialog.latency_section.values['measured'].text()
            engine._round_trip = {**trip, 'block': engine.buffer_size * 2}
            dialog.refresh()
            assert dialog.latency_section.values['measured'].text() == UNKNOWN, \
                "a measurement at another block size does not describe this one"
            assert '12.34' in dialog.latency_section.values['last'].text()
        finally:
            dialog.done(0)


class TestRoundTripHonesty:
    def test_a_loopback_measurement_is_not_a_round_trip(self, engine):
        """The endpoint's own loopback is the digital path only: never reported as the round trip."""
        engine._round_trip = {'path': 'loopback', 'measured_ms': 61.4, 'sample_rate': engine.sample_rate,
                              'block': engine.buffer_size}
        stats = engine.get_performance_stats()
        assert stats['measured_round_trip_ms'] is None
        assert stats['round_trip']['measured_ms'] == 61.4

    def test_measuring_is_refused_while_the_engine_runs_or_off_the_native_host(self, engine):
        with pytest.raises(ValueError):
            engine.measure_round_trip(999_999)


class TestPluginBrowser:
    def test_every_module_is_listed_with_why_it_is_not_offered(self, qt_app, tmp_path):
        from tonesphere.plugins import PluginInfo
        from tonesphere.plugins.scan import CRASHED, OK, WRONG_ARCHITECTURE, ScanResult
        from tonesphere.ui.plugin_views import PluginBrowser

        good = PluginInfo(path='a.vst3', uid='1', name='Gain', vendor='V', version='1.0', category='Audio Module Class',
                          subcategories='Fx', sdk_version='3.8', is_audio_effect=True)
        synth = PluginInfo(path='d.vst3', uid='2', name='Synth', vendor='V', version='1.0',
                           category='Audio Module Class', subcategories='Instrument|Synth', sdk_version='3.8',
                           is_audio_effect=True)
        results = [ScanResult('a.vst3', OK, '', [good], 'x64'),
                   ScanResult('d.vst3', OK, '', [synth], 'x64'),
                   ScanResult('b.vst3', CRASHED, 'access violation in the factory', [], 'x64'),
                   ScanResult('c.vst3', WRONG_ARCHITECTURE, 'x86 binary', [], 'x86')]
        browser = PluginBrowser(AudioEngine(), ConfigManager(db_path=tmp_path / "s.db"), picking=True,
                                auto_scan=False)
        chosen = []
        browser.chosen.connect(chosen.append)
        browser.show_results(results)

        assert browser.table.rowCount() == 4
        assert browser.table.item(2, 0).toolTip() == 'access violation in the factory'
        browser.table.selectRow(2)
        assert browser.insert_button.isEnabled() is False, "a crashed module cannot be inserted"
        browser.table.selectRow(1)
        assert browser.insert_button.isEnabled() is False, "an instrument has no MIDI to play it"
        ok, message = AudioEngine().add_plugin(0, synth, is_input=False)
        assert not ok and 'MIDI' in message
        browser.table.selectRow(0)
        browser._insert()
        assert chosen == [good]
