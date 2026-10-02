"""
What a user does in the first minute, without hardware: monitor an input and hear it, patch
a cable and have it play, find the setup as they left it, see why ASIO is not offered.

v0.2.0 failed every one of these on a real machine — a monitor cable created muted on an
engine nobody had started, a master meter that danced with the input while the headphones
stayed silent, a session that vanished on close. The audio itself through these paths is
proven on hardware in tests/hardware/test_first_run.py.
"""

import sys

import pytest
from PySide6.QtCore import QPointF

from tonesphere.core.engine import AudioEngine
from tonesphere.engine.devices import DeviceInfo, HostApi
from tonesphere.ui.monitor_dialog import choose_defaults, hardware_name, is_built_in


def device(name: str, inputs: int, outputs: int, key: str) -> DeviceInfo:
    return DeviceInfo(index=0, name=name, host_api=HostApi.WASAPI, host_api_name='Windows WASAPI',
                      max_input_channels=inputs, max_output_channels=outputs, default_samplerate=48000,
                      default_low_input_latency_ms=0.0, default_low_output_latency_ms=0.0,
                      default_high_input_latency_ms=0.0, default_high_output_latency_ms=0.0,
                      endpoint_id=f'{{id}}.{key}')


LAPTOP_AND_INTERFACE = [
    device('Speakers (Realtek(R) Audio)', 0, 2, 'realtek-out'),
    device('Microphone Array (Intel® Smart Sound Technology for Digital Microphones)', 4, 0, 'intel-mic'),
    device('Speakers (AI-04)', 0, 2, 'ai04-out'),
    device('Line (AI-04)', 2, 0, 'ai04-in'),
]


@pytest.fixture
def engine(monkeypatch):
    e = AudioEngine(host_backend='portaudio')
    monkeypatch.setattr(e, '_enumerate', lambda: list(LAPTOP_AND_INTERFACE))
    e.initialize()
    started = []
    monkeypatch.setattr(e, 'start_engine', lambda: started.append(True))
    e.started = started
    yield e
    e.cleanup()


def ids(engine) -> dict[str, int]:
    return {d.name: i for i, d in engine._device_by_id.items()}


@pytest.fixture
def qt_app(monkeypatch):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication

    return QApplication.instance() or QApplication([])


class TestMonitorChoosesTheInterface:
    def listing(self, engine):
        return engine.get_devices(include_all_backends=True)

    def test_an_interface_is_preferred_over_the_laptops_own_devices(self, engine):
        devices = self.listing(engine)
        inputs = [d for d in devices if d['direction'] == 'input' and d['origin'] != 'loopback']
        outputs = [d for d in devices if d['direction'] == 'output']
        named = ids(engine)
        source, dest, channel = choose_defaults(inputs, outputs, named['Microphone Array (Intel® Smart Sound '
                                                                        'Technology for Digital Microphones)'],
                                                named['Speakers (Realtek(R) Audio)'])
        assert (source, dest, channel) == (named['Line (AI-04)'], named['Speakers (AI-04)'], 0), \
            "the interface's input 1 into its own output"

    def test_with_no_interface_windows_defaults_are_used_whole(self):
        inputs = [{'id': 1, 'name': 'Microphone (Realtek(R) Audio)', 'channels': 2}]
        outputs = [{'id': 2, 'name': 'Speakers (Realtek(R) Audio)', 'channels': 2}]
        assert choose_defaults(inputs, outputs, 1, 2) == (1, 2, None)

    def test_names_match_by_the_hardware_in_brackets(self):
        assert hardware_name('Line (AI-04)') == hardware_name('Speakers (AI-04)') == 'ai-04'
        assert hardware_name('Speakers (Realtek(R) Audio)') == 'realtek(r) audio'
        assert is_built_in('Microphone Array (Intel® Smart Sound Technology)') and not is_built_in('Line (AI-04)')

    def test_the_dialog_warns_about_feedback_only_for_built_in_mic_and_speakers(self, qt_app, engine):
        from tonesphere.ui.monitor_dialog import MonitorDialog

        named = ids(engine)
        dialog = MonitorDialog(engine.get_devices(include_all_backends=True), None, None)
        assert dialog.choice() == (named['Line (AI-04)'], named['Speakers (AI-04)'], 0)
        assert dialog.warning.text() == ''
        dialog.input_combo.setCurrentIndex(dialog.input_combo.findData(
            named['Microphone Array (Intel® Smart Sound Technology for Digital Microphones)']))
        dialog.output_combo.setCurrentIndex(dialog.output_combo.findData(named['Speakers (Realtek(R) Audio)']))
        assert 'feedback' in dialog.warning.text().lower()


class TestMonitorIsHeard:
    def test_monitor_makes_an_unmuted_route_from_the_chosen_channel_and_starts(self, engine):
        named = ids(engine)
        source, dest = named['Line (AI-04)'], named['Speakers (AI-04)']
        ok, message = engine.monitor(source, dest, 0)
        assert ok, message
        route = engine.routing_matrix.connections[(source, dest)]
        assert not route.muted and route.source_channel == 0
        assert engine.started, "the engine must be running, or nothing is heard"
        assert engine._build_graph().connections[0].source_channel == 0

    def test_monitor_unmutes_a_route_that_already_exists(self, engine):
        named = ids(engine)
        source, dest = named['Line (AI-04)'], named['Speakers (AI-04)']
        engine.create_routing(source, dest)
        engine.set_routing_mute(source, dest, True)
        assert engine.monitor(source, dest, None)[0]
        assert not engine.routing_matrix.connections[(source, dest)].muted

    def test_a_channel_the_input_does_not_have_is_refused(self, engine):
        named = ids(engine)
        ok, message = engine.monitor(named['Line (AI-04)'], named['Speakers (AI-04)'], 5)
        assert not ok and 'channel 6' in message


class TestTheSessionComesBack:
    def test_routes_gains_and_the_input_channel_survive_a_restart(self, engine, monkeypatch, tmp_path):
        named = ids(engine)
        source, dest = named['Line (AI-04)'], named['Speakers (AI-04)']
        engine.monitor(source, dest, 0)
        engine.set_routing_volume_db(source, dest, -6.0)
        engine.save_session()

        again = AudioEngine(host_backend='portaudio')
        monkeypatch.setattr(again, '_enumerate', lambda: list(LAPTOP_AND_INTERFACE))
        again.initialize()
        try:
            assert again.restore_session() is None, "everything was restored"
            route = again.routing_matrix.connections[(ids(again)['Line (AI-04)'], ids(again)['Speakers (AI-04)'])]
            assert route.source_channel == 0 and not route.muted
            assert route.volume == pytest.approx(10 ** (-6 / 20), rel=1e-3)
        finally:
            again.cleanup()

    def test_a_missing_device_is_said_not_silently_dropped(self, engine, monkeypatch):
        named = ids(engine)
        engine.monitor(named['Line (AI-04)'], named['Speakers (AI-04)'], 0)
        engine.save_session()

        unplugged = AudioEngine(host_backend='portaudio')
        monkeypatch.setattr(unplugged, '_enumerate', lambda: LAPTOP_AND_INTERFACE[:2])
        unplugged.initialize()
        try:
            summary = unplugged.restore_session()
            assert summary and 'AI-04' in summary
        finally:
            unplugged.cleanup()

    def test_no_session_yet_is_a_clean_start(self, engine):
        assert engine.restore_session() is None


class TestThePatchbayConnectsWhereTheUserLetsGo:
    @pytest.fixture
    def scene(self, qt_app):
        from tonesphere.ui.routing_view import RoutingScene

        scene = RoutingScene()
        scene.add_node(1, 'Line (AI-04)', 'WASAPI', can_input=False, can_output=True, is_bus=False,
                       position=QPointF(0, 0))
        scene.add_node(2, 'Speakers (AI-04)', 'WASAPI', can_input=True, can_output=False, is_bus=False,
                       position=QPointF(400, 0))
        scene.add_node(3, 'Microphone Array (Intel)', 'WASAPI', can_input=False, can_output=True, is_bus=False,
                       position=QPointF(400, 200))
        connected, refused = [], []
        scene.connect_requested.connect(lambda s, d: connected.append((s, d)))
        scene.connect_refused.connect(refused.append)
        return scene, connected, refused

    def drop(self, scene, at: QPointF):
        scene.begin_cable(scene.nodes[1].output_port)
        scene._finish_cable(at)

    def test_anywhere_on_the_node_connects(self, scene):
        scene, connected, _ = scene
        for point in (QPointF(410, 10), QPointF(560, 50), QPointF(480, 28)):
            self.drop(scene, point)
        assert connected == [(1, 2)] * 3

    def test_just_beside_the_node_connects(self, scene):
        scene, connected, _ = scene
        self.drop(scene, QPointF(385, 28))
        assert connected == [(1, 2)]

    def test_empty_space_and_an_input_say_why_nothing_connected(self, scene):
        scene, connected, refused = scene
        self.drop(scene, QPointF(250, 600))
        self.drop(scene, QPointF(450, 220))
        assert connected == []
        assert len(refused) == 2 and 'Microphone Array (Intel)' in refused[1]


class TestTheWindowStartsReady:
    def test_launch_restores_and_starts_the_engine_and_lists_asio_with_its_reason(self, qt_app, tmp_path,
                                                                                  monkeypatch):
        from tonesphere.ui.main_window import MainWindow
        from tonesphere.utils.config import ConfigManager

        window = MainWindow(ConfigManager(db_path=tmp_path / 'settings.db'))
        try:
            assert window.tasks.flush(30)
            assert window.engine.engine._started, "the engine is running after launch, not waiting for a button"
            if window._view.get('plugin_host') and 'ASIO' not in window._view['drivers']:
                combo = window.backend_combo
                last = combo.model().item(combo.count() - 1)
                assert 'ASIO' in last.text() and not last.isEnabled() and last.toolTip()
        finally:
            window.close()

    def test_the_master_meter_reads_the_outputs_not_the_loudest_input(self, qt_app, tmp_path, monkeypatch):
        from tonesphere.ui.main_window import MainWindow
        from tonesphere.utils.config import ConfigManager

        window = MainWindow(ConfigManager(db_path=tmp_path / 'settings.db'))
        try:
            assert window.tasks.flush(30)
            # Ids no real device has, so the machine's own strips take no part.
            window._view['devices'] = [{'id': 9101, 'direction': 'input'}, {'id': 9102, 'direction': 'output'}]

            def side(db):
                return {'peak_db': db, 'rms_db': db - 3, 'peak_hold_db': db, 'clipped': False,
                        'channel_peak_db': [db, db], 'channel_rms_db': [db - 3, db - 3],
                        'channel_peak_hold_db': [db, db]}

            window._update_meters({9101: {**side(-3.0), 'sides': {'input': side(-3.0)}},
                                   9102: {**side(-40.0), 'sides': {'output': side(-40.0)}}})
            assert window.master_strip.meter._peaks[0] == pytest.approx(-40.0)
        finally:
            window.close()


class TestADoubleClickOpensTheApp:
    def test_no_arguments_starts_the_interface(self, monkeypatch):
        """v0.2.0's exe printed usage to a console a windowed build does not have, and exited."""
        import threading

        import main
        import tonesphere.ui

        # main() installs the crash hooks process-wide; these put pytest's own back afterwards.
        monkeypatch.setattr(sys, 'excepthook', sys.excepthook)
        monkeypatch.setattr(threading, 'excepthook', threading.excepthook)
        opened = []
        monkeypatch.setattr(sys, 'argv', ['ToneSphere.exe'])
        monkeypatch.setattr(tonesphere.ui, 'run', lambda config: opened.append(config) or 0)
        assert main.main() == 0
        assert opened, "a launch with no arguments must open the window"

    def test_help_is_still_there_for_a_terminal(self, monkeypatch, capsys):
        import main

        monkeypatch.setattr(sys, 'argv', ['ToneSphere.exe', 'help'])
        assert main.main() == 0
        assert 'ToneSphere server' in capsys.readouterr().out


class TestAFailureLeavesAReport:
    def test_an_uncaught_exception_is_written_to_the_logs_folder(self):
        from tonesphere.utils import crash
        from tonesphere.utils.logger import default_log_dir

        try:
            raise RuntimeError("the device vanished mid-start")
        except RuntimeError as e:
            path = crash.write_report(type(e), e, e.__traceback__, 'main thread')
        assert path is not None and path.parent == default_log_dir()
        text = path.read_text(encoding='utf-8')
        assert 'the device vanished mid-start' in text and 'Traceback' in text and 'main thread' in text
