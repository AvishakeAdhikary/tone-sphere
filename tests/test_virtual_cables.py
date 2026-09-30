"""
The virtual cables' control logic, without the driver: what a refused or deferred Windows
change is reported as, that ToneSphere lets go of a cable it streams through before changing
it (and only then), and the dialog's buttons. That the changes happen to real devices is
proved in the driver VM (`tests/hardware/test_virtual_driver.py`).
"""

import subprocess
import sys

import pytest

from tonesphere.core.engine import AudioEngine
from tonesphere.engine import virtual_cables
from tonesphere.engine.devices import DeviceInfo, HostApi
from tonesphere.engine.virtual_cables import Cable


def device(name: str, output: bool) -> DeviceInfo:
    return DeviceInfo(index=0, name=name, host_api=HostApi.WASAPI, host_api_name='Windows WASAPI',
                      max_input_channels=0 if output else 2, max_output_channels=2 if output else 0,
                      default_samplerate=48000, default_low_input_latency_ms=0.0, default_low_output_latency_ms=0.0,
                      default_high_input_latency_ms=0.0, default_high_output_latency_ms=0.0,
                      endpoint_id=f'{{id}}.{name}')


class TestWhatWindowsSaysIsReportedAsIs:
    def run_pnputil(self, monkeypatch, code, output):
        monkeypatch.setattr(subprocess, 'run',
                            lambda *a, **k: subprocess.CompletedProcess(a[0], code, stdout=output, stderr=''))

    def test_a_change_windows_defers_to_a_restart_is_not_reported_as_done(self, monkeypatch):
        self.run_pnputil(monkeypatch, 3010, "System reboot is needed to complete configuration operations!")
        with pytest.raises(virtual_cables.RestartRequired, match='next restart'):
            virtual_cables._pnputil('/disable-device', 'ROOT\\MEDIA\\0001')

    def test_a_device_pending_a_restart_is_reported_as_such(self, monkeypatch):
        self.run_pnputil(monkeypatch, 50, "Device is pending system reboot to complete a previous operation.")
        with pytest.raises(virtual_cables.RestartRequired):
            virtual_cables._pnputil('/enable-device', 'ROOT\\MEDIA\\0001')

    def test_a_failure_carries_windows_reason(self, monkeypatch):
        self.run_pnputil(monkeypatch, 1, "Failed to remove device: access denied")
        with pytest.raises(virtual_cables.CableError, match='access denied'):
            virtual_cables._pnputil('/remove-device', 'ROOT\\MEDIA\\0001')

    class CfgMgr:
        """CfgMgr32's two calls here, answering from a script and recording each disable's flags."""

        def __init__(self, *answers):
            self.answers, self.flags = list(answers), []

        def CM_Locate_DevNodeW(self, devinst, instance_id, flags):   # noqa: N802 - Windows' name
            return 0

        def CM_Disable_DevNode(self, devinst, flags):   # noqa: N802 - Windows' name
            self.flags.append(flags)
            return self.answers.pop(0) if len(self.answers) > 1 else self.answers[0]

    @pytest.mark.skipif(sys.platform != 'win32', reason="CfgMgr32 is Windows")
    def test_a_veto_is_a_clean_refusal_with_nothing_recorded_for_a_restart(self, monkeypatch):
        cfgmgr = self.CfgMgr(virtual_cables.CR_REMOVE_VETOED)
        monkeypatch.setattr(virtual_cables.ctypes.windll, 'cfgmgr32', cfgmgr)
        monkeypatch.setattr(virtual_cables, 'VETO_PATIENCE_S', 0.0)
        with pytest.raises(virtual_cables.CableInUse, match='has this cable open'):
            virtual_cables._admin_disable('ROOT\\MEDIA\\0001')
        assert cfgmgr.flags and all(f == virtual_cables.CM_DISABLE_UI_NOT_OK for f in cfgmgr.flags), \
            "without UI_NOT_OK a veto becomes a pending restart; with PERSIST it is recorded for one"

    @pytest.mark.skipif(sys.platform != 'win32', reason="CfgMgr32 is Windows")
    def test_a_passing_veto_is_waited_out_and_only_then_made_persistent(self, monkeypatch):
        """The audio engine's own brief look at a cable that just arrived is not a program holding it."""
        cfgmgr = self.CfgMgr(virtual_cables.CR_REMOVE_VETOED, virtual_cables.CR_REMOVE_VETOED,
                             virtual_cables.CR_SUCCESS, virtual_cables.CR_SUCCESS)
        monkeypatch.setattr(virtual_cables.ctypes.windll, 'cfgmgr32', cfgmgr)
        monkeypatch.setattr(virtual_cables.time, 'sleep', lambda s: None)
        assert virtual_cables._admin_disable('ROOT\\MEDIA\\0001') == {'instance_id': 'ROOT\\MEDIA\\0001'}
        persist = virtual_cables.CM_DISABLE_UI_NOT_OK | virtual_cables.CM_DISABLE_PERSIST
        assert cfgmgr.flags == [virtual_cables.CM_DISABLE_UI_NOT_OK] * 3 + [persist]


class TestToneSphereLetsGoOfACableBeforeChangingIt:
    @pytest.fixture
    def engine(self, monkeypatch):
        e = AudioEngine(host_backend='portaudio')
        present = [device('Speakers (Cable A)', True), device('Microphone Array (Cable A)', False),
                   device('Speakers (Cable B)', True), device('Microphone Array (Cable B)', False),
                   device('Speakers (Laptop)', True)]
        monkeypatch.setattr(e, '_enumerate', lambda: list(present))
        e.initialize()
        calls = []
        running = {'now': True}
        monkeypatch.setattr(type(e.host), 'is_running', property(lambda self: running['now']))

        def stop():
            calls.append('stop')
            running['now'] = False

        def start():
            calls.append('start')
            running['now'] = True

        monkeypatch.setattr(e.host, 'stop', stop)
        monkeypatch.setattr(e, 'start_engine', start)
        monkeypatch.setattr(e, 'handle_device_change', lambda: calls.append('reconcile'))
        monkeypatch.setattr(e, '_await_cable_endpoints', lambda names, present: calls.append(
            ('settle', sorted(names), present)))
        monkeypatch.setattr(virtual_cables, 'cables', lambda: [
            Cable('ROOT\\MEDIA\\0000', 'Cable A', True, None, True),
            Cable('ROOT\\MEDIA\\0001', 'Cable B', True, None, True)])
        e.calls = calls
        return e

    def ids(self, engine):
        return {d.name: i for i, d in engine._device_by_id.items()}

    def elevate(self, monkeypatch, engine, error=None):
        def run(operation, *args):
            engine.calls.append(operation)
            if error:
                raise error
            return {'instance_id': args[0] if args else None}
        monkeypatch.setattr(virtual_cables, 'elevate', run)

    def test_a_cable_the_routing_uses_is_released_then_the_engine_starts_again(self, engine, monkeypatch):
        ids = self.ids(engine)
        engine.routing_matrix.create_routing(ids['Microphone Array (Cable B)'], ids['Speakers (Laptop)'])
        self.elevate(monkeypatch, engine)
        ok, _ = engine.manage_virtual_cable('disable', 'ROOT\\MEDIA\\0001')
        assert ok
        assert engine.calls == ['stop', 'disable', ('settle', ['Cable B'], False), 'reconcile', 'start']

    def test_another_cable_is_changed_without_touching_the_running_engine(self, engine, monkeypatch):
        ids = self.ids(engine)
        engine.routing_matrix.create_routing(ids['Microphone Array (Cable B)'], ids['Speakers (Laptop)'])
        self.elevate(monkeypatch, engine)
        ok, _ = engine.manage_virtual_cable('disable', 'ROOT\\MEDIA\\0000')
        assert ok
        assert engine.calls == ['disable', ('settle', ['Cable A'], False), 'reconcile']

    def test_a_refused_change_still_starts_the_engine_again(self, engine, monkeypatch):
        ids = self.ids(engine)
        engine.routing_matrix.create_routing(ids['Speakers (Laptop)'], ids['Speakers (Cable A)'])
        self.elevate(monkeypatch, engine, virtual_cables.CableInUse(virtual_cables.IN_USE))
        ok, message = engine.manage_virtual_cable('remove', 'ROOT\\MEDIA\\0000')
        assert not ok and 'has this cable open' in message
        assert engine.calls == ['stop', 'remove', 'reconcile', 'start'], "nothing changed, so nothing to wait for"

    def test_removing_the_driver_releases_any_cable_in_use(self, engine, monkeypatch):
        ids = self.ids(engine)
        engine.routing_matrix.create_routing(ids['Speakers (Laptop)'], ids['Speakers (Cable A)'])
        self.elevate(monkeypatch, engine)
        assert engine.manage_virtual_cable('remove-driver')[0]
        assert engine.calls == ['stop', 'remove-driver', ('settle', ['Cable A', 'Cable B'], False), 'reconcile',
                                'start']

    def test_an_enabled_cable_is_waited_for_by_its_own_name(self, engine, monkeypatch):
        self.elevate(monkeypatch, engine)
        assert engine.manage_virtual_cable('enable', 'ROOT\\MEDIA\\0001')[0]
        assert engine.calls == ['enable', ('settle', ['Cable B'], True), 'reconcile']

    def test_a_declined_prompt_is_reported_as_cancelled(self, engine, monkeypatch):
        self.elevate(monkeypatch, engine, virtual_cables.ElevationRefused("the administrator prompt was declined"))
        ok, message = engine.manage_virtual_cable('add', 'Chat')
        assert not ok and message.startswith('Cancelled')


class FakeEngine:
    def __init__(self):
        self.cables = [{'instance_id': 'ROOT\\MEDIA\\0000', 'name': 'Cable A', 'state': 'working', 'enabled': True,
                        'render': {'name': 'Speakers (Cable A)', 'id': 0},
                        'capture': {'name': 'Microphone Array (Cable A)', 'id': 1}},
                       {'instance_id': 'ROOT\\MEDIA\\0001', 'name': 'Cable B', 'state': 'working', 'enabled': True,
                        'render': None, 'capture': None}]
        self.calls = []
        self.refuse = None

    def virtual_device_status(self):
        return {'installed': True, 'cables': [dict(c) for c in self.cables], 'error': None,
                'platform_supported': True}

    def manage_virtual_cable(self, operation, *args):
        self.calls.append((operation, *args))
        if self.refuse:
            return False, self.refuse
        for c in self.cables:
            if args and c['instance_id'] == args[0]:
                if operation in ('disable', 'enable'):
                    c['enabled'] = operation == 'enable'
                    c['state'] = 'working' if c['enabled'] else 'disabled'
        return True, operation


class TestTheDialog:
    @pytest.fixture
    def dialog(self, monkeypatch):
        monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
        from PySide6.QtWidgets import QApplication

        from tonesphere.ui.cables_view import CablesDialog

        QApplication.instance() or QApplication([])

        engine = FakeEngine()
        d = CablesDialog(engine)
        assert d.tasks.flush()
        yield d, engine
        d.done(0)

    def test_it_lists_each_cable_with_its_state_and_endpoints(self, dialog):
        d, _ = dialog
        rows = [[d.table.item(r, c).text() for c in range(4)] for r in range(d.table.rowCount())]
        assert rows == [['Cable A', 'working', 'Speakers (Cable A)', 'Microphone Array (Cable A)'],
                        ['Cable B', 'working', '--', '--']]

    def test_disable_then_enable_acts_on_the_selected_cable_only(self, dialog):
        d, engine = dialog
        d.table.selectRow(1)
        assert d.toggle_button.text() == 'Disable'
        d.toggle_selected()
        assert d.tasks.flush() and d.tasks.flush()
        assert engine.calls == [('disable', 'ROOT\\MEDIA\\0001')]
        assert d.table.item(1, 1).text() == 'disabled' and d.table.item(0, 1).text() == 'working'
        assert d.toggle_button.text() == 'Enable', "the selection survives the refresh"
        d.toggle_selected()
        assert d.tasks.flush() and d.tasks.flush()
        assert engine.calls[-1] == ('enable', 'ROOT\\MEDIA\\0001')
        assert d.table.item(1, 1).text() == 'working'

    def test_a_refusal_is_shown_with_windows_reason(self, dialog):
        d, engine = dialog
        engine.refuse = "a program has this cable open"
        d.table.selectRow(0)
        d.remove_selected()
        assert d.tasks.flush() and d.tasks.flush()
        assert d.status.text() == "a program has this cable open"
        assert d.table.rowCount() == 2
