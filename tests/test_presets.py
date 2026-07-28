"""
Presets.

The design point being tested: devices are referenced by stable key, never by PortAudio
index, and recall is partial rather than all-or-nothing. A preset saved with an interface
attached has to load usefully when that interface is absent, and say what it could not do.
"""

import pytest
import yaml

from tonesphere.core.engine import AudioEngine
from tonesphere.core.presets import PresetManager, RecallResult


@pytest.fixture
def engine():
    e = AudioEngine()
    e.initialize()
    yield e
    e.cleanup()


@pytest.fixture
def manager(engine, tmp_path):
    return PresetManager(engine, directory=tmp_path)


class TestCapture:
    def test_captures_buses(self, engine, manager):
        engine.create_virtual_input("Guitar", channels=2)
        engine.create_virtual_output("Phones", channels=2)

        preset = manager.capture("test")

        names = {bus['name'] for bus in preset['buses']}
        assert names == {"Guitar", "Phones"}

    def test_captures_routes_with_their_parameters(self, engine, manager):
        source = engine.create_virtual_input("A")
        dest = engine.create_virtual_output("B")
        engine.create_routing(source, dest, 0.5)
        engine.set_routing_pan(source, dest, -0.4)
        engine.set_routing_mute(source, dest, True)

        preset = manager.capture("test")

        assert len(preset['routes']) == 1
        route = preset['routes'][0]
        assert route['gain'] == pytest.approx(0.5)
        assert route['pan'] == pytest.approx(-0.4)
        assert route['muted'] is True

    def test_references_are_names_not_indices(self, engine, manager):
        """
        PortAudio indices shift whenever hardware is plugged in. A preset keyed on them
        would silently point at a different device after any change.
        """
        source = engine.create_virtual_input("Guitar")
        dest = engine.create_virtual_output("Phones")
        engine.create_routing(source, dest)

        preset = manager.capture("test")
        route = preset['routes'][0]

        assert route['source'] == "bus:Guitar"
        assert route['dest'] == "bus:Phones"
        assert not route['source'].split(':')[-1].isdigit()

    def test_captures_engine_settings(self, engine, manager):
        preset = manager.capture("test")

        assert preset['engine']['sample_rate'] == engine.sample_rate
        assert preset['engine']['buffer_size'] == engine.buffer_size

    def test_includes_a_version(self, manager):
        """So a future format change can be detected rather than misread."""
        assert manager.capture("test")['version'] >= 1


class TestRecall:
    def test_round_trips_a_patch(self, engine, manager):
        source = engine.create_virtual_input("Guitar")
        dest = engine.create_virtual_output("Phones")
        engine.create_routing(source, dest, 0.75)

        preset = manager.capture("test")
        engine.clear_all_routing()
        assert len(engine.get_routing_matrix()) == 0

        result = manager.apply(preset)

        assert result.applied is True
        assert result.restored_routes == 1

        routes = list(engine.get_routing_matrix().values())
        assert len(routes) == 1
        assert routes[0]['volume'] == pytest.approx(0.75)

    def test_restores_pan_and_polarity(self, engine, manager):
        source = engine.create_virtual_input("A")
        dest = engine.create_virtual_output("B")
        engine.create_routing(source, dest)
        engine.set_routing_pan(source, dest, 0.6)
        engine.set_routing_invert(source, dest, True)

        preset = manager.capture("test")
        engine.clear_all_routing()
        manager.apply(preset)

        route = list(engine.get_routing_matrix().values())[0]
        assert route['pan'] == pytest.approx(0.6)
        assert route['inverted'] is True

    def test_restores_channel_settings(self, engine, manager):
        bus = engine.create_virtual_input("A")
        engine.set_device_master_volume(bus, 0.3)
        engine.set_channel_mute(bus, 0, True)

        preset = manager.capture("test")

        engine.set_device_master_volume(bus, 1.0)
        engine.set_channel_mute(bus, 0, False)

        manager.apply(preset)

        control = engine.channel_control_manager.device_controls[bus]
        assert control.master_volume == pytest.approx(0.3)
        assert control.channels[0].muted is True

    def test_missing_device_is_reported_not_fatal(self, engine, manager):
        """
        A preset saved with an interface attached must still restore everything else when
        the interface is gone, and say what was missing.
        """
        source = engine.create_virtual_input("Guitar")
        dest = engine.create_virtual_output("Phones")
        engine.create_routing(source, dest)

        preset = manager.capture("test")
        preset['routes'].append({
            'source': 'device:Nonexistent Interface',
            'dest': 'bus:Phones',
            'gain': 1.0, 'muted': False, 'pan': 0.0, 'inverted': False,
        })

        engine.clear_all_routing()
        result = manager.apply(preset)

        assert result.applied is True
        assert result.restored_routes == 1, "the possible route should still be restored"
        assert result.skipped_routes == 1
        assert result.missing_devices
        assert result.is_complete is False

    def test_applying_twice_does_not_duplicate_buses(self, engine, manager):
        engine.create_virtual_input("Guitar")
        preset = manager.capture("test")

        manager.apply(preset)
        manager.apply(preset)

        names = [meta['name'] for meta in engine._bus_meta.values()]
        assert names.count("Guitar") == 1

    def test_recall_replaces_rather_than_merges(self, engine, manager):
        """Loading a preset should give you that preset, not it plus whatever was there."""
        a = engine.create_virtual_input("A")
        b = engine.create_virtual_output("B")
        engine.create_routing(a, b)
        preset = manager.capture("one route")

        c = engine.create_virtual_input("C")
        engine.create_routing(c, b)
        assert len(engine.get_routing_matrix()) == 2

        manager.apply(preset)
        assert len(engine.get_routing_matrix()) == 1

    def test_newer_format_version_warns_rather_than_failing(self, engine, manager):
        preset = manager.capture("test")
        preset['version'] = 999

        result = manager.apply(preset)

        assert result.applied is True
        assert any('newer' in w for w in result.warnings)


class TestFiles:
    def test_save_and_load(self, engine, manager, tmp_path):
        source = engine.create_virtual_input("Guitar")
        dest = engine.create_virtual_output("Phones")
        engine.create_routing(source, dest, 0.6)

        path = manager.save("My Rig")
        assert path.is_file()

        engine.clear_all_routing()
        result = manager.load(path)

        assert result.restored_routes == 1

    def test_saved_file_is_readable_yaml(self, engine, manager):
        """
        Plain YAML on purpose: a preset should be inspectable and hand-editable, not an
        opaque blob.
        """
        engine.create_virtual_input("Guitar")
        path = manager.save("My Rig")

        with open(path, encoding='utf-8') as handle:
            data = yaml.safe_load(handle)

        assert data['name'] == "My Rig"
        assert 'routes' in data

    def test_filename_is_sanitised(self, engine, manager):
        path = manager.save("My Rig: <weird> / name")

        assert path.is_file()
        assert '<' not in path.name
        assert '/' not in path.name

    def test_listing_finds_saved_presets(self, engine, manager):
        manager.save("First")
        manager.save("Second")

        found = manager.list_presets()
        names = {item['name'] for item in found}

        assert names == {"First", "Second"}

    def test_loading_a_missing_file_reports_rather_than_raising(self, manager, tmp_path):
        result = manager.load(tmp_path / "nope.yaml")

        assert result.applied is False
        assert result.warnings

    def test_loading_a_corrupt_file_reports_rather_than_raising(self, manager, tmp_path):
        bad = tmp_path / "bad.yaml"
        bad.write_text("this: is: not: valid: yaml:", encoding='utf-8')

        result = manager.load(bad)

        assert result.applied is False
        assert result.warnings

    def test_listing_skips_unreadable_files_without_failing(self, manager, tmp_path):
        manager.save("Good")
        (tmp_path / "broken.yaml").write_text("{[}", encoding='utf-8')

        found = manager.list_presets()

        assert any(item['name'] == "Good" for item in found)


class TestRecallResult:
    def test_complete_recall_is_reported_as_complete(self):
        assert RecallResult(applied=True, restored_routes=3).is_complete is True

    def test_partial_recall_is_not_complete(self):
        result = RecallResult(applied=True, restored_routes=2, skipped_routes=1)
        assert result.is_complete is False

    def test_summary_mentions_what_was_missing(self):
        result = RecallResult(applied=True, restored_routes=1,
                              missing_devices=["Focusrite Scarlett"])
        assert "Focusrite" in result.summary()
