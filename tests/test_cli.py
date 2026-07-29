"""
CLI commands that turned out to be stubs.

`ruff` flagged `_delete_virtual_device` as computing `device_id` and never using it — the
method printed "not yet implemented" no matter what you entered. The engine has had a
working `delete_virtual_device` since Phase 1; this command was simply never wired to it,
which is the exact "looks like it works but doesn't" pattern the rest of this project was
rebuilt to eliminate.
"""

from unittest.mock import patch

import pytest

from tonesphere.cli.interface import AudioEngineCLI


@pytest.fixture
def cli():
    instance = AudioEngineCLI()
    instance.initialize_engine()
    yield instance
    if instance.engine:
        instance.engine.cleanup()


class TestDeleteVirtualDevice:
    def test_deletes_a_real_bus(self, cli, capsys):
        bus_id = cli.engine.create_virtual_input("Test bus")

        with patch('builtins.input', return_value=str(bus_id)):
            cli._delete_virtual_device()

        output = capsys.readouterr().out
        assert "Deleted" in output
        assert cli.engine.engine.get_device_info is not None
        assert bus_id not in cli.engine.engine._bus_meta

    def test_unknown_id_reports_failure_not_silence(self, cli, capsys):
        with patch('builtins.input', return_value="999999"):
            cli._delete_virtual_device()

        output = capsys.readouterr().out
        assert "No such bus" in output
        assert "not yet implemented" not in output

    def test_non_numeric_input_is_rejected(self, cli, capsys):
        with patch('builtins.input', return_value="not-a-number"):
            cli._delete_virtual_device()

        assert "Invalid device ID" in capsys.readouterr().out

    def test_without_an_engine_it_says_so_rather_than_crashing(self, capsys):
        instance = AudioEngineCLI()

        instance._delete_virtual_device()

        assert "not initialized" in capsys.readouterr().out.lower()
