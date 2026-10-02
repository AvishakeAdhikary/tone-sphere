"""The version every push to main releases (`scripts/release_version.py`)."""

import importlib.util
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    'release_version', Path(__file__).resolve().parents[1] / 'scripts' / 'release_version.py')
release_version = importlib.util.module_from_spec(spec)
spec.loader.exec_module(release_version)
next_version = release_version.next_version


def test_a_push_releases_the_next_patch():
    assert next_version(['v0.1.0', 'v0.2.0'], '0.2.0', 'Fix the monitor') == '0.2.1'


def test_tags_are_compared_as_numbers_not_text():
    assert next_version(['v0.2.9', 'v0.2.10'], '0.1.0', 'x') == '0.2.11'


@pytest.mark.parametrize('message, expected', [
    ('Add instruments [minor]', '0.3.0'),
    ('Rewrite the engine [major]', '1.0.0'),
])
def test_the_commit_message_can_ask_for_a_bigger_step(message, expected):
    assert next_version(['v0.2.4'], '0.2.0', message) == expected


def test_pyproject_is_a_floor_for_starting_a_new_series():
    assert next_version(['v0.2.4'], '1.0.0', 'x') == '1.0.0'


def test_the_first_release_is_pyprojects_version():
    assert next_version([], '0.2.0', 'x') == '0.2.0'


def test_other_tags_are_ignored():
    assert next_version(['v0.2.0', 'gpl-source', 'vnext'], '0.1.0', 'x') == '0.2.1'


def test_stamping_writes_every_file_that_reports_the_version(tmp_path, monkeypatch):
    for name in ('pyproject.toml', 'tonesphere/__init__.py', 'packaging/msix/AppxManifest.xml'):
        target = tmp_path / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text((release_version.ROOT / name).read_text(encoding='utf-8'), encoding='utf-8')
    monkeypatch.setattr(release_version, 'ROOT', tmp_path)
    release_version.stamp('3.4.5')
    assert 'version = "3.4.5"' in (tmp_path / 'pyproject.toml').read_text(encoding='utf-8')
    assert '__version__ = "3.4.5"' in (tmp_path / 'tonesphere/__init__.py').read_text(encoding='utf-8')
    manifest = (tmp_path / 'packaging/msix/AppxManifest.xml').read_text(encoding='utf-8')
    assert 'Version="3.4.5.0"' in manifest and 'MinVersion="10.0.17763.0"' in manifest
