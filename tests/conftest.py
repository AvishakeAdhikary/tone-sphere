"""
Every test runs with its own per-user data folder: the app now restores and autosaves the
last session, caches plugin scans and writes logs there, and a test must neither read the
developer's real setup nor leave its own behind in it.
"""

import os

import pytest


@pytest.fixture(scope='session', autouse=True)
def isolated_user_data(tmp_path_factory):
    home = tmp_path_factory.mktemp('user-data')
    saved = os.environ.get('TONESPHERE_DATA_DIR')
    os.environ['TONESPHERE_DATA_DIR'] = str(home)
    yield home
    if saved is None:
        os.environ.pop('TONESPHERE_DATA_DIR', None)
    else:
        os.environ['TONESPHERE_DATA_DIR'] = saved
