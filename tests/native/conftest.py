"""
Native tests need `tonesphere_native.dll`. Off Windows they are skipped; on Windows a
missing DLL is a failure, not a skip — a native suite that quietly skipped would look
exactly like one that passed. Build it with `uv run python scripts/build_native.py`.
"""

import sys

import pytest


def pytest_collection_modifyitems(config, items):
    if sys.platform == "win32":
        return
    skip = pytest.mark.skip(reason="the native engine is Windows-only")
    for item in items:
        if "tests/native" in item.nodeid.replace("\\", "/"):
            item.add_marker(skip)
