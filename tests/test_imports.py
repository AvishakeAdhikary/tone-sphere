"""
Every module must parse and import on every platform.

Motivation: `tonesphere/drivers/linux_jack.py` shipped with a SyntaxError and was never
once loadable. Because the driver registry imports backends inside `try/except
Exception`, the failure was swallowed and JACK simply reported itself unavailable — for
however long that had been true. A broad except around an import hides typos forever, so
the parse has to be checked directly.
"""

import importlib
import pkgutil

import pytest

import tonesphere


def discover_modules():
    return sorted(
        module.name
        for module in pkgutil.walk_packages(tonesphere.__path__, f"{tonesphere.__name__}.")
    )


@pytest.mark.parametrize("module_name", discover_modules())
def test_module_imports(module_name):
    """
    A platform-specific backend may legitimately be missing its third-party library, but
    it must still *parse*. ImportError for an absent optional dependency is acceptable;
    SyntaxError, NameError and friends are not.
    """
    try:
        importlib.import_module(module_name)
    except ImportError as e:
        pytest.skip(f"optional dependency missing: {e}")
    except Exception as e:
        pytest.fail(f"{module_name} failed to import: {type(e).__name__}: {e}")


def test_package_exposes_version():
    assert tonesphere.__version__
