"""
The control plane's locking rule, applied the same way to every class that owns engine state.

Each such object has one reentrant lock, `self._lock`, and `synchronized` makes every public
method and property take it. That is deliberately coarse: control calls are rare next to
audio blocks, and a single lock per object cannot deadlock against itself, where a lock per
field would need an ordering nobody can check by reading one method. The order between
objects is fixed — `AudioEngine._lock` before a host's `_lock`, never the reverse — and
nothing holding a host's lock calls back into the engine.

The audio thread never takes any of these (it is native and never enters Python), and
nothing here is a real-time guarantee: a data-path caller such as a network playout thread
can wait while the UI's worker reconfigures the engine. What it guarantees is that every
caller — the Qt worker, the API's threadpool, the CLI, network and capture threads — sees
each change whole.
"""

import functools
from collections.abc import Callable, Iterable

LOCKED = '__tonesphere_locked__'


def _locked(fn: Callable) -> Callable:
    @functools.wraps(fn)
    def call(self, *args, **kwargs):
        with self._lock:
            return fn(self, *args, **kwargs)
    setattr(call, LOCKED, True)
    return call


def synchronized(unlocked: Iterable[str] = ()):
    """
    Class decorator: wrap every public method and property of the class itself in its
    instance's `_lock`, except the names in `unlocked` — data paths that must never wait
    on the control lock, because the threads that call them are joined while it is held.
    Each exception is named where the decorator is applied, with its reason.
    """
    exempt = frozenset(unlocked)

    def apply(cls):
        for name, member in list(vars(cls).items()):
            if name.startswith('_') or name in exempt:
                continue
            if isinstance(member, property):
                setattr(cls, name, property(
                    _locked(member.fget) if member.fget else None,
                    _locked(member.fset) if member.fset else None,
                    member.fdel, member.__doc__))
            elif callable(member) and not isinstance(member, (staticmethod, classmethod, type)):
                setattr(cls, name, _locked(member))
        return cls

    return apply


def is_locked(member) -> bool:
    """For tests: whether a method or property was wrapped by `synchronized`."""
    if isinstance(member, property):
        return all(getattr(f, LOCKED, False) for f in (member.fget, member.fset) if f is not None)
    return getattr(member, LOCKED, False)
