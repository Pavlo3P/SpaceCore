"""Tests for the scoped ambient state behind ``use_context`` / ``use_check_level``.

The ambient default context and validation level are *swappable* global state.
They used to be plain module globals mutated by save-and-restore, which is the
Global Data smell in its most dangerous form (Fowler, *Refactoring* 2e, ch. 3:
"Global data is especially nasty when it's mutable"). They now live in
:class:`contextvars.ContextVar`, so an override is scoped to the current thread /
async task and unwinds through a ``Token``.

Note the asymmetry these tests pin, which is the whole point of the design: the
*registry* half of ``Contextual`` (``register_ops`` / ``available_ops``) is
monotonic and stays process-wide, while only the *policy* half is scoped.
Scoping the registry would be a bug — a backend registered inside a ``with``
block would vanish on exit.

Concurrency assertions use an explicit :class:`threading.Barrier` rather than
sleeps so the overlap is forced, not raced for (Meszaros, *xUnit Test Patterns*:
avoid the Erratic Test smell). Each test states one property.
"""
from __future__ import annotations

import asyncio
import threading
from contextvars import copy_context

import numpy as np
import pytest

import spacecore as sc
from spacecore.backend import ops_registry


def _ctx(dtype):
    """A context distinguishable by dtype alone, so no second backend is needed."""
    return sc.Context(sc.NumpyOps(), dtype=dtype)


F32 = np.dtype(np.float32)
F64 = np.dtype(np.float64)


# ---------------------------------------------------------------------------
# Nesting and unwinding
# ---------------------------------------------------------------------------
class TestNesting:
    def test_check_level_unwinds_to_each_enclosing_level(self):
        outer = sc.get_check_level()
        with sc.use_check_level("strict"):
            assert sc.get_check_level() == "strict"
            with sc.use_check_level("none"):
                assert sc.get_check_level() == "none"
            assert sc.get_check_level() == "strict"
        assert sc.get_check_level() == outer

    def test_context_unwinds_to_each_enclosing_context(self):
        outer = sc.get_context()
        with sc.use_context(_ctx(F32)):
            assert sc.get_context().dtype == F32
            with sc.use_context(_ctx(F64)):
                assert sc.get_context().dtype == F64
            assert sc.get_context().dtype == F32
        assert sc.get_context() == outer

    def test_override_unwinds_on_exception(self):
        outer = sc.get_check_level()
        with pytest.raises(RuntimeError):
            with sc.use_check_level("strict"):
                raise RuntimeError("boom")
        assert sc.get_check_level() == outer


# ---------------------------------------------------------------------------
# ``set_*`` moves the baseline; ``use_*`` installs a scoped override
# ---------------------------------------------------------------------------
class TestBaselineVersusOverride:
    def test_set_check_level_does_not_disturb_an_active_override(self):
        with sc.use_check_level("strict"):
            sc.set_check_level("none")           # moves the baseline only
            assert sc.get_check_level() == "strict"
        assert sc.get_check_level() == "none"    # ...revealed once the scope exits

    def test_set_context_from_a_worker_thread_is_visible_in_main(self):
        """``set_context`` is process-wide by contract, unlike ``use_context``."""
        previous = sc.get_context()
        try:
            t = threading.Thread(target=lambda: sc.set_context(_ctx(F32)))
            t.start()
            t.join()
            assert sc.get_context().dtype == F32
        finally:
            sc.set_context(previous)

    def test_use_context_in_a_worker_thread_is_not_visible_in_main(self):
        started, release = threading.Event(), threading.Event()

        def worker():
            with sc.use_context(_ctx(F32)):
                started.set()
                release.wait(timeout=5)

        t = threading.Thread(target=worker)
        t.start()
        try:
            assert started.wait(timeout=5)
            assert sc.get_context().dtype != F32   # main keeps the baseline
        finally:
            release.set()
            t.join()


# ---------------------------------------------------------------------------
# Isolation between concurrent scopes — the bug this design fixes
# ---------------------------------------------------------------------------
class TestConcurrentIsolation:
    def test_threads_do_not_clobber_each_others_check_level(self):
        """Two overlapping scopes, with the interleaving pinned by events rather
        than left to the scheduler:

            A enters "strict" -> B enters "none" -> A reads -> B reads

        Over a shared global, A's read at step 3 returns B's "none". Each thread
        now reads its own ``ContextVar``, so both see what they set.
        """
        seen: dict[str, str] = {}
        a_in, b_in, a_read = threading.Event(), threading.Event(), threading.Event()

        def thread_a() -> None:
            with sc.use_check_level("strict"):
                a_in.set()
                assert b_in.wait(timeout=5)       # B is now inside its own scope
                seen["a"] = sc.get_check_level()
                a_read.set()

        def thread_b() -> None:
            assert a_in.wait(timeout=5)           # A is already inside its scope
            with sc.use_check_level("none"):
                b_in.set()
                assert a_read.wait(timeout=5)     # hold the scope open past A's read
                seen["b"] = sc.get_check_level()

        threads = [threading.Thread(target=thread_a), threading.Thread(target=thread_b)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)

        assert seen == {"a": "strict", "b": "none"}

    def test_interleaved_coroutines_keep_their_own_check_level(self):
        """Each asyncio task copies the ambient context at creation, so suspending
        inside a ``use_check_level`` block cannot clobber a sibling task."""

        async def task(level: str, out: dict, key: str) -> None:
            with sc.use_check_level(level):
                await asyncio.sleep(0)           # suspend *inside* the scope
                out[key] = sc.get_check_level()

        async def main() -> dict:
            out: dict[str, str] = {}
            await asyncio.gather(task("strict", out, "x"), task("none", out, "y"))
            return out

        outer = sc.get_check_level()
        assert asyncio.run(main()) == {"x": "strict", "y": "none"}
        assert sc.get_check_level() == outer     # and nothing leaked out


# ---------------------------------------------------------------------------
# Thread non-inheritance is deliberate, and has a documented escape hatch
# ---------------------------------------------------------------------------
class TestThreadInheritance:
    def test_spawned_thread_sees_the_baseline_not_the_enclosing_override(self):
        """Pinned so the documented behaviour cannot change silently: a scoped
        override must not leak into threads spawned inside it. See the caveat in
        ``use_context``'s docstring."""
        box: dict[str, str] = {}
        baseline = sc.get_check_level()
        with sc.use_check_level("strict"):
            t = threading.Thread(target=lambda: box.update(v=sc.get_check_level()))
            t.start()
            t.join()
        assert box["v"] == baseline

    def test_copy_context_propagates_the_override_on_purpose(self):
        """The documented way to opt in, for thread pools."""
        box: dict[str, str] = {}
        with sc.use_check_level("strict"):
            snapshot = copy_context()
            t = threading.Thread(
                target=lambda: snapshot.run(lambda: box.update(v=sc.get_check_level()))
            )
            t.start()
            t.join()
        assert box["v"] == "strict"


# ---------------------------------------------------------------------------
# The registry half stays process-wide
# ---------------------------------------------------------------------------
def test_backend_registry_is_not_scoped_by_use_context():
    """``available_ops`` is monotonic state, not ambient policy: entering and
    leaving a scope must not add or remove backends."""

    before = dict(ops_registry.classes())
    with sc.use_context(_ctx(F32)):
        assert dict(ops_registry.classes()) == before
    assert dict(ops_registry.classes()) == before


def test_scoped_context_is_honoured_by_object_construction():
    """The override must reach every resolution path, not just ``get_context``:
    ``normalize_context(None)`` inside ``ContextBound.__init__`` reads through the
    same property."""
    with sc.use_context(_ctx(F32)):
        space = sc.DenseCoordinateSpace((2,))
        assert space.ctx.dtype == F32
