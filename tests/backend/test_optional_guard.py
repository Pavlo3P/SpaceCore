"""Guard behavior for optional backend imports (``backend._optional``).

An optional backend can fail to load in several distinct ways, and only one of
them means "not installed":

* **absent** — the dependency is simply not installed (``ModuleNotFoundError``
  naming the dependency itself);
* **broken / shadowed** — the dependency *is* importable but its import fails
  (a partial install, an ABI mismatch, or a namespace shim), raising a plain
  ``ImportError``;
* **transitive-missing** — a *different* module the backend needs is absent
  (``ModuleNotFoundError`` naming something other than the dependency).

Only *absent* is a routine condition; the others are installation faults. In
every case loading an optional backend must degrade gracefully rather than
abort ``import spacecore`` — crashing the whole library over an optional
backend the user was never going to use is the wrong failure mode (Ousterhout,
*A Philosophy of Software Design*: crash on invariant violation, handle
environmental errors gracefully; Hunt & Thomas, *The Pragmatic Programmer*:
fail, but do not corrupt).

These tests replace ``importlib.import_module`` with a *Test Stub* (Meszaros,
*xUnit Test Patterns*) so the SUT is isolated from whatever backends happen to
be installed — the suite must be *Repeatable* regardless of the environment
(same source, principle: isolate the SUT from irrelevant dependencies), and
must not become an *Erratic Test*.
"""
from __future__ import annotations

import warnings

import pytest

import spacecore as sc
from spacecore.backend import _optional


@pytest.fixture(autouse=True)
def _fresh_discovery_cache():
    """Run every test in this module against uncached backend discovery.

    ``available_ops`` is memoized for the process, which is right in production
    (availability cannot change within a run, and re-probing makes one broken
    install warn repeatedly) but wrong here: these tests monkeypatch the import
    machinery, then assert on what discovery *now* returns; against a warm cache
    they would read the pre-patch answer.

    Clearing afterwards as well stops a patched result leaking into a later test.
    """
    _optional.available_ops.cache_clear()
    yield
    _optional.available_ops.cache_clear()


def _raises(exc):
    def _stub(*_args, **_kwargs):
        raise exc

    return _stub


class TestImportBackendGuard:
    def test_absent_dependency_returns_none_without_warning(self, monkeypatch):
        # ModuleNotFoundError naming the dependency itself == "not installed".
        monkeypatch.setattr(
            _optional.importlib,
            "import_module",
            _raises(ModuleNotFoundError("No module named 'jax'", name="jax")),
        )
        with warnings.catch_warnings():
            warnings.simplefilter("error")  # any warning would fail the test
            assert _optional.import_backend(".jax", "jax") is None

    def test_broken_backend_warns_and_skips(self, monkeypatch):
        # Installed-but-broken raises a plain ImportError (not ModuleNotFound).
        monkeypatch.setattr(
            _optional.importlib,
            "import_module",
            _raises(ImportError("cannot import name 'abs' from partially initialized module")),
        )
        with pytest.warns(UserWarning):
            assert _optional.import_backend(".cupy", "cupy") is None

    def test_transitive_missing_dependency_warns_and_skips(self, monkeypatch):
        # ModuleNotFoundError for a *different* module (a transitive dependency)
        # must not be treated as fatal / re-raised.
        monkeypatch.setattr(
            _optional.importlib,
            "import_module",
            _raises(ModuleNotFoundError("No module named 'cupyx'", name="cupyx")),
        )
        with pytest.warns(UserWarning):
            assert _optional.import_backend(".cupy", "cupy") is None

    def test_successful_import_is_returned(self, monkeypatch):
        sentinel = object()
        monkeypatch.setattr(_optional.importlib, "import_module", lambda *a, **k: sentinel)
        assert _optional.import_backend(".jax", "jax") is sentinel


class TestAvailableOpsResilience:
    def test_numpy_survives_when_every_optional_backend_is_broken(self, monkeypatch):
        # Even if all optional backends fail to import, NumpyOps remains and no
        # exception escapes.
        monkeypatch.setattr(
            _optional.importlib, "import_module", _raises(ImportError("boom"))
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ops = _optional.available_ops()
        names = [o.__name__ for o in ops]
        assert names == ["NumpyOps"]

    def test_available_ops_starts_with_numpy(self):
        # Guard the real environment: NumpyOps is always first, unconditionally.
        assert _optional.available_ops()[0] is sc.NumpyOps
