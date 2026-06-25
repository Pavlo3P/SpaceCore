"""Tests for the :func:`spacecore.jax_pytree_class` decorator.

The decorator registers a class as a JAX PyTree node so that instances can
flow through ``jax.tree_util``, ``jax.jit``, ``jax.vmap``, and ``jax.grad``.
When JAX is not installed, the decorator is a no-op.
"""
from __future__ import annotations

import pytest

import spacecore as sc

from tests._helpers import has_jax


@pytest.mark.skipif(not has_jax(), reason="jax is not installed")
def test_jax_pytree_class_is_noop_when_import_fails(monkeypatch):
    """backend-001: force ``from jax import tree_util`` to raise, covering the
    ``except Exception: return klass`` branch of the decorator (jax/_pytree.py
    lines 24-27).

    When JAX is installed we cannot truly uninstall it, so we make the symbol
    the decorator imports unavailable: deleting ``jax.tree_util`` makes
    ``from jax import tree_util`` raise ``ImportError``. The decorator must
    swallow it and return the class unchanged, and the class must remain a
    usable plain class.
    """
    import jax

    # Removing the attribute makes ``from jax import tree_util`` raise.
    monkeypatch.delattr(jax, "tree_util", raising=True)

    @sc.jax_pytree_class
    class Foo:
        def __init__(self, a: int, b: int) -> None:
            self.a = a
            self.b = b

    # Decorator returned the class unchanged (no registration happened).
    inst = Foo(1, 2)
    assert isinstance(inst, Foo)
    assert (inst.a, inst.b) == (1, 2)


def test_jax_pytree_class_is_noop_when_import_fails_no_jax(monkeypatch):
    """backend-001 (no-JAX path): if ``jax`` itself is unimportable the
    decorator returns the class unchanged. Simulated by blocking the import.
    """
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "jax" or name.startswith("jax."):
            raise ImportError("simulated: jax not installed")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    @sc.jax_pytree_class
    class Bar:
        def __init__(self, v: int) -> None:
            self.v = v

    inst = Bar(5)
    assert isinstance(inst, Bar)
    assert inst.v == 5


@pytest.mark.skipif(not has_jax(), reason="jax is not installed")
def test_jax_pytree_class_round_trips_via_tree_util():
    """A decorated class with proper flatten/unflatten round-trips through
    ``jax.tree_util.tree_flatten`` and ``tree_unflatten``.
    """
    import jax.tree_util as jtu

    @sc.jax_pytree_class
    class Pair:
        def __init__(self, x: float, y: float) -> None:
            self.x = x
            self.y = y

        def tree_flatten(self):
            return (self.x, self.y), None

        @classmethod
        def tree_unflatten(cls, _aux, children):
            x, y = children
            inst = cls.__new__(cls)
            inst.x = x
            inst.y = y
            return inst

        def __eq__(self, other):
            return isinstance(other, Pair) and self.x == other.x and self.y == other.y

    inst = Pair(1.0, 2.0)
    leaves, treedef = jtu.tree_flatten(inst)
    assert tuple(leaves) == (1.0, 2.0)
    rebuilt = jtu.tree_unflatten(treedef, leaves)
    assert rebuilt == inst


@pytest.mark.skipif(not has_jax(), reason="jax is not installed")
def test_jax_pytree_class_supports_tree_map():
    """``jax.tree_util.tree_map`` applies a leaf transformation across the
    structure when the class is registered.
    """
    import jax.tree_util as jtu

    @sc.jax_pytree_class
    class Vec3:
        def __init__(self, a, b, c):
            self.a = a
            self.b = b
            self.c = c

        def tree_flatten(self):
            return (self.a, self.b, self.c), None

        @classmethod
        def tree_unflatten(cls, _aux, children):
            inst = cls.__new__(cls)
            inst.a, inst.b, inst.c = children
            return inst

    v = Vec3(1.0, 2.0, 3.0)
    doubled = jtu.tree_map(lambda x: x * 2.0, v)
    assert doubled.a == 2.0 and doubled.b == 4.0 and doubled.c == 6.0


@pytest.mark.skipif(not has_jax(), reason="jax is not installed")
def test_backend_ops_classes_are_themselves_pytree_compatible():
    """Every ``*Ops`` class is decorated, so ``tree_map`` does not error on
    a single ``BackendOps`` instance."""
    import jax.tree_util as jtu

    ops = sc.NumpyOps()
    # NumpyOps as a leaf: tree_map identity-mapping a leaf returns the leaf.
    result = jtu.tree_map(lambda x: x, ops)
    assert isinstance(result, sc.NumpyOps)


@pytest.mark.skipif(not has_jax(), reason="jax is not installed")
def test_jax_pytree_class_handles_redundant_registration():
    """Re-decorating an already-registered class is tolerated (catches the
    JAX ``ValueError`` for duplicate registration internally).
    """
    @sc.jax_pytree_class
    class Once:
        def __init__(self, v):
            self.v = v

        def tree_flatten(self):
            return (self.v,), None

        @classmethod
        def tree_unflatten(cls, _aux, children):
            inst = cls.__new__(cls)
            inst.v = children[0]
            return inst

    # Re-application must not raise.
    Reapplied = sc.jax_pytree_class(Once)
    assert Reapplied is Once
