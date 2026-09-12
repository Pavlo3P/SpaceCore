"""Tests for the backend-neutral container registry (:mod:`spacecore.backend._container`).

The wiring contract is exercised against a *fake* registrar rather than a real
backend: a registrar is just ``Callable[[type], None]`` and
:class:`PyTreeRegistry` is instantiable, so the whole class x backend
cross-product logic is testable with no backend installed and no arrays. Only
the integration test that asserts against a real backend's registry needs one.
"""
from __future__ import annotations

import inspect
from abc import abstractmethod

import pytest

from spacecore.backend._container import PyTreeNode, PyTreeRegistry, registry as global_registry

from tests._helpers import has_jax, has_torch


class _Recorder:
    """Fake tree protocol: records the classes handed to it."""

    def __init__(self) -> None:
        self.seen: list[type] = []

    def __call__(self, cls: type) -> None:
        self.seen.append(cls)


class _Alpha:
    pass


class _Beta:
    pass


# --------------------------------------------------------------------------
# Two-axis back-fill
# --------------------------------------------------------------------------

def test_class_then_backend_wires_the_pair():
    """Backend arriving after the class back-fills it (backend loaded late)."""
    reg, rec = PyTreeRegistry(), _Recorder()
    reg.register_class(_Alpha)
    assert rec.seen == []
    reg.register_backend("fake", rec)
    assert rec.seen == [_Alpha]
    assert reg.is_wired("fake", _Alpha)


def test_backend_then_class_wires_the_pair():
    """Class arriving after the backend forward-fills (the normal import order)."""
    reg, rec = PyTreeRegistry(), _Recorder()
    reg.register_backend("fake", rec)
    reg.register_class(_Alpha)
    assert rec.seen == [_Alpha]
    assert reg.is_wired("fake", _Alpha)


def test_wiring_is_order_independent():
    """Either arrival order must reach the same set of wired pairs."""
    a, rec_a = PyTreeRegistry(), _Recorder()
    a.register_class(_Alpha)
    a.register_class(_Beta)
    a.register_backend("fake", rec_a)

    b, rec_b = PyTreeRegistry(), _Recorder()
    b.register_backend("fake", rec_b)
    b.register_class(_Alpha)
    b.register_class(_Beta)

    assert set(rec_a.seen) == set(rec_b.seen) == {_Alpha, _Beta}
    assert a.registered_classes() == b.registered_classes()


def test_every_backend_gets_every_class():
    """The registry maintains the full cross-product, not just the last pair."""
    reg = PyTreeRegistry()
    first, second = _Recorder(), _Recorder()
    reg.register_backend("one", first)
    reg.register_class(_Alpha)
    reg.register_backend("two", second)   # back-fills _Alpha
    reg.register_class(_Beta)             # forward-fills into both

    assert set(first.seen) == {_Alpha, _Beta}
    assert set(second.seen) == {_Alpha, _Beta}
    assert set(reg.backends()) == {"one", "two"}


# --------------------------------------------------------------------------
# Idempotency
# --------------------------------------------------------------------------

def test_duplicate_class_registration_wires_once():
    reg, rec = PyTreeRegistry(), _Recorder()
    reg.register_backend("fake", rec)
    reg.register_class(_Alpha)
    reg.register_class(_Alpha)
    assert rec.seen == [_Alpha]
    assert reg.registered_classes() == (_Alpha,)


def test_duplicate_backend_registration_wires_once():
    """Re-installing a protocol must not re-register (backends raise on that)."""
    reg, rec = PyTreeRegistry(), _Recorder()
    reg.register_class(_Alpha)
    reg.register_backend("fake", rec)
    reg.register_backend("fake", rec)
    assert rec.seen == [_Alpha]


# --------------------------------------------------------------------------
# Failure containment: import must survive a broken registrar
# --------------------------------------------------------------------------

def test_failing_registrar_warns_instead_of_raising():
    """A backend that rejects registration must not abort ``import spacecore``."""
    def boom(cls: type) -> None:
        raise ValueError("duplicate registration")

    reg = PyTreeRegistry()
    reg.register_class(_Alpha)
    with pytest.warns(RuntimeWarning, match="duplicate registration"):
        reg.register_backend("fake", boom)


def test_failing_registrar_is_not_retried():
    """The pair is marked done on failure, so there is no repeat attempt or spam."""
    calls: list[type] = []

    def boom(cls: type) -> None:
        calls.append(cls)
        raise ValueError("nope")

    reg = PyTreeRegistry()
    with pytest.warns(RuntimeWarning):
        reg.register_backend("fake", boom)
        reg.register_class(_Alpha)
    reg.register_class(_Alpha)   # second attempt must be a no-op
    assert calls == [_Alpha]


# --------------------------------------------------------------------------
# The ABCMeta gate: which subclasses auto-register
# --------------------------------------------------------------------------

def test_concrete_subclass_is_auto_registered():
    class Concrete(PyTreeNode):
        def __init__(self, x):
            self.x = x

        def tree_flatten(self):
            return (self.x,), None

        @classmethod
        def tree_unflatten(cls, aux, children):
            return cls(children[0])

    assert Concrete in global_registry.registered_classes()


def test_subclass_without_the_protocol_is_not_registered():
    """A subclass that has not implemented ``tree_flatten`` must be skipped."""
    class StillAbstract(PyTreeNode):
        pass

    assert StillAbstract not in global_registry.registered_classes()


def test_subclass_with_protocol_but_abstract_elsewhere_is_registered():
    """Flattenable-but-abstract-for-other-reasons still registers.

    Pins the semantics of the gate: it asks "is this class flattenable yet?",
    not "is it fully concrete?". ``inspect.isabstract`` would answer the latter
    and skip this class, but a registrar only records the type and consults the
    two protocol methods, both of which are present here.
    """
    class Extra(PyTreeNode):
        @abstractmethod
        def apply(self):        # unrelated abstract method keeps the class abstract
            ...

        def tree_flatten(self):
            return (), None

        @classmethod
        def tree_unflatten(cls, aux, children):
            return cls()

    assert inspect.isabstract(Extra)                       # genuinely still abstract
    assert Extra in global_registry.registered_classes()   # yet registered


def test_pytree_node_itself_is_not_registered():
    assert PyTreeNode not in global_registry.registered_classes()


def test_abstract_subclass_cannot_be_instantiated():
    class StillAbstract(PyTreeNode):
        pass

    with pytest.raises(TypeError):
        StillAbstract()


# --------------------------------------------------------------------------
# Mixin composition
# --------------------------------------------------------------------------

def test_init_subclass_chain_is_cooperative():
    """PyTreeNode must not swallow ``__init_subclass__`` for sibling mixins."""
    observed: list[str] = []

    class OtherMixin:
        def __init_subclass__(cls, **kwargs):
            super().__init_subclass__(**kwargs)
            observed.append(cls.__name__)

    class Combined(PyTreeNode, OtherMixin):
        def tree_flatten(self):
            return (), None

        @classmethod
        def tree_unflatten(cls, aux, children):
            return cls()

    assert observed == ["Combined"]                       # sibling hook still ran
    assert Combined in global_registry.registered_classes()   # and ours did too


# --------------------------------------------------------------------------
# Completeness against the real backend
# --------------------------------------------------------------------------

@pytest.mark.skipif(not has_jax(), reason="jax is not installed")
def test_every_registered_class_reached_jax():
    """Each class the registry recorded really is a JAX pytree node.

    This is the guard the old decorator could not offer: it kept no list of what
    it had decorated, so "did anything get missed?" was unanswerable. The probe
    is JAX's own public API — re-registering an already-registered type raises.
    """
    import jax

    import spacecore  # noqa: F401 - ensure every container module has imported

    recorded = global_registry.registered_classes()
    assert len(recorded) > 40, f"suspiciously few containers registered: {len(recorded)}"

    unreached = []
    for cls in recorded:
        try:
            jax.tree_util.register_pytree_node_class(cls)
        except ValueError:
            continue          # already registered - the expected outcome
        unreached.append(f"{cls.__module__}.{cls.__qualname__}")
    assert not unreached, f"recorded but never reached JAX: {unreached}"


@pytest.mark.skipif(not has_jax(), reason="jax is not installed")
def test_jax_protocol_is_installed_on_import():
    import spacecore  # noqa: F401

    assert "jax" in global_registry.backends()


@pytest.mark.skipif(not has_jax(), reason="jax is not installed")
def test_registered_class_round_trips_through_jax_tree_util():
    """A PyTreeNode subclass decomposes and rebuilds via ``jax.tree_util``.

    Re-homed from the retired ``test_jax_pytree_class.py``: the capability it
    covered is unchanged, only the mechanism that grants it.
    """
    import jax.tree_util as jtu

    class JaxPair(PyTreeNode):
        def __init__(self, x, y):
            self.x, self.y = x, y

        def tree_flatten(self):
            return (self.x, self.y), None

        @classmethod
        def tree_unflatten(cls, aux, children):
            return cls(*children)

        def __eq__(self, other):
            return isinstance(other, JaxPair) and (self.x, self.y) == (other.x, other.y)

    inst = JaxPair(1.0, 2.0)
    leaves, treedef = jtu.tree_flatten(inst)
    assert tuple(leaves) == (1.0, 2.0)          # decomposed, not an opaque leaf
    assert jtu.tree_unflatten(treedef, leaves) == inst


@pytest.mark.skipif(not has_jax(), reason="jax is not installed")
def test_registered_class_supports_tree_map():
    """``tree_map`` reaches the leaves of a registered container."""
    import jax.tree_util as jtu

    class JaxVec3(PyTreeNode):
        def __init__(self, a, b, c):
            self.a, self.b, self.c = a, b, c

        def tree_flatten(self):
            return (self.a, self.b, self.c), None

        @classmethod
        def tree_unflatten(cls, aux, children):
            return cls(*children)

    doubled = jtu.tree_map(lambda v: v * 2.0, JaxVec3(1.0, 2.0, 3.0))
    assert (doubled.a, doubled.b, doubled.c) == (2.0, 4.0, 6.0)


@pytest.mark.skipif(not has_torch(), reason="torch is not installed")
def test_torch_protocol_is_installed_on_import():
    import spacecore  # noqa: F401

    assert "torch" in global_registry.backends()


@pytest.mark.skipif(not has_torch(), reason="torch is not installed")
def test_every_registered_class_reached_torch():
    """Torch's registry received every container — the second-backend guard.

    Mirrors the JAX completeness check; together they show the seam scales to a
    second backend with no per-class edits.
    """
    import torch.utils._pytree as torch_pytree

    import spacecore  # noqa: F401

    unreached = []
    for cls in global_registry.registered_classes():
        try:
            torch_pytree.register_pytree_node(
                cls, lambda o: ([], None), lambda children, aux: None
            )
        except ValueError:
            continue          # already registered - the expected outcome
        unreached.append(f"{cls.__module__}.{cls.__qualname__}")
    assert not unreached, f"recorded but never reached torch: {unreached}"


@pytest.mark.skipif(not has_torch(), reason="torch is not installed")
def test_torch_backed_container_round_trips_and_maps():
    """A real Torch-backed operator decomposes, rebuilds, and maps.

    Exercises the adapter's translation: Torch wants ``list`` children and calls
    ``unflatten(children, context)`` — the reverse of the ``(aux, children)``
    order :class:`PyTreeNode` defines.
    """
    import torch
    import torch.utils._pytree as torch_pytree

    import spacecore as sc

    ctx = sc.Context(sc.TorchOps(), dtype=torch.float64)
    space = sc.DenseCoordinateSpace((3,), ctx=ctx)
    op = sc.DenseLinOp(torch.eye(3, dtype=torch.float64), space, space)

    leaves, spec = torch_pytree.tree_flatten(op)
    assert leaves and torch.is_tensor(leaves[0])          # decomposed, not opaque
    assert torch_pytree.tree_unflatten(leaves, spec) == op

    doubled = torch_pytree.tree_map(lambda t: t * 2 if torch.is_tensor(t) else t, op)
    assert torch.allclose(doubled.to_matrix(), 2 * torch.eye(3, dtype=torch.float64))


@pytest.mark.skipif(not has_torch(), reason="torch is not installed")
def test_torch_registration_also_covers_cxx_pytree():
    """One registration serves both Torch pytree implementations.

    Torch mirrors registrations into ``_cxx_pytree`` (the optree-backed variant
    selected by ``PYTORCH_USE_CXX_PYTREE=1``), so the adapter registers once —
    registering with both explicitly would raise on the second call.
    """
    import torch
    import torch.utils._cxx_pytree as cxx_pytree

    import spacecore as sc

    ctx = sc.Context(sc.TorchOps(), dtype=torch.float64)
    space = sc.DenseCoordinateSpace((3,), ctx=ctx)
    op = sc.DenseLinOp(torch.eye(3, dtype=torch.float64), space, space)

    leaves, _ = cxx_pytree.tree_flatten(op)
    assert leaves and torch.is_tensor(leaves[0])


@pytest.mark.skipif(not has_jax(), reason="jax is not installed")
def test_backend_ops_are_deliberately_not_containers():
    """``*Ops`` classes are opaque leaves, not pytree nodes — by design.

    Replaces a test that asserted the opposite in its docstring yet passed
    vacuously: ``tree_map`` returns an *unregistered* object unchanged because
    it is a leaf, so the old assertion held whether or not registration existed.
    """
    import jax.tree_util as jtu
    import spacecore as sc

    ops = sc.NumpyOps()
    assert type(ops) not in global_registry.registered_classes()
    leaves, _ = jtu.tree_flatten(ops)
    assert leaves == [ops]        # exactly one leaf: itself


def test_round_trip_contract():
    class Pair(PyTreeNode):
        def __init__(self, x, y):
            self.x, self.y = x, y

        def tree_flatten(self):
            return (self.x, self.y), "meta"

        @classmethod
        def tree_unflatten(cls, aux, children):
            assert aux == "meta"
            return cls(*children)

        def __eq__(self, other):
            return isinstance(other, Pair) and (self.x, self.y) == (other.x, other.y)

    original = Pair(1.0, 2.0)
    children, aux = original.tree_flatten()
    assert Pair.tree_unflatten(aux, children) == original
