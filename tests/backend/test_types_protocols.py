"""Tests for :mod:`spacecore.types` array protocols.

Checklist section 4:

* ``sc.ArrayLike``, ``sc.DenseArray``, and ``sc.SparseArray`` are all usable
  with :func:`isinstance` at runtime without raising ``TypeError``.
* NumPy dense arrays satisfy all three protocols structurally.
* SciPy sparse matrices satisfy all three protocols structurally.
* Plain Python containers (lists) and bare objects -- which lack ``.shape``
  and ``.dtype`` -- satisfy none of them.
* ``DenseArray`` / ``SparseArray`` instances are also ``ArrayLike`` (the
  protocols nest structurally).
* Known limitation: the structural ``SparseArray`` protocol cannot reject a
  dense array, so ``isinstance(dense, SparseArray)`` is ``True`` (documented
  below; a true dense/sparse split needs a source change, out of scope here).
"""
from __future__ import annotations

import typing

import numpy as np
import pytest
import scipy.sparse as sps

import spacecore as sc
import spacecore.types as sct
from tests._helpers import has_jax, has_torch


class _ShapeAndDtype:
    """Minimal duck-typed object exposing only ``.shape`` and ``.dtype``."""

    shape = (3,)
    dtype = np.dtype(np.float64)


class _ShapeOnly:
    """Object exposing ``.shape`` but no ``.dtype``."""

    shape = (3,)


# Required members of ``DenseArray`` beyond the ``ArrayLike`` base
# (``shape``/``dtype``). Read from spacecore/types/_array.py: ``DenseArray``
# adds ``ndim`` plus the operator surface below.
_DENSE_MEMBERS = (
    "ndim",
    "T",
    "conj",
    "reshape",
    "__len__",
    "__getitem__",
    "__setitem__",
    "__add__",
    "__radd__",
    "__sub__",
    "__rsub__",
    "__mul__",
    "__rmul__",
    "__truediv__",
    "__rtruediv__",
    "__neg__",
    "__matmul__",
    "__rmatmul__",
)

# Required members of ``SparseArray`` beyond ``ArrayLike``. Read from
# spacecore/types/_array.py: ``SparseArray`` adds ``T``/``conj``/``reshape``/
# ``__matmul__``.
_SPARSE_MEMBERS = ("T", "conj", "reshape", "__matmul__")


def _full_member_namespace(members):
    """Build a class namespace satisfying ``ArrayLike`` plus ``members``."""
    ns = {"shape": (3,), "dtype": np.dtype(np.float64)}
    for name in members:
        if name == "ndim":
            ns[name] = 1
        elif name in ("shape", "dtype", "T"):
            ns[name] = ns.get(name, (3,))
        else:
            # Both regular and dunder methods are provided as callables;
            # runtime_checkable protocols only check member presence.
            ns[name] = lambda self, *a, **k: self
    ns["T"] = (3,)
    return ns


def _make_with_member_dropped(members, dropped):
    """Return an instance that has every required member except ``dropped``.

    Dunder methods are resolved on the *type* for protocol ``isinstance``
    checks, so the namespace is materialized into a class with ``type()``.
    """
    ns = _full_member_namespace(members)
    del ns[dropped]
    cls = type(f"_Missing_{dropped.strip('_')}", (), ns)
    return cls()


# ===========================================================================
# ArrayLike
# ===========================================================================
class TestArrayLike:
    def test_numpy_dense_is_arraylike(self):
        assert isinstance(np.zeros((3, 3)), sc.ArrayLike) is True

    def test_scipy_csr_is_arraylike(self):
        assert isinstance(sps.csr_matrix(np.eye(3)), sc.ArrayLike) is True

    def test_duck_typed_shape_and_dtype_is_arraylike(self):
        assert isinstance(_ShapeAndDtype(), sc.ArrayLike) is True

    def test_plain_list_is_not_arraylike(self):
        assert isinstance([1.0, 2.0], sc.ArrayLike) is False

    def test_bare_object_is_not_arraylike(self):
        assert isinstance(object(), sc.ArrayLike) is False

    def test_shape_without_dtype_is_not_arraylike(self):
        assert isinstance(_ShapeOnly(), sc.ArrayLike) is False


# ===========================================================================
# DenseArray
# ===========================================================================
class TestDenseArray:
    def test_numpy_dense_is_densearray(self):
        assert isinstance(np.zeros((3, 3)), sc.DenseArray) is True

    def test_plain_list_is_not_densearray(self):
        assert isinstance([1.0, 2.0], sc.DenseArray) is False

    def test_dense_instance_is_also_arraylike(self):
        x = np.zeros((3, 3))
        assert isinstance(x, sc.DenseArray) is True
        assert isinstance(x, sc.ArrayLike) is True

    @pytest.mark.skipif(not has_jax(), reason="jax is not installed")
    def test_jax_array_is_densearray(self):
        import jax.numpy as jnp

        assert isinstance(jnp.zeros((3, 3)), sc.DenseArray) is True

    @pytest.mark.skipif(not has_torch(), reason="torch is not installed")
    def test_torch_tensor_is_densearray(self):
        import torch

        assert isinstance(torch.zeros((3, 3)), sc.DenseArray) is True

    def test_full_member_object_is_densearray(self):
        # Sanity check that the synthetic full-member object is accepted, so
        # the negative cases below isolate the dropped member as the cause.
        ns = _full_member_namespace(_DENSE_MEMBERS)
        cls = type("_FullDense", (), ns)
        assert isinstance(cls(), sc.DenseArray) is True

    @pytest.mark.parametrize("dropped", _DENSE_MEMBERS)
    def test_missing_one_required_member_is_not_densearray(self, dropped):
        # Each required member must actually discriminate: an object that has
        # every other DenseArray member but is missing exactly one fails the
        # structural isinstance check.
        obj = _make_with_member_dropped(_DENSE_MEMBERS, dropped)
        assert isinstance(obj, sc.DenseArray) is False


# ===========================================================================
# SparseArray
# ===========================================================================
class TestSparseArray:
    @pytest.mark.parametrize(
        "fmt",
        [sps.csr_matrix, sps.csc_matrix, sps.coo_matrix],
    )
    def test_scipy_sparse_is_sparsearray(self, fmt):
        assert isinstance(fmt(np.eye(3)), sc.SparseArray) is True

    def test_sparse_instance_is_also_arraylike(self):
        x = sps.csr_matrix(np.eye(3))
        assert isinstance(x, sc.SparseArray) is True
        assert isinstance(x, sc.ArrayLike) is True

    def test_full_member_object_is_sparsearray(self):
        ns = _full_member_namespace(_SPARSE_MEMBERS)
        cls = type("_FullSparse", (), ns)
        assert isinstance(cls(), sc.SparseArray) is True

    @pytest.mark.parametrize("dropped", _SPARSE_MEMBERS)
    def test_missing_one_required_member_is_not_sparsearray(self, dropped):
        # Each SparseArray member (T/conj/reshape/__matmul__) discriminates:
        # dropping exactly one fails the structural isinstance check.
        obj = _make_with_member_dropped(_SPARSE_MEMBERS, dropped)
        assert isinstance(obj, sc.SparseArray) is False

    def test_dense_array_is_not_rejected_by_sparse_protocol(self):
        # Known limitation: ``SparseArray`` is a *structural* Protocol -- it
        # only checks for the presence of ``shape``/``dtype``/``T``/``conj``/
        # ``reshape``/``__matmul__``, all of which a dense NumPy array also
        # provides. There is no structural feature that distinguishes a sparse
        # matrix from a dense array, so a dense array passes the check. The
        # checklist's "dense rejected" goal is not achievable without a source
        # change (e.g. an explicit ``__subclasshook__``), which is out of scope
        # here. This assertion pins the current behavior as a regression guard.
        assert isinstance(np.zeros((3, 3)), sc.SparseArray) is True


# ===========================================================================
# Runtime checkability (regression guard)
# ===========================================================================
class TestRuntimeCheckable:
    @pytest.mark.parametrize(
        "protocol",
        [sc.ArrayLike, sc.DenseArray, sc.SparseArray],
    )
    def test_isinstance_does_not_raise(self, protocol):
        # A non-runtime-checkable Protocol raises TypeError on isinstance.
        # All three array protocols must support isinstance at runtime.
        try:
            isinstance(object(), protocol)
        except TypeError as exc:  # pragma: no cover - failure path
            pytest.fail(f"isinstance against {protocol!r} raised TypeError: {exc}")


# ===========================================================================
# Public type aliases / TypeVars
# ===========================================================================
class TestPublicTypeAliases:
    @pytest.mark.parametrize(
        "name",
        ["DType", "Index", "T", "Carry", "X", "Y", "R"],
    )
    def test_name_is_exported(self, name):
        assert name in sct.__all__
        assert hasattr(sct, name)

    def test_dtype_is_any(self):
        # Source: spacecore/types/_dtype.py -- ``DType: TypeAlias = Any``.
        assert sct.DType is typing.Any

    def test_index_is_expected_union(self):
        # Source: spacecore/types/_misc.py -- ``Index = Union[int, slice,
        # Any, Tuple[Any, ...]]``. ``Any`` absorbs the other members, so the
        # simplified union collapses, but it must equal the source definition.
        expected = typing.Union[int, slice, typing.Any, typing.Tuple[typing.Any, ...]]
        assert sct.Index == expected

    @pytest.mark.parametrize("name", ["T", "Carry", "X", "Y", "R"])
    def test_typevars_are_typevar_instances(self, name):
        var = getattr(sct, name)
        assert isinstance(var, typing.TypeVar)
        assert var.__name__ == name
