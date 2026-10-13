"""A space with no coordinates, only its own geometry, is enough for operators,
solvers and elements; nothing densifies behind the user's back."""
import warnings

import numpy as np
import pytest

import spacecore as sc
from tests.linalg.test_native_krylov import Pair, PairSpace, as_numpy


class OpaqueElement(sc.Element):
    """A representation with no plain member form."""

    _children = ("token",)

    def _check(self):
        pass


def _setup():
    ctx = sc.Context(sc.NumpyOps(), dtype=np.float64)
    X = PairSpace(ctx)
    diag = np.array([3.0, 2.0, 5.0])  # self-adjoint for inner = 2 x0 y0 + x1.y1

    def apply(x):
        return Pair(diag[0] * x.head, diag[1:] * x.tail)

    with warnings.catch_warnings():
        warnings.simplefilter("error")  # a native space must not trigger the coordinate-adjoint warning
        A = sc.MatrixFreeLinOp(apply, apply, X, X, ctx)
    b = Pair(np.asarray(1.0), np.array([2.0, -1.0]))
    return ctx, X, A, b, diag


def test_non_coordinate_space_carries_its_own_geometry():
    ctx, X, A, b, diag = _setup()
    assert X.is_euclidean is False
    np.testing.assert_allclose(as_numpy(X.riesz_inverse(X.riesz(b))), as_numpy(b))
    x = sc.cg(A, b, maxiter=10, tol=1e-12).x
    np.testing.assert_allclose(as_numpy(x), as_numpy(b) / diag)


def test_element_does_not_densify_at_level_none():
    calls = []

    class Counting(sc.DenseElement):
        def densify(self):
            calls.append(1)
            return self.array

    ctx = sc.Context(sc.NumpyOps(), dtype=np.float64)
    Counting(sc.DenseCoordinateSpace((2,), ctx, check_level="none"), np.ones(2))
    assert calls == []
    Counting(sc.DenseCoordinateSpace((2,), ctx, check_level="standard"), np.ones(2))
    assert calls == [1]


def test_element_without_member_form_raises_capability_error():
    ctx, X, A, b, diag = _setup()
    e = OpaqueElement(X, object())
    with pytest.raises(sc.CapabilityError):
        e + e
    with pytest.raises(sc.CapabilityError):
        2.0 * e
    with pytest.raises(sc.CapabilityError):
        A @ e
    # A native member wrapped as-is uses the space's own arithmetic.
    s = sc.DenseElement(X, b) + sc.DenseElement(X, b)
    np.testing.assert_allclose(as_numpy(s.array), 2 * as_numpy(b))
