"""Krylov methods use space algebra even when elements have no coordinates API."""
from dataclasses import dataclass

import numpy as np
import pytest
import scipy.linalg

import spacecore as sc
from spacecore.backend import PyTreeNode
from tests._helpers import to_numpy
from tests.linalg._helpers import make_ctx, numpy_jax_params


@dataclass(frozen=True)
class Pair(PyTreeNode):
    head: object
    tail: object

    def __array__(self, *args, **kwargs):
        raise AssertionError("Native elements must not be converted to arrays")

    def tree_flatten(self):
        return (self.head, self.tail), None

    @classmethod
    def tree_unflatten(cls, aux, children):
        return cls(*children)


class PairSpace(sc.InnerProductSpace):
    """Weighted geometry on two differently shaped leaves, with no shape attribute."""

    @property
    def is_euclidean(self):
        return False

    def _check_member(self, x):
        assert isinstance(x, Pair)
        assert x.head.shape == () and x.tail.shape == (2,)

    def _convert(self, ctx):
        return PairSpace(ctx)

    def zeros(self):
        return Pair(self.ops.zeros((), dtype=self.dtype), self.ops.zeros((2,), dtype=self.dtype))

    def _add_core(self, x, y):
        return Pair(x.head + y.head, x.tail + y.tail)

    def _scale_core(self, a, x):
        return Pair(a * x.head, a * x.tail)

    def _inner_core(self, x, y):
        return 2 * self.ops.conj(x.head) * y.head + self.ops.vdot(x.tail, y.tail)

    def flatten(self, x):
        raise AssertionError("Krylov algorithms must not flatten elements")

    def unflatten(self, x):
        raise AssertionError("Krylov algorithms must not unflatten elements")


def as_numpy(x):
    return np.concatenate((np.asarray(to_numpy(x.head)).reshape(1), to_numpy(x.tail)))


@pytest.mark.parametrize("backend_name,dtype", numpy_jax_params())
@pytest.mark.parametrize("complex_values", [False, True])
def test_native_elements_match_dense_truth_and_jit(backend_name, dtype, complex_values):
    if complex_values:
        dtype = np.complex128 if dtype == np.float64 else np.complex64
    ctx = make_ctx(backend_name, dtype, check_level="standard")
    X = PairSpace(ctx)
    h = 1j if complex_values else 1.0
    symmetric = np.array([[4, h, 0], [np.conj(h), 3, .2], [0, .2, 4]], dtype=dtype)
    weights = np.array([2, 1, 1])
    matrix = symmetric / weights[:, None]
    entries = ctx.asarray(matrix)

    def apply(x):
        return Pair(
            entries[0, 0] * x.head + entries[0, 1:] @ x.tail,
            entries[1:, 0] * x.head + entries[1:, 1:] @ x.tail,
        )

    A = sc.MatrixFreeLinOp(apply, apply, X, X, ctx, euclidean_adjoint=False)
    v = Pair(ctx.asarray(1.0), ctx.asarray([-.5, .75]))
    original = as_numpy(v).copy()
    time = -.2j if complex_values else -.2

    def solve(x):
        return (
            sc.lanczos_smallest(A, x, max_iter=6, tol=1e-5, check_every=3),
            sc.expm_multiply(A, x, t=time, max_iter=6, tol=1e-5),
        )

    runs = [solve(v)]
    if backend_name == "jax":
        import jax
        runs.append(jax.jit(solve)(v))
    expected_eigenvalue = np.linalg.eigvalsh(symmetric / np.sqrt(weights[:, None] * weights[None, :]))[0]
    for eigen, exponential in runs:
        assert isinstance(eigen.eigenvector, Pair)
        assert isinstance(exponential.result, Pair)
        x = as_numpy(eigen.eigenvector)
        np.testing.assert_allclose(to_numpy(eigen.eigenvalue), expected_eigenvalue, atol=2e-5)
        np.testing.assert_allclose(matrix @ x, expected_eigenvalue * x, atol=2e-5)
        np.testing.assert_allclose(np.vdot(x, weights * x), 1, atol=2e-5)
        np.testing.assert_allclose(as_numpy(exponential.result), scipy.linalg.expm(time * matrix) @ original, atol=2e-5)
        assert bool(to_numpy(eigen.converged))
        assert int(to_numpy(eigen.krylov_dim)) == 3
    np.testing.assert_array_equal(as_numpy(v), original)

    # Zero handling must also work with an opaque element and under tracing.
    with pytest.raises(ValueError, match="nonzero"):
        sc.lanczos_smallest(A, X.zeros(), max_iter=4)
    zero = sc.expm_multiply(A, X.zeros(), max_iter=4)
    np.testing.assert_array_equal(as_numpy(zero.result), np.zeros(3))
    assert int(to_numpy(zero.krylov_dim)) == 0
    if backend_name == "jax":
        eigen, exponential = jax.jit(solve)(X.zeros())
        assert np.isnan(to_numpy(eigen.eigenvalue))
        assert not bool(to_numpy(eigen.converged))
        np.testing.assert_array_equal(as_numpy(exponential.result), np.zeros(3))


@pytest.mark.parametrize("backend_name,dtype", numpy_jax_params())
def test_dense_matrix_elements_keep_shape(backend_name, dtype):
    ctx = make_ctx(backend_name, dtype)
    X = sc.DenseCoordinateSpace((2, 2), ctx)
    diagonal = ctx.asarray([[1., 2.], [3., 4.]])
    A = sc.MatrixFreeLinOp(lambda x: diagonal * x, lambda x: diagonal * x, X, X, ctx)
    v = ctx.asarray([[1., -.5], [.75, 1.5]])

    def solve(x):
        return sc.lanczos_smallest(A, x, max_iter=4, tol=1e-5), sc.expm_multiply(A, x, t=-.2, max_iter=4)

    if backend_name == "jax":
        import jax
        solve = jax.jit(solve)
    eigen, exponential = solve(v)
    assert eigen.eigenvector.shape == exponential.result.shape == (2, 2)
    np.testing.assert_allclose(to_numpy(eigen.eigenvalue), 1., atol=2e-5)
    np.testing.assert_allclose(to_numpy(exponential.result), np.exp(-.2 * to_numpy(diagonal)) * to_numpy(v), atol=2e-5)


def test_zero_exponential_does_not_apply_operator():
    ctx = make_ctx()
    X = PairSpace(ctx)

    def forbidden(x):
        raise AssertionError("Zero exponential action should not evaluate the operator")

    A = sc.MatrixFreeLinOp(forbidden, forbidden, X, X, ctx, euclidean_adjoint=False)
    result = sc.expm_multiply(A, X.zeros())
    np.testing.assert_array_equal(as_numpy(result.result), np.zeros(3))
    assert result.krylov_dim == 0
    assert result.residual_estimate == 0
    assert result.converged
