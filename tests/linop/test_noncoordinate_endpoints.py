"""Linear operators need coordinates only for coordinate operations."""
import numpy as np
import pytest

from spacecore import Context, NumpyOps, DenseCoordinateSpace
from spacecore._batching import _batched_inner
from spacecore._errors import CapabilityError
from spacecore.linop import IdentityLinOp, MatrixFreeLinOp, ZeroLinOp
from spacecore.linop._algebra import _same_space_for_algebra
from spacecore.linop._metric import _metric_is_hermitian_by_basis, metric_rapply
from spacecore.space.base import CoordinateSpace, InnerProduct, InnerProductSpace, VectorSpace
from spacecore.space._capabilities import (
    _CAP_BATCH, _NON_DISPATCH_CAPABILITIES, _space_capabilities, probe_element,
    registry_key, require,
)


class ArraySpace(VectorSpace):
    """Array-backed vectors with no public coordinate or geometry surface."""

    def zeros(self):
        return self.ctx.asarray([0., 0.])

    def add(self, x, y):
        return x + y

    def scale(self, a, x):
        return a * x

    def _convert(self, new_ctx):
        return type(self)(new_ctx, check_level=self.check_level)


class DiagonalGeometry(InnerProduct):
    def __init__(self, weights):
        self.weights = np.asarray(weights)

    def inner(self, ops, x, y):
        return ops.vdot(x, self.weights * y)

    def riesz(self, ops, x):
        return self.weights * x

    def riesz_inverse(self, ops, x):
        return x / self.weights


class MetricArraySpace(ArraySpace, InnerProductSpace):
    def ones(self):
        """Deterministic non-zero probe element; no coordinate surface needed."""
        return self.ctx.asarray([1., 1.])

    def __init__(self, ctx, weights=(2., 5.), check_level=None):
        super().__init__(ctx, check_level=check_level)
        self.geometry = DiagonalGeometry(weights)

    def _eq_algebra(self, other):
        return super()._eq_algebra(other) and np.array_equal(
            self.geometry.weights, other.geometry.weights
        )

    def _convert(self, new_ctx):
        return type(self)(new_ctx, self.geometry.weights, self.check_level)


@pytest.fixture(params=['none', 'standard', 'strict'])
def example(request):
    ctx = Context(NumpyOps(), dtype=np.float64)
    dom = MetricArraySpace(ctx, (2., 5.), request.param)
    cod = MetricArraySpace(ctx, (3., 7.), request.param)
    matrix = np.array([[1., 2.], [-3., 4.]])
    op = MatrixFreeLinOp(
        lambda x: matrix @ x, lambda y: matrix.T @ y, dom, cod,
        euclidean_adjoint=True, check_level=request.param,
    )
    return op, matrix


def test_noncoordinate_algebra_and_metric_adjoint(example):
    op, matrix = example
    x, y = np.array([1., -2.]), np.array([3., 4.])
    expected = (matrix.T @ (op.codomain.geometry.weights * y)) / op.domain.geometry.weights
    assert isinstance(op.domain, VectorSpace)
    assert isinstance(op.domain, InnerProductSpace)
    assert not isinstance(op.domain, CoordinateSpace)
    for name in ('shape', 'size', 'flatten', 'unflatten', 'stacked'):
        assert not hasattr(op.domain, name)
    np.testing.assert_allclose(op.apply(x), matrix @ x)
    np.testing.assert_allclose(op.rapply(y), expected)
    np.testing.assert_allclose(op.H.apply(y), expected)
    np.testing.assert_allclose(op.H.rapply(x), matrix @ x)
    np.testing.assert_allclose((op + op).apply(x), 2 * (matrix @ x))
    np.testing.assert_allclose((op + op).rapply(y), 2 * expected)
    np.testing.assert_allclose((3 * op).apply(x), 3 * (matrix @ x))
    np.testing.assert_allclose((op * 3).rapply(y), 3 * expected)
    np.testing.assert_allclose((op.H @ op).apply(x), op.rapply(op.apply(x)))
    np.testing.assert_allclose((op.H @ op).rapply(x), op.rapply(op.apply(x)))
    np.testing.assert_allclose(op.codomain.inner(op.apply(x), y), op.domain.inner(x, op.rapply(y)))
    assert not np.allclose(expected, matrix.T @ y)
    assert _metric_is_hermitian_by_basis(op.H @ op) is None


@pytest.mark.parametrize('method', ['to_dense', 'to_matrix', 'vapply', 'rvapply'])
@pytest.mark.parametrize('variant', ['original', 'scaled', 'sum', 'composed', 'adjoint', 'zero', 'identity'])
def test_coordinate_operations_reject_noncoordinate_endpoints(example, method, variant):
    op, _ = example
    op = {
        'original': op, 'scaled': 2 * op, 'sum': op + op,
        'composed': op.H @ op, 'adjoint': op.H,
        'zero': ZeroLinOp(op.domain, op.codomain, check_level=op.check_level),
        'identity': IdentityLinOp(op.domain, check_level=op.check_level),
    }[variant]
    args = (np.ones((3, 2)),) if method in ('vapply', 'rvapply') else ()
    with pytest.raises(CapabilityError, match='CoordinateSpace'):
        getattr(op, method)(*args)


@pytest.mark.parametrize('method', ['vapply', 'rvapply'])
def test_explicit_batch_callbacks_require_coordinates(example, method):
    op, _ = example
    batch_op = MatrixFreeLinOp(
        op.apply, op.rapply, op.domain, op.codomain,
        vapply=lambda xs: xs, rvapply=lambda ys: ys, euclidean_adjoint=False,
        check_level=op.check_level,
    )
    with pytest.raises(CapabilityError, match='CoordinateSpace'):
        getattr(batch_op, method)(np.ones((3, 2)))


@pytest.mark.parametrize('missing_endpoint', ['domain', 'codomain'])
def test_plain_vector_space_forward_works_but_adjoint_requires_inner(missing_endpoint):
    ctx = Context(NumpyOps(), dtype=np.float64)
    plain, metric = ArraySpace(ctx), MetricArraySpace(ctx)
    dom, cod = (plain, metric) if missing_endpoint == 'domain' else (metric, plain)
    op = MatrixFreeLinOp(lambda x: 2 * x, lambda y: 2 * y, dom, cod)
    x = np.array([1., 2.])
    np.testing.assert_array_equal(op.apply(x), 2 * x)
    with pytest.raises(CapabilityError, match='InnerProductSpace'):
        op.rapply(x)
    with pytest.raises(CapabilityError, match='InnerProductSpace'):
        metric_rapply(dom, cod, lambda y: y, x)


def test_require_names_space_capability_and_operation():
    space = ArraySpace(Context(NumpyOps(), dtype=np.float64))
    with pytest.raises(CapabilityError, match='probe.*CoordinateSpace.*ArraySpace'):
        require(space, _CAP_BATCH, 'probe')
    require(space, VectorSpace, 'add')
    with pytest.raises(CapabilityError, match='CoordinateSpace'):
        _batched_inner(space, np.ones((2, 2)), np.ones((2, 2)))
    assert _space_capabilities(space) == frozenset()
    assert CoordinateSpace in _space_capabilities(DenseCoordinateSpace((2,), space.ctx))


def test_algebra_compatibility_does_not_inspect_shape():
    ctx = Context(NumpyOps(), dtype=np.float64)
    assert not _same_space_for_algebra(MetricArraySpace(ctx, (2., 5.)), MetricArraySpace(ctx, (3., 7.)))


def test_strict_probe_runs_on_noncoordinate_space_and_catches_a_wrong_adjoint():
    """A space with `ones` is probed even without coordinates (finding 1)."""
    ctx = Context(NumpyOps(), dtype=np.float64)
    dom, cod = MetricArraySpace(ctx, (2., 5.)), MetricArraySpace(ctx, (3., 7.))
    matrix = np.array([[1., 2.], [-3., 4.]])
    assert not isinstance(dom, CoordinateSpace)

    # The coordinate transpose is *not* the metric adjoint on this geometry, and
    # euclidean_adjoint=False asserts that it is. Strict checks must catch that.
    with pytest.raises(ValueError, match='adjoint consistency check failed'):
        MatrixFreeLinOp(
            lambda x: matrix @ x, lambda y: matrix.T @ y, dom, cod,
            euclidean_adjoint=False, check_level='strict',
        )

    # The wrapped coordinate adjoint is correct, so the same probe passes.
    MatrixFreeLinOp(
        lambda x: matrix @ x, lambda y: matrix.T @ y, dom, cod,
        euclidean_adjoint=True, check_level='strict',
    )


def test_strict_probe_skipped_when_no_element_can_be_built():
    """A bare VectorSpace offers only zeros, so the probe is skipped, not crashed."""
    ctx = Context(NumpyOps(), dtype=np.float64)

    class NoOnes(MetricArraySpace):
        ones = None   # withdraw the capability

    dom = NoOnes(ctx, (2., 5.))
    assert probe_element(dom, dom.ctx.ops, dom.dtype) is None
    MatrixFreeLinOp(
        lambda x: 2. * x, lambda y: 2. * y, dom, dom,
        euclidean_adjoint=False, check_level='strict',
    )


def test_registry_key_strips_non_dispatch_capabilities():
    """Registry lookup survives a capability that does not select a class (finding 2)."""
    ctx = Context(NumpyOps(), dtype=np.float64)
    leaf = DenseCoordinateSpace((2,), ctx)
    caps = _space_capabilities(leaf)

    assert CoordinateSpace in caps
    assert CoordinateSpace not in registry_key(caps)
    assert registry_key(caps) == caps - _NON_DISPATCH_CAPABILITIES
    # Dispatch still reaches the inner-product specialization rather than the base.
    assert isinstance(leaf.stacked(3), InnerProductSpace)
