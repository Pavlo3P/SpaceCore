"""One proximal-gradient/PDHG implementation across metrics and native elements.

Run ``python -m examples.proximal_contract``. Optimization loops belong to this
consumer example; the package supplies spaces, maps, and proximal operations.
"""
from dataclasses import dataclass

import numpy as np

import spacecore as sc
from spacecore.backend import PyTreeNode
from spacecore.mapping import register


@dataclass(frozen=True)
class SplitValue(PyTreeNode):
    """A scalar and a vector, with no array conversion or arithmetic overloads."""

    head: object
    tail: object

    def __array__(self, *args, **kwargs):
        raise TypeError("SplitValue must stay native")

    def tree_flatten(self):
        return (self.head, self.tail), None

    @classmethod
    def tree_unflatten(cls, aux, children):
        return cls(*children)


class SplitSpace(sc.InnerProductSpace):
    """Exact native coefficient algebra; deliberately has no shape/flatten API."""

    def __init__(self, weights, ctx, check_level="none"):
        super().__init__(ctx, check_level=check_level)
        self.weights = ctx.asarray(weights)

    @property
    def is_euclidean(self):
        return False  # Native metric actions are supplied explicitly.

    def _convert(self, ctx):
        return type(self)(self.weights, ctx, self.check_level)

    def _eq_algebra(self, other):
        return self is other or self.ops.allclose(self.weights, other.weights)

    def _check_member(self, x):
        if not isinstance(x, SplitValue) or x.head.shape != () or x.tail.shape != self.weights[1:].shape:
            raise sc.SpaceValidationError("Expected matching SplitValue leaves")

    def from_array(self, x):
        x = self.ctx.asarray(x)
        return SplitValue(x[0], x[1:])

    def zeros(self):
        return self.from_array(self.ops.zeros_like(self.weights))

    def add(self, x, y):
        return SplitValue(x.head + y.head, x.tail + y.tail)

    def scale(self, a, x):
        return SplitValue(a * x.head, a * x.tail)

    def multiply(self, diagonal, x):
        return SplitValue(diagonal[0] * x.head, diagonal[1:] * x.tail)

    def inner(self, x, y):
        return (self.weights[0] * self.ops.conj(x.head) * y.head
                + self.ops.vdot(x.tail, self.weights[1:] * y.tail))

    def riesz(self, x):
        return self.multiply(self.weights, x)

    def riesz_inverse(self, x):
        return self.multiply(1 / self.weights, x)

    def l1(self, x):
        return self.ops.abs(x.head) + self.ops.sum(self.ops.abs(x.tail))

    def shrink(self, x, step):
        def shrink_leaf(v, w):
            r = self.ops.abs(v)
            return self.ops.sign(v) * self.ops.maximum(r - step / w, self.ops.zeros_like(r))
        return SplitValue(shrink_leaf(x.head, self.weights[0]), shrink_leaf(x.tail, self.weights[1:]))


class NativeL1(sc.Functional):
    """Coordinate L1 in the native coefficient basis, with a registered prox."""

    def value(self, x):
        return self.domain.l1(x)

    def tree_flatten(self):
        return (), self.domain

    @classmethod
    def tree_unflatten(cls, aux, children):
        return cls(aux)

    def _convert(self, ctx):
        return type(self)(self.domain.convert(ctx), check_level=self.check_level)


register("prox", NativeL1, object, lambda f, geometry, x, step: f.domain.shrink(x, step))


class SquaredDistance(sc.Functional):
    """Half the squared distance to b in the domain metric."""

    def __init__(self, domain, b):
        super().__init__(domain)
        self.b = b

    def value(self, x):
        d = self.domain.axpy(-1, self.b, x)
        return self.ops.real(self.domain.inner(d, d)) / 2

    def grad(self, x):
        return self.domain.axpy(-1, self.b, x)

    def tree_flatten(self):
        return (self.b,), self.domain

    @classmethod
    def tree_unflatten(cls, aux, children):
        return cls(aux, children[0])

    def _convert(self, ctx):
        X = self.domain.convert(ctx)
        b = (SplitValue(ctx.asarray(self.b.head), ctx.asarray(self.b.tail))
             if isinstance(self.b, SplitValue) else ctx.asarray(self.b))
        return type(self)(X, b)


def _distance_prox(f, geometry, x, step):
    X = f.domain
    return X.scale(1 / (1 + step), X.axpy(step, f.b, x))


register("prox", SquaredDistance, object, _distance_prox)


def proximal_gradient(K, f, g, *, iterations=200, step=.2):
    """Run fixed-count proximal gradient using only the public mathematical API."""
    X, smooth = K.domain, f @ K

    def update(i, x):
        return g.prox(X.axpy(-step, smooth.grad(x), x), step)

    return X.ops.fori_loop(0, iterations, update, X.zeros())


def pdhg(K, f, g, *, iterations=200, step=.4):
    """Run PDHG with dual Riesz representers and Hilbert conjugate prox."""
    X, Y = K.domain, K.codomain

    def update(i, carry):
        x, p, extrapolated = carry
        p = f.conjugate_prox(Y.axpy(step, K(extrapolated), p), step)
        next_x = g.prox(X.axpy(-step, K.rapply(p), x), step)
        return next_x, p, X.axpy(-1, x, X.scale(2, next_x))

    return X.ops.fori_loop(0, iterations, update, (X.zeros(), Y.zeros(), X.zeros()))[0]


def build_problem(*, native=False, weighted=False, backend="numpy", count=16):
    """Build the same objective with either storage and either primal metric.

    min .5 ||diag(k)x-b||_2^2 + .15 ||x||_1; 1 <= k_i <= 2,
    1 <= w_i <= 3. Hence ||K||_(X->Y) <= 2 for both metrics.
    """
    ctx = sc.Context(sc.JaxOps() if backend == "jax" else sc.NumpyOps(), np.float32)
    weights_np = np.linspace(1, 3, count) if weighted else np.ones(count)
    k_np, b_np = np.linspace(1, 2, count), np.linspace(-2, 2, count)
    weights, k = ctx.asarray(weights_np), ctx.asarray(k_np)
    if native:
        X, Y = SplitSpace(weights, ctx), SplitSpace(np.ones(count), ctx)
        K = sc.MatrixFreeLinOp(lambda x: X.multiply(k, x),
                               lambda y: X.multiply(k / weights, y), X, Y,
                               euclidean_adjoint=False, check_level="none")
        b, g = Y.from_array(b_np), .15 * NativeL1(X)
    else:
        geometry = sc.WeightedInnerProduct(weights) if weighted else None
        X = sc.DenseCoordinateSpace((count,), ctx, geometry=geometry, check_level="none")
        Y = sc.DenseCoordinateSpace((count,), ctx, check_level="none")
        K = sc.MatrixFreeLinOp(lambda x: k * x, lambda y: k * y / weights, X, Y,
                               euclidean_adjoint=False, check_level="none")
        b, g = ctx.asarray(b_np), .15 * sc.L1NormFunctional(X)
    f = SquaredDistance(Y, b)
    # Independent separable KKT solution; the metric changes the algorithm,
    # not this objective or its optimum.
    optimum = np.sign(b_np) * np.maximum(np.abs(k_np * b_np) - .15, 0) / k_np**2
    return K, f, g, optimum


def as_numpy(x):
    """Extract coordinates for reporting only; algorithms never call this."""
    if isinstance(x, SplitValue):
        return np.concatenate((np.asarray(x.head).reshape(1), np.asarray(x.tail)))
    return np.asarray(x)


if __name__ == "__main__":
    for native in (False, True):
        for weighted in (False, True):
            K, f, g, optimum = build_problem(native=native, weighted=weighted)
            for method in (proximal_gradient, pdhg):
                error = np.linalg.norm(as_numpy(method(K, f, g)) - optimum)
                assert error < 2e-5
                print(f"{method.__name__:18} native={native} weighted={weighted}: error={error:.2e}")
