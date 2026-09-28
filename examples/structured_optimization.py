"""One mathematical problem, three algorithms, two native representations.

Run: python -m examples.structured_optimization
See structured_optimization.md for the mathematics and extension points.
"""
from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Any, Callable

import numpy as np

import spacecore as sc


@dataclass(frozen=True)
class Coefficients:
    """A native sparse value; vector operations belong to its space."""

    entries: dict[int, float]

    def __array__(self, *args, **kwargs):
        raise TypeError("Coefficients deliberately cannot be converted to a NumPy array")


class DenseCoefficients(sc.DenseCoordinateSpace):
    """Dense implementation of the coefficient operations used by this problem."""

    def from_components(self, values):
        return np.asarray(tuple(values), dtype=np.float64)

    def component(self, x, i):
        return x[i]

    def l1(self, x):
        return np.sum(np.abs(x))

    def shrink(self, x, threshold):
        return np.sign(x) * np.maximum(np.abs(x) - threshold, 0.0)


class SparseCoefficients(sc.InnerProductSpace):
    """Euclidean coefficient geometry without a coordinate-array interface.

    The dimension is adapter metadata. Algorithms do not inspect it, and this
    space supplies neither shape nor flatten/unflatten.
    """

    geometry = sc.EuclideanInnerProduct()

    def __init__(self, count, ctx):
        super().__init__(ctx)
        self.count = count

    def _eq_algebra(self, other):
        return super()._eq_algebra(other) and self.count == other.count

    def _check_member(self, x):
        if not isinstance(x, Coefficients):
            raise sc.SpaceValidationError("Expected a Coefficients value")
        if any(
            not isinstance(i, int) or not 0 <= i < self.count
            or not np.isreal(v) or not np.isfinite(v)
            for i, v in x.entries.items()
        ):
            raise sc.SpaceValidationError("Invalid coefficient index or value")

    def _convert(self, ctx):
        if ctx != self.ctx:
            raise NotImplementedError("This example implements NumPy float64 execution only")
        return type(self)(self.count, ctx)

    def from_components(self, values):
        values = tuple(values)
        if len(values) != self.count:
            raise ValueError("Wrong coefficient count")
        result = Coefficients({i: float(v) for i, v in enumerate(values) if v != 0})
        self.check_member(result)
        return result

    def component(self, x, i):
        return x.entries.get(i, 0.0)

    def zeros(self):
        return Coefficients({})

    def add(self, x, y):
        values = x.entries.copy()
        for i, value in y.entries.items():
            values[i] = values.get(i, 0.0) + value
        return Coefficients({i: value for i, value in values.items() if value != 0})

    def scale(self, a, x):
        self.check_scalar(a)
        return Coefficients({i: a * value for i, value in x.entries.items() if a * value != 0})

    def inner(self, x, y):
        return sum(value * y.entries.get(i, 0.0) for i, value in x.entries.items())

    def l1(self, x):
        return sum(abs(value) for value in x.entries.values())

    def shrink(self, x, threshold):
        return Coefficients({
            i: float(np.sign(value) * (abs(value) - threshold))
            for i, value in x.entries.items() if abs(value) > threshold
        })


class SquaredDistance(sc.Functional):
    """f(y) = ||y-b||²/2; gradient and conjugate prox use only space algebra."""

    def __init__(self, space, b):
        super().__init__(space)
        space.check_member(b)
        self.b = b

    def value(self, y):
        residual = self.grad(y)
        return self.domain.inner(residual, residual) / 2

    def grad(self, y):
        return self.domain.axpy(-1, self.b, y)

    def conjugate_prox(self, y, step):
        # f*(p) = ||p||²/2 + <p,b> in the declared Euclidean geometry.
        return self.domain.scale(1 / (1 + step), self.domain.axpy(-step, self.b, y))

    def tree_flatten(self):
        return (self.b,), self.domain

    @classmethod
    def tree_unflatten(cls, space, children):
        return cls(space, children[0])

    def _convert(self, ctx):
        if ctx != self.ctx:
            raise NotImplementedError("The example has no cross-backend value conversion")
        return type(self)(self.domain, self.b)


class L1Penalty(sc.Functional):
    """g(x) = weight * ||x||₁; native kernels are supplied by the adapter."""

    def __init__(self, space, weight):
        super().__init__(space)
        if not np.isfinite(weight) or weight <= 0:
            raise ValueError("weight must be finite and positive")
        self.weight = weight

    def value(self, x):
        return self.weight * self.domain.l1(x)

    def prox(self, x, step):
        return self.domain.shrink(x, step * self.weight)

    def tree_flatten(self):
        return (), (self.domain, self.weight)

    @classmethod
    def tree_unflatten(cls, aux, children):
        return cls(*aux)

    def _convert(self, ctx):
        if ctx != self.ctx:
            raise NotImplementedError("The example has no cross-backend value conversion")
        return type(self)(self.domain, self.weight)


@dataclass
class Problem:
    """Inspectable mathematical pieces shared by every algorithm."""

    K: sc.LinOp
    f: SquaredDistance
    g: L1Penalty
    norm_bound: float

    def __post_init__(self):
        self.smooth = self.f @ self.K
        self.objective = self.g + self.smooth

    @property
    def X(self):
        return self.K.domain

    @property
    def Y(self):
        return self.K.codomain

    def stationarity(self, x):
        # A zero proximal-gradient mapping is the convex optimality condition.
        step = 1 / self.norm_bound**2
        next_x = self.g.prox(self.X.axpy(-step, self.smooth.grad(x), x), step)
        residual = self.X.norm(self.X.axpy(-1, next_x, x)) / step
        return float(residual) / max(1.0, float(self.X.norm(x)))


@dataclass
class Result:
    x: Any
    iterations: int
    stationarity: float
    primal_residual: float = 0.0
    dual_residual: float = 0.0


def proximal_gradient(problem, *, tol=1e-8, maxiter=3000):
    X, g = problem.X, problem.g
    x = X.zeros()
    step = 0.95 / problem.norm_bound**2
    for iteration in range(1, maxiter + 1):
        x = g.prox(X.axpy(-step, problem.smooth.grad(x), x), step)
        residual = problem.stationarity(x)
        if residual <= tol:
            return Result(x, iteration, residual)
    raise RuntimeError("Proximal gradient did not converge")


def pdhg(problem, *, tol=1e-8, maxiter=3000):
    X, Y, K, f, g = problem.X, problem.Y, problem.K, problem.f, problem.g
    x, x_bar, y = X.zeros(), X.zeros(), Y.zeros()
    tau = sigma = 0.9 / problem.norm_bound  # tau*sigma*||K||² <= 0.81 < 1.
    for iteration in range(1, maxiter + 1):
        y = f.conjugate_prox(Y.axpy(sigma, K.apply(x_bar), y), sigma)
        previous = x
        x = g.prox(X.axpy(-tau, K.rapply(y), x), tau)
        x_bar = X.axpy(1, X.axpy(-1, previous, x), x)
        residual = problem.stationarity(x)
        if residual <= tol:
            return Result(x, iteration, residual)
    raise RuntimeError("PDHG did not converge")


def admm(problem, solve_x, *, rho=1.0, tol=1e-8, maxiter=3000):
    """Split x=z; solve_x(rhs, initial) solves (K.H K + rho I)x = rhs.

    The supplied solver must be prepared for this same rho. ADMM never inspects
    whether it is CG, a factorization, or another native implementation.
    """
    X, g = problem.X, problem.g
    x, z, u = X.zeros(), X.zeros(), X.zeros()
    data_rhs = problem.K.rapply(problem.f.b)
    for iteration in range(1, maxiter + 1):
        rhs = X.axpy(rho, X.axpy(-1, u, z), data_rhs)
        x = solve_x(rhs, x)
        previous = z
        z = g.prox(X.add(x, u), 1 / rho)
        u = X.add(u, X.axpy(-1, z, x))
        scale = max(1.0, float(X.norm(x)), float(X.norm(z)))
        primal = float(X.norm(X.axpy(-1, z, x))) / scale
        dual = rho * float(X.norm(X.axpy(-1, previous, z))) / scale
        residual = problem.stationarity(z)
        if max(residual, primal, dual) <= tol:
            return Result(z, iteration, residual, primal, dual)
    raise RuntimeError("ADMM did not converge")


def prepare_cg(problem, rho, *, maxiter):
    """Build the system once; warm-start each subproblem using SpaceCore CG."""
    system = problem.K.H @ problem.K + rho * sc.IdentityLinOp(problem.X)
    work = {"setups": 1, "solves": 0, "krylov_iterations": 0}

    def solve(rhs, initial):
        result = sc.cg(system, rhs, x0=initial, tol=1e-12, maxiter=maxiter, check_every=1)
        work["solves"] += 1
        work["krylov_iterations"] += int(result.num_iters)
        if not bool(result.converged):
            raise RuntimeError("The ADMM x subproblem did not converge")
        return result.x

    return solve, work


def prepare_banded_solve(space, diagonal, lower, rho):
    """Representation-specific optimization: factor K.T K + rho I once.

    K is lower bidiagonal, so its normal system is tridiagonal. LDL.T factors
    and triangular substitutions take O(n) storage/work. Native values are
    accessed only here, outside the ADMM algorithm. No dense matrix is formed.
    """
    pivots = diagonal**2 + rho
    pivots[:-1] += lower**2
    off_diagonal = lower * diagonal[1:]
    multipliers = np.zeros(len(diagonal))
    for i in range(1, len(diagonal)):
        multipliers[i] = off_diagonal[i - 1] / pivots[i - 1]
        pivots[i] -= multipliers[i] * off_diagonal[i - 1]
    work = {"factorizations": 1, "solves": 0}

    def solve(rhs, initial):
        work["solves"] += 1
        values = [space.component(rhs, i) for i in range(len(diagonal))]
        for i in range(1, len(values)):
            values[i] -= multipliers[i] * values[i - 1]
        for i in range(len(values)):
            values[i] /= pivots[i]
        for i in range(len(values) - 2, -1, -1):
            values[i] -= multipliers[i + 1] * values[i + 1]
        return space.from_components(values)

    return solve, work


def build_problem(representation="dense", count=32):
    """Build identical data and a KKT-certified unique optimum for both adapters."""
    if representation not in {"dense", "sparse"} or count < 4:
        raise ValueError("Choose dense/sparse and count >= 4")
    ctx = sc.Context(sc.NumpyOps(), dtype=np.float64)
    X = (DenseCoefficients((count,), ctx) if representation == "dense"
         else SparseCoefficients(count, ctx))
    diagonal = np.linspace(1.0, 2.0, count)
    lower = -0.25
    if representation == "dense":
        matrix = np.diag(diagonal) + np.diag(np.full(count - 1, lower), -1)
        K = sc.DenseLinOp(matrix, X, X)
    else:
        def forward(x):
            return X.from_components(
                diagonal[i] * X.component(x, i) + lower * X.component(x, i - 1)
                for i in range(count)
            )

        def adjoint(y):
            return X.from_components(
                diagonal[i] * X.component(y, i) + lower * X.component(y, i + 1)
                for i in range(count)
            )

        K = sc.MatrixFreeLinOp(forward, adjoint, X, X, euclidean_adjoint=False)

    weight = 0.15
    target_values = np.zeros(count)
    target_values[::4] = np.linspace(0.5, 1.5, len(target_values[::4]))
    target_values[2::7] = -0.8
    # s is a subgradient of ||x*||_1. Solve K.T residual = weight*s.
    subgradient = np.where(target_values != 0, np.sign(target_values), 0.2)
    residual = weight * subgradient
    for i in range(count - 1, -1, -1):
        residual[i] = (residual[i] - (lower * residual[i + 1] if i + 1 < count else 0)) / diagonal[i]
    target = X.from_components(target_values)
    b = X.add(K.apply(target), X.from_components(residual))
    # ||diag(d) + lower*shift|| <= max(d) + |lower|, since ||shift|| <= 1.
    problem = Problem(K, SquaredDistance(X, b), L1Penalty(X, weight), 2.0 + abs(lower))
    return problem, target, diagonal, lower


def run_demo():
    print("min_x 0.5 ||Kx-b||² + 0.15 ||x||₁; 32 coefficients")
    print(f"{'representation / method':31} {'iters':>6} {'objective':>12} {'error':>11} {'KKT':>11} {'ms':>9}")
    for representation in ("dense", "sparse"):
        problem, target, diagonal, lower = build_problem(representation)
        cg_solve, cg_work = prepare_cg(problem, 1.0, maxiter=64)
        direct_solve, direct_work = prepare_banded_solve(problem.X, diagonal, lower, 1.0)
        methods: tuple[tuple[str, Callable], ...] = (
            ("proximal gradient", lambda: proximal_gradient(problem)),
            ("PDHG", lambda: pdhg(problem)),
            ("ADMM / CG", lambda: admm(problem, cg_solve)),
            ("ADMM / prepared LDL", lambda: admm(problem, direct_solve)),
        )
        for name, method in methods:
            started = perf_counter()
            result = method()
            elapsed = 1000 * (perf_counter() - started)
            error = float(problem.X.norm(problem.X.axpy(-1, target, result.x)))
            assert error < 1e-6, (representation, name, error)
            print(f"{representation + ' / ' + name:31} {result.iterations:6d} "
                  f"{float(problem.objective(result.x)):12.8f} {error:11.2e} "
                  f"{result.stationarity:11.2e} {elapsed:9.2f}")
        print(f"  CG work: {cg_work}; prepared solve work: {direct_work}")
    print("All methods agree with the independently constructed KKT solution.")
    print("Timings include diagnostics, exclude setup, and are illustrative single runs.")


if __name__ == "__main__":
    run_demo()
