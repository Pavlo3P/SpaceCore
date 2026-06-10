"""Weighted Tikhonov inverse problem on non-Euclidean spaces.

Run with:

    python examples/weighted_tikhonov.py

The example solves the same inverse problem twice: once with an independent
NumPy dense normal equation, and once with SpaceCore spaces, metric adjoints,
operator algebra, and conjugate gradients.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Any

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import spacecore as sc


@dataclass(frozen=True)
class WeightedTikhonovProblem:
    """Coordinate data for the weighted Tikhonov example."""

    M: np.ndarray
    Gx: np.ndarray
    Gy: np.ndarray
    b: np.ndarray
    lam: float

    @property
    def x_weights(self) -> np.ndarray:
        """Return the diagonal metric weights for the domain space."""
        return np.diag(self.Gx).copy()

    @property
    def y_weights(self) -> np.ndarray:
        """Return the diagonal metric weights for the codomain space."""
        return np.diag(self.Gy).copy()


@dataclass(frozen=True)
class SolveDiagnostics:
    """Diagnostics for a candidate solution."""

    objective: float
    residual_norm_y: float
    regularization_norm_x: float
    first_order_residual_norm: float


@dataclass(frozen=True)
class SpaceCoreSolve:
    """SpaceCore solution and diagnostics."""

    x: np.ndarray
    diagnostics: SolveDiagnostics
    cg_converged: bool
    cg_num_iters: int
    cg_residual_norm: float


@dataclass(frozen=True)
class AdjointDiagnostics:
    """Errors for the correct metric adjoint and the wrong transpose adjoint."""

    metric_identity_error: float
    wrong_transpose_identity_error: float


def make_weighted_tikhonov_problem(
    n: int = 32,
    m: int = 48,
    lam: float = 1e-2,
    seed: int = 0,
) -> WeightedTikhonovProblem:
    """Create deterministic weighted least-squares data.

    The metrics are diagonal SPD matrices. This keeps the example on public
    SpaceCore APIs while still making the coordinate transpose mathematically
    wrong as an adjoint.
    """
    if n <= 0 or m <= 0:
        raise ValueError("n and m must be positive.")
    if lam <= 0:
        raise ValueError("lam must be positive.")

    rng = np.random.default_rng(seed)
    M = rng.normal(size=(m, n)) / np.sqrt(float(n))

    x_weights = 0.7 + np.linspace(0.0, 1.3, n) + 0.15 * rng.random(n)
    y_weights = 1.1 + np.linspace(0.0, 1.7, m) + 0.20 * rng.random(m)
    Gx = np.diag(x_weights)
    Gy = np.diag(y_weights)

    x_true = rng.normal(size=n)
    noise = 0.03 * rng.normal(size=m)
    b = M @ x_true + noise

    return WeightedTikhonovProblem(M=M, Gx=Gx, Gy=Gy, b=b, lam=float(lam))


def objective_value(problem: WeightedTikhonovProblem, x: np.ndarray) -> float:
    """Return the weighted Tikhonov objective at ``x``."""
    residual = problem.M @ x - problem.b
    data = residual @ problem.Gy @ residual
    regularization = x @ problem.Gx @ x
    return float(0.5 * data + 0.5 * problem.lam * regularization)


def first_order_residual(problem: WeightedTikhonovProblem, x: np.ndarray) -> np.ndarray:
    """Return the dense first-order residual in coordinates."""
    lhs = problem.M.T @ problem.Gy @ problem.M + problem.lam * problem.Gx
    rhs = problem.M.T @ problem.Gy @ problem.b
    return lhs @ x - rhs


def diagnostics(problem: WeightedTikhonovProblem, x: np.ndarray) -> SolveDiagnostics:
    """Compute dense diagnostics for a candidate solution."""
    residual = problem.M @ x - problem.b
    return SolveDiagnostics(
        objective=objective_value(problem, x),
        residual_norm_y=float(np.sqrt(residual @ problem.Gy @ residual)),
        regularization_norm_x=float(np.sqrt(x @ problem.Gx @ x)),
        first_order_residual_norm=float(np.linalg.norm(first_order_residual(problem, x))),
    )


def dense_reference_solve(problem: WeightedTikhonovProblem) -> tuple[np.ndarray, SolveDiagnostics]:
    """Solve the independent dense NumPy reference system."""
    lhs = problem.M.T @ problem.Gy @ problem.M + problem.lam * problem.Gx
    rhs = problem.M.T @ problem.Gy @ problem.b
    x_ref = np.linalg.solve(lhs, rhs)
    return x_ref, diagnostics(problem, x_ref)


def build_spacecore_operator(
    problem: WeightedTikhonovProblem,
    *,
    enable_checks: bool = True,
) -> tuple[Any, Any, Any, Any]:
    """Build ``X``, ``Y``, ``A : X -> Y``, and a SpaceCore context."""
    ctx = sc.Context(sc.NumpyOps(), dtype=np.float64, enable_checks=enable_checks)
    X = sc.DenseVectorSpace(
        (problem.M.shape[1],),
        ctx,
        geometry=sc.WeightedInnerProduct(ctx.asarray(problem.x_weights)),
    )
    Y = sc.DenseVectorSpace(
        (problem.M.shape[0],),
        ctx,
        geometry=sc.WeightedInnerProduct(ctx.asarray(problem.y_weights)),
    )
    A = sc.DenseLinOp(ctx.asarray(problem.M), X, Y, ctx)
    return ctx, X, Y, A


def solve_with_spacecore(
    problem: WeightedTikhonovProblem,
    *,
    tol: float = 1e-12,
    maxiter: int | None = None,
) -> SpaceCoreSolve:
    """Solve the weighted problem with SpaceCore spaces and CG."""
    _ctx, X, _Y, A = build_spacecore_operator(problem)
    normal = A.H @ A + problem.lam * sc.IdentityLinOp(X)
    rhs = A.H.apply(A.codomain.ctx.asarray(problem.b))
    if maxiter is None:
        maxiter = 2 * problem.M.shape[1]

    result = sc.cg(normal, rhs, tol=tol, atol=0.0, maxiter=maxiter, check_every=1)
    x_sc = np.asarray(result.x, dtype=np.float64)
    return SpaceCoreSolve(
        x=x_sc,
        diagnostics=diagnostics(problem, x_sc),
        cg_converged=bool(result.converged),
        cg_num_iters=int(result.num_iters),
        cg_residual_norm=float(result.residual_norm),
    )


def adjoint_diagnostics(problem: WeightedTikhonovProblem, seed: int = 17) -> AdjointDiagnostics:
    """Compare SpaceCore's metric adjoint with the coordinate transpose."""
    ctx, X, Y, A = build_spacecore_operator(problem)
    rng = np.random.default_rng(seed)
    x = ctx.asarray(rng.normal(size=problem.M.shape[1]))
    y = ctx.asarray(rng.normal(size=problem.M.shape[0]))

    lhs = Y.inner(A.apply(x), y)
    rhs_metric = X.inner(x, A.H.apply(y))
    rhs_wrong = X.inner(x, ctx.asarray(problem.M.T @ np.asarray(y)))

    return AdjointDiagnostics(
        metric_identity_error=float(abs(np.asarray(lhs - rhs_metric))),
        wrong_transpose_identity_error=float(abs(np.asarray(lhs - rhs_wrong))),
    )


def run_example() -> dict[str, Any]:
    """Run the example and return all values useful for tests and docs."""
    problem = make_weighted_tikhonov_problem()
    x_ref, ref_diag = dense_reference_solve(problem)
    sc_solve = solve_with_spacecore(problem)
    adjoint = adjoint_diagnostics(problem)
    rel_solution_error = float(
        np.linalg.norm(sc_solve.x - x_ref) / max(1.0, np.linalg.norm(x_ref))
    )
    return {
        "problem": problem,
        "x_reference": x_ref,
        "reference": ref_diag,
        "spacecore": sc_solve,
        "adjoint": adjoint,
        "relative_solution_error": rel_solution_error,
        "objective_difference": abs(sc_solve.diagnostics.objective - ref_diag.objective),
    }


def _format_float(value: float) -> str:
    return f"{value:.6e}"


def main() -> None:
    """Run the worked example and print a compact comparison table."""
    result = run_example()
    ref = result["reference"]
    sc_solve = result["spacecore"]
    adj = result["adjoint"]

    print("Weighted Tikhonov inverse problem on non-Euclidean spaces")
    print(f"CG converged: {sc_solve.cg_converged} in {sc_solve.cg_num_iters} iterations")
    print()
    print(f"{'quantity':36s} {'reference':>14s} {'SpaceCore':>14s} {'difference':>14s}")
    print("-" * 82)
    print(
        f"{'objective value':36s} "
        f"{_format_float(ref.objective):>14s} "
        f"{_format_float(sc_solve.diagnostics.objective):>14s} "
        f"{_format_float(result['objective_difference']):>14s}"
    )
    print(
        f"{'relative solution error':36s} "
        f"{'--':>14s} {'--':>14s} "
        f"{_format_float(result['relative_solution_error']):>14s}"
    )
    print(
        f"{'first-order residual norm':36s} "
        f"{_format_float(ref.first_order_residual_norm):>14s} "
        f"{_format_float(sc_solve.diagnostics.first_order_residual_norm):>14s} "
        f"{'--':>14s}"
    )
    print(
        f"{'metric-adjoint identity error':36s} "
        f"{'--':>14s} {_format_float(adj.metric_identity_error):>14s} {'--':>14s}"
    )
    print(
        f"{'wrong-transpose identity error':36s} "
        f"{'--':>14s} {_format_float(adj.wrong_transpose_identity_error):>14s} {'--':>14s}"
    )


if __name__ == "__main__":
    main()
