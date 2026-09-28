"""Reproduce the representation research on the 2026-09-26 working tree.

Run from the repository root:
    .venv/bin/python docs/dev/representation_vision_probe.py

This is a research snapshot, not a permanent regression suite: several checks
deliberately record current limitations and should change after the refactor.
The MPS example tests extension feasibility, not scalability or compression.
"""

from dataclasses import dataclass
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import spacecore as sc  # noqa: E402


def observe(name, operation):
    """Print an observed success or failure without hiding its exception type."""
    try:
        result = operation()
    except Exception as exc:
        print(f"{name}: {type(exc).__name__}: {str(exc).splitlines()[0]}")
    else:
        print(f"{name}: {result}")


@dataclass(frozen=True)
class TwoSiteMPS:
    """Two local cores, with contraction left[i, r] * right[r, j]."""

    left: np.ndarray
    right: np.ndarray

    def densify(self):
        raise AssertionError("Dense state materialization is forbidden in this probe")


class MPSHilbertSpace(sc.InnerProductSpace):
    """Small real Hilbert space represented by unrestricted two-site factors.

    The adapter is NumPy-only. No shape, flatten, unflatten, or MPS truncation
    is provided. The geometry is the ambient Euclidean Hilbert pairing, not
    the Euclidean inner product of the stored factor entries.
    """

    geometry = sc.EuclideanInnerProduct()

    def zeros(self):
        return TwoSiteMPS(np.zeros((2, 1)), np.zeros((1, 2)))

    def add(self, x, y):
        return TwoSiteMPS(
            np.concatenate((x.left, y.left), axis=1),
            np.concatenate((x.right, y.right), axis=0),
        )

    def scale(self, a, x):
        return TwoSiteMPS(a * x.left, x.right)

    def inner(self, x, y):
        if x is y:
            # A norm formed from two Gram matrices can catastrophically cancel
            # for redundant sum factors. Canonicalize the first core instead.
            _, triangular = np.linalg.qr(x.left, mode="reduced")
            last_core = triangular @ x.right
            return np.vdot(last_core, last_core)
        left_overlap = x.left.conj().T @ y.left
        right_overlap = x.right.conj() @ y.right.T
        return np.sum(left_overlap * right_overlap)

    def _check_member(self, x):
        if not isinstance(x, TwoSiteMPS):
            raise TypeError("Expected TwoSiteMPS")
        for core in (x.left, x.right):
            if not isinstance(core, np.ndarray) or core.dtype != np.dtype(np.float64):
                raise TypeError("This example requires NumPy float64 cores")
        if (
            x.left.ndim != 2 or x.right.ndim != 2
            or x.left.shape[0] != 2 or x.right.shape[1] != 2
            or x.left.shape[1] != x.right.shape[0]
        ):
            raise ValueError("Invalid physical dimensions or MPS bond")

    def _convert(self, ctx):
        if ctx != self.ctx:
            raise NotImplementedError("This small research adapter is NumPy float64 only")
        return type(self)(ctx, check_level=self.check_level)


def bound_element_probes(ctx):
    X = sc.DenseCoordinateSpace((2,), ctx)
    W = sc.DenseCoordinateSpace(
        (2,), ctx, geometry=sc.WeightedInnerProduct(np.array([2.0, 5.0]))
    )
    A = sc.IdentityLinOp(X)
    x = sc.DenseElement(X, np.array([1.0, 2.0]))
    w = sc.DenseElement(W, np.array([1.0, 2.0]))
    observe("A @ DenseElement", lambda: type(A @ x).__name__)
    observe("A.apply(DenseElement)", lambda: A.apply(x))
    observe("X.inner(DenseElement, DenseElement)", lambda: X.inner(x, x))
    observe("cg(A, DenseElement)", lambda: sc.cg(A, x, maxiter=3))
    observe("functional(DenseElement)", lambda: sc.SquaredL2NormFunctional(X)(x))
    observe("Different-geometry element accepted and rebound", lambda: (A @ w).space == X)

    H = sc.HermitianSpace(2, ctx=ctx)
    p = sc.LowRankElement(H, np.ones((2, 1)), np.array([1.0]))
    observe("Identity @ LowRankElement", lambda: type(sc.IdentityLinOp(H) @ p).__name__)
    invalid = sc.LowRankElement(H, np.array([[1.0], [1.0j]]), np.array([1.0]))
    observe("Complex factor accepted in real space at construction", lambda: type(invalid).__name__)
    observe("Same invalid element fails on densify", invalid.densify)
    observe(
        "Hermitian flatten(unflatten([0,1,0,0]))",
        lambda: H.flatten(H.unflatten(np.array([0.0, 1.0, 0.0, 0.0]))).tolist(),
    )


def mps_probe(ctx):
    X = MPSHilbertSpace(ctx)

    def apply(x):
        # Positive-definite local action diag(1,2) on the first physical site.
        return TwoSiteMPS(np.array([1.0, 2.0])[:, None] * x.left, x.right)

    A = sc.MatrixFreeLinOp(apply, apply, X, X, euclidean_adjoint=False)
    b = TwoSiteMPS(np.eye(2), np.array([[1.0, 2.0], [3.0, 4.0]]))
    result = sc.cg(A, b, maxiter=4, tol=1e-10, check_every=1)
    residual = X.norm(X.add(apply(result.x), X.scale(-1, b)))
    # Verify individual entries without assembling the physical state array.
    for i in range(2):
        for j in range(2):
            actual = result.x.left[i] @ result.x.right[:, j]
            expected = (b.left[i] @ b.right[:, j]) / (i + 1)
            np.testing.assert_allclose(actual, expected, atol=1e-12)
    assert bool(result.converged) and float(residual) < 1e-10
    print("MPS CG:", {
        "converged": bool(result.converged),
        "iterations": int(result.num_iters),
        "recomputed_residual": float(residual),
        "input_bond": b.left.shape[1],
        "output_bond": result.x.left.shape[1],
        "dense_state_materializations": 0,
    })
    observe("MPS CG with default maxiter", lambda: sc.cg(A, b).converged)
    strict = sc.MatrixFreeLinOp(
        apply, apply, X, X, euclidean_adjoint=False, check_level="strict"
    )
    observe("MPS CG strict with maxiter", lambda: sc.cg(strict, b, maxiter=4).converged)
    observe("TreeSpace with MPS leaf", lambda: sc.TreeSpace.from_leaf_spaces([X]))


def main():
    ctx = sc.Context(sc.NumpyOps(), dtype=np.float64)
    bound_element_probes(ctx)
    mps_probe(ctx)
    # These tiny dense arrays are mathematical witnesses, not solver storage.
    a, b = np.array([1.0, 0.0]), np.array([0.0, 1.0])
    assert np.linalg.matrix_rank(np.outer(a, a) + np.outer(b, b)) == 2
    print("Closure witness: two bond-1 product states can sum to a rank-2 state")

    def model(theta, t):
        return theta**2 * t

    assert model(1 + 1, 1) != model(1, 1) + model(1, 1)
    print("Parameter witness: Phi(1+1)=4 while Phi(1)+Phi(1)=2")


if __name__ == "__main__":
    main()
