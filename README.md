# SpaceCore

[![CI](https://github.com/Pavlo3P/SpaceCore/actions/workflows/ci.yml/badge.svg)](https://github.com/Pavlo3P/SpaceCore/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/spacecore.svg)](https://pypi.org/project/spacecore/)
[![Python](https://img.shields.io/pypi/pyversions/spacecore.svg)](https://pypi.org/project/spacecore/)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)

Typed vector spaces and linear operators for NumPy, JAX, Torch, and CuPy.
A space knows its inner product; an operator `A : X -> Y` knows its spaces, so
its adjoint is the metric adjoint, not the matrix transpose.

## Install

```bash
pip install spacecore            # NumPy backend
pip install "spacecore[jax]"     # or [torch], [cupy], [optax]
```

Python 3.11+.

## Example

```python
import numpy as np
import spacecore as sc

ctx = sc.Context(sc.NumpyOps(), dtype=np.float64)
X = sc.DenseCoordinateSpace((2,), ctx, geometry=sc.WeightedInnerProduct(ctx.asarray([2.0, 5.0])))
A = sc.DenseLinOp(ctx.asarray([[1.0, 2.0], [0.0, 1.0]]), X, X, ctx)
x = ctx.asarray([1.0, 2.0])

print(X.inner(x, x))                  # 22.0          (2·1 + 5·4)
print(A.rapply(x))                    # [1.  2.8]     metric adjoint; A.T @ x is [1. 4.]

N = A.H @ A + 0.1 * sc.IdentityLinOp(X, ctx)
print(N.is_hermitian())               # True          inferred from the structure
print(sc.cg(N, x, tol=1e-12).converged)  # True
```

Nested structures are spaces too:

```python
P = sc.DenseCoordinateSpace((3,), ctx)
R = sc.DenseCoordinateSpace((1,), ctx)
B = sc.BlockDiagonalLinOp({
    "position": sc.IdentityLinOp(P, ctx),
    "twist": {"angular": sc.DiagonalLinOp(ctx.asarray([0.5]), R, ctx)},
})
y = B.dom.element({"position": ctx.asarray([1.0, 0.0, 0.0]), "twist": {"angular": ctx.asarray([4.0])}})

print(B.apply(y))         # {'position': array([1., 0., 0.]), 'twist': {'angular': array([2.])}}
print(B.dom.inner(y, y))  # 17.0
```

## What's in it

- **Spaces:** `DenseCoordinateSpace`, `DenseVectorSpace`, `HermitianSpace`,
  `ElementwiseJordanSpace`, `TreeSpace` (nested direct products via `optree`), `StackedSpace`.
- **Operators:** dense, sparse, diagonal, matrix-free, identity, zero, block; lazy
  sums, scalings and compositions. `apply` / `rapply` (forward / metric adjoint).
- **Functionals:** linear, quadratic, least squares, Lp/L1 norms, entropy, KL, Huber,
  nuclear norm, and metric-aware proximal maps (`prox_l1`, `prox_l2sq`, `project_nonneg`).
  Gradients are Riesz gradients in the domain geometry.
- **Linear algebra:** `cg`, `lsqr`, `lanczos_smallest`, `power_iteration`, `expm_multiply`.
  Small by design; they reject provably invalid inputs (e.g. `cg` on a non-self-adjoint map).
- **Optimizers:** `minimize_scipy`, `line_search_scipy`, `minimize_optax` — the
  metric-to-coordinate gradient handoff is done for you.
- **Backends:** `NumpyOps` always; `JaxOps`, `TorchOps`, `CuPyOps` when installed.
  Cross-backend behaviour: [conformance matrix](https://pavlo3p.github.io/SpaceCore/design/backend_conformance.html),
  [deviations](https://pavlo3p.github.io/SpaceCore/design/backend_deviations.html).

## Validation

Each space, operator and functional has a `check_level`: `"none"`, `"cheap"`
(shape/dtype/structure), `"standard"` (+ membership, Hermiticity), `"strict"`
(+ bounded probes). Set it per object with `check_level=`, or globally with
`sc.set_check_level(...)` / `sc.use_check_level(...)`. Run hot paths at `"none"`,
tests at `"strict"`. Failures raise `sc.SpaceValidationError`.

Solvers validate once at entry, not per iteration. Batching goes through
`vapply` / `rvapply` / `vvalue`; iterative solvers are unbatched.

## Docs

[Tutorials](https://pavlo3p.github.io/SpaceCore/tutorials/index.html) ·
[Design notes](https://pavlo3p.github.io/SpaceCore/design/index.html) ·
[API](https://pavlo3p.github.io/SpaceCore/api/index.html) ·
[Release notes](https://pavlo3p.github.io/SpaceCore/release_notes.html)

Experimental (0.x): the API may change before 1.0.
Contributions welcome — see [CONTRIBUTING.md](CONTRIBUTING.md). Apache 2.0.

```bibtex
@software{spacecore,
  author = {Pavlo Pelikh},
  title = {SpaceCore: Backend-aware vector spaces and linear operators},
  url = {https://github.com/Pavlo3P/SpaceCore},
  year = {2026},
}
```
