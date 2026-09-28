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

### Build a problem and its algorithms

[One problem, three algorithms, two representations](examples/structured_optimization.md)
shows proximal gradient, PDHG, and ADMM using the same mathematical objects with
dense arrays or custom coefficient values. A prepared factorization replaces
ADMM's inner CG solve without changing the iteration.

```bash
python -m examples.structured_optimization
```

[Metric proximal operations](docs/source/design/proximal_contract.rst) add
optional `f.prox(x, step)`, `f.conjugate_prox(x, step)`, and set indicators.
Run `python -m examples.proximal_contract` for unchanged proximal-gradient and
PDHG loops across dense/custom elements and Euclidean/weighted metrics.

## Features

- **Spaces:** dense vectors, matrices, Jordan algebras, nested trees, and stacks.
- **Operators:** dense, sparse, diagonal, matrix-free, and block operators;
  lazy sums, scaling, and composition.
- **Functionals:** linear, quadratic, least squares, norms, entropy, KL, and Huber;
  geometry-aware gradients and proximal maps.
- **Solvers:** CG, LSQR, Lanczos, power iteration, and matrix exponential action.
- **Optimizer adapters:** SciPy and Optax, with metric-to-coordinate gradient conversion.
- **Backends:** NumPy, JAX, Torch, and CuPy. See
  [conformance](https://pavlo3p.github.io/SpaceCore/design/backend_conformance.html) and
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
