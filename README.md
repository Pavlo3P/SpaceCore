# SpaceCore

[![CI](https://github.com/Pavlo3P/SpaceCore/actions/workflows/ci.yml/badge.svg)](https://github.com/Pavlo3P/SpaceCore/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/spacecore.svg)](https://pypi.org/project/spacecore/)
[![Python](https://img.shields.io/pypi/pyversions/spacecore.svg)](https://pypi.org/project/spacecore/)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)

SpaceCore provides typed vector spaces, structured elements, linear operators,
functionals, and small linear-algebra utilities for backend-aware numerical
code. An operator is a typed map `A : X -> Y` between spaces, not merely an
array. Spaces carry the rules needed to validate elements, compute inner
products, flatten structured values, and interpret adjoints.

The execution backend is explicit. A `Context` owns the backend operations,
default dtype, and validation policy used by spaces and operators. NumPy is the
baseline backend; JAX, Torch, and CuPy are optional backends when their extras
are installed.

SpaceCore's native solvers are intentionally small. They are a correctness
baseline and substrate layer for space-aware algorithms, not a replacement for
mature solver ecosystems such as SciPy, PETSc, Krylov.jl, or PyLops. External
adapters and backend-specific fast paths can be layered on top where breadth or
performance is required.

## Install

```bash
pip install spacecore
pip install "spacecore[jax]"
pip install "spacecore[torch]"
pip install "spacecore[cupy]"
```

Python 3.11+ is required.

## Quick Start

```python
import numpy as np
import spacecore as sc

ctx = sc.Context(sc.NumpyOps(), dtype=np.float64)
X = sc.DenseCoordinateSpace((2,), ctx)
A = sc.DenseLinOp(ctx.asarray([[2.0, 0.0], [0.0, 3.0]]), X, X, ctx)
b = ctx.asarray([4.0, 9.0])

result = sc.cg(A, b, tol=1e-12, maxiter=10)
print(result.x)
print(result.converged)
```

Expected output:

```text
[2. 3.]
True
```

## Core Ideas

**Spaces.** `DenseCoordinateSpace`, `DenseVectorSpace`,
`ElementwiseJordanSpace`, `EuclideanElementwiseJordanSpace`, `HermitianSpace`,
`ProductSpace`, and `StackedSpace` describe element structure and geometry.
Dense coordinate spaces can use Euclidean or weighted inner products.

```python
import numpy as np
import spacecore as sc

ctx = sc.Context(sc.NumpyOps(), dtype=np.float64)
weights = ctx.asarray([2.0, 5.0])
X = sc.DenseCoordinateSpace((2,), ctx, geometry=sc.WeightedInnerProduct(weights))
x = ctx.asarray([1.0, 2.0])
y = ctx.asarray([3.0, 4.0])

print(X.inner(x, y))
print(X.riesz(x))
```

Expected output:

```text
46.0
[ 2. 10.]
```

**Linear operators.** `DenseLinOp`, `SparseLinOp`, `DiagonalLinOp`,
`MatrixFreeLinOp`, `IdentityLinOp`, `ZeroLinOp`, and the algebraic operators
represent maps `A : X -> Y`. `apply` computes the forward map. `rapply`
computes the metric adjoint: the coordinate conjugate transpose only agrees
with it when both spaces use Euclidean geometry.

```python
import numpy as np
import spacecore as sc

ctx = sc.Context(sc.NumpyOps(), dtype=np.float64)
X = sc.DenseCoordinateSpace((2,), ctx)
A = sc.DiagonalLinOp(ctx.asarray([2.0, 3.0]), X, ctx)

print(A.apply(ctx.asarray([1.0, 2.0])))
print(A.rapply(ctx.asarray([1.0, 1.0])))
```

Expected output:

```text
[2. 6.]
[2. 3.]
```

**Functionals.** `LinearFunctional`, `InnerProductFunctional`,
`MatrixFreeLinearFunctional`, `QuadraticForm`, and `LinOpQuadraticForm` model
scalar-valued maps on spaces. Gradients are represented in the domain geometry.

**Linear algebra.** `cg`, `lsqr`, `lanczos_smallest`, `power_iteration`, and
`expm_multiply` operate on SpaceCore operators and spaces. They document their
mathematical preconditions; for example `cg` expects a square Hermitian positive
definite map `A : X -> X` with respect to `X.inner`.

**Backends.** `NumpyOps` is always available. `JaxOps`, `TorchOps`, and
`CuPyOps` are exported only when their optional dependencies are installed.
Backend portability means SpaceCore uses the same abstract operations and data
model; it does not erase backend-specific dtype, device, sparse, tracing, or
autograd behavior.

## Batching

A space describes one element type. Batched computation is handled by vectorized
application methods such as `vapply`, `rvapply`, `vvalue`, and backend
vectorization. Batching does not change the mathematical domain or codomain of
an operator unless the operator itself is explicitly built over a stacked or
product space.

## Documentation

- [Tutorials](https://pavlo3p.github.io/SpaceCore/tutorials/index.html)
- [Design notes](https://pavlo3p.github.io/SpaceCore/design/index.html)
- [API reference](https://pavlo3p.github.io/SpaceCore/api/index.html)
- [Release notes](https://pavlo3p.github.io/SpaceCore/release_notes.html)

## Project Status

SpaceCore is experimental `0.3.x` software. Core abstractions are usable for
research and prototyping, but API details may still change before a stable
release.

## Contributing

Bug reports, feature requests, and PRs are welcome. See [CONTRIBUTING.md](CONTRIBUTING.md).

## License

Apache 2.0. See [LICENSE](LICENSE).

## Citation

```bibtex
@software{spacecore,
  author = {Pavlo Pelikh},
  title = {SpaceCore: Backend-aware vector spaces and linear operators},
  url = {https://github.com/Pavlo3P/SpaceCore},
  year = {2026},
}
```
