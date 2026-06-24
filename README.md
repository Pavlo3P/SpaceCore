# SpaceCore

[![CI](https://github.com/Pavlo3P/SpaceCore/actions/workflows/ci.yml/badge.svg)](https://github.com/Pavlo3P/SpaceCore/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/spacecore.svg)](https://pypi.org/project/spacecore/)
[![Python](https://img.shields.io/pypi/pyversions/spacecore.svg)](https://pypi.org/project/spacecore/)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)

In most numerical code a vector is just an array. The mathematics that gives it
meaning — which space it belongs to, what inner product measures its length, how
an operator's adjoint is actually defined — lives in the author's head and in
comments rather than in the code. When that structure stays implicit, it is easy
to combine elements from incompatible spaces, assume a Euclidean inner product
where the geometry is weighted, or take a matrix transpose where the true
adjoint is something else.

SpaceCore makes that structure explicit. A space is a typed object that knows
how to validate its elements, compute inner products, flatten structured values,
and represent adjoints. An operator is a typed map `A : X -> Y` between spaces,
not merely an array, so the spaces it connects — and the rules they carry — are
part of its definition.

The execution backend is explicit too. A `Context` owns the backend operations,
the default dtype, and the validation policy used by spaces and operators. NumPy
is the baseline backend; JAX, Torch, and CuPy are optional backends when their
extras are installed.

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
`TreeSpace`, and `StackedSpace` describe element structure and geometry. Dense
coordinate spaces can use Euclidean or weighted inner products. Each space also
exposes a `field` (`"real"` or `"complex"`) — the mathematical scalar-field
contract derived from the context dtype — which capability guards consult
instead of inspecting precision-bearing dtypes.

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
print(X.field)
```

Expected output:

```text
46.0
[ 2. 10.]
real
```

**Linear operators.** `DenseLinOp`, `SparseLinOp`, `DiagonalLinOp`,
`MatrixFreeLinOp`, `IdentityLinOp`, `ZeroLinOp`, and the lazy algebraic
operators (`ComposedLinOp`, `ScaledLinOp`, `SumLinOp`) represent maps
`A : X -> Y`. `apply` computes the forward map. `rapply` computes the metric
adjoint: the coordinate conjugate transpose only agrees with it when both spaces
use Euclidean geometry. The algebraic operators infer Hermiticity structurally,
so a normal operator `A.H @ A + lam * Identity` is recognized as self-adjoint.

```python
import numpy as np
import spacecore as sc

ctx = sc.Context(sc.NumpyOps(), dtype=np.float64)
X = sc.DenseCoordinateSpace((2,), ctx)
A = sc.DiagonalLinOp(ctx.asarray([2.0, 3.0]), X, ctx)

print(A.apply(ctx.asarray([1.0, 2.0])))
print(A.rapply(ctx.asarray([1.0, 1.0])))

M = sc.DenseLinOp(ctx.asarray([[1.0, 2.0], [0.0, 1.0]]), X, X, ctx)
print((M.H @ M).is_hermitian())
```

Expected output:

```text
[2. 6.]
[2. 3.]
True
```

**Structured (tree) spaces.** `TreeSpace` is the finite direct-product space:
it organizes leaf spaces by an `optree` definition, so an element can be a
named, nested Python structure (tuple, dict, namedtuple, or any nesting of
them) rather than a flat vector. Operations like `inner` act across the whole
tree, and `BlockDiagonalLinOp`, `BlockMatrixLinOp`, and the `TreeLinOp` base
operate over tree domains with correct metric-adjoint behavior.

A block-diagonal operator can be built directly from a matching tree of blocks;
its domain and codomain are inferred from the blocks, so the names and nesting
are carried through to the result.

```python
import numpy as np
import spacecore as sc

ctx = sc.Context(sc.NumpyOps(), dtype=np.float64)
position = sc.DenseCoordinateSpace((3,), ctx)
linear = sc.DenseCoordinateSpace((3,), ctx)
angular = sc.DenseCoordinateSpace((1,), ctx)

# A named, nested tree of blocks: a top-level "position" block and a grouped
# "twist" block with its own "linear" and "angular" sub-blocks.
A = sc.BlockDiagonalLinOp({
    "position": sc.IdentityLinOp(position, ctx),
    "twist": {
        "linear": sc.DiagonalLinOp(ctx.asarray([2.0, 2.0, 2.0]), linear, ctx),
        "angular": sc.DiagonalLinOp(ctx.asarray([0.5]), angular, ctx),
    },
})

x = A.dom.element({
    "position": ctx.asarray([1.0, 0.0, 0.0]),
    "twist": {
        "linear": ctx.asarray([1.0, 1.0, 1.0]),
        "angular": ctx.asarray([4.0]),
    },
})

print(A.apply(x))
print(A.dom.inner(x, x))
```

Expected output:

```text
{'position': array([1., 0., 0.]), 'twist': {'linear': array([2., 2., 2.]), 'angular': array([2.])}}
20.0
```

**Functionals.** `LinearFunctional`, `InnerProductFunctional`,
`MatrixFreeLinearFunctional`, `QuadraticForm`, and `LinOpQuadraticForm` model
scalar-valued maps on spaces. Gradients are represented in the domain geometry.

**Everyday toolbox.** `spacecore.functional.tools` (re-exported at the top
level) adds named constructors over that machinery, with no new core types:
`least_squares` for `½‖Ax−b‖²`, coordinate norms (`SquaredL2NormFunctional`,
`LpNormFunctional`, `L1NormFunctional`), `NegativeEntropyFunctional`,
`KLDivergenceFunctional`, `HuberFunctional`, the spectral
`SpectralLpNormFunctional`/`NuclearNormFunctional`, and the metric-aware
proximal primitive `generalized_shrinkage` with the wrappers `prox_l1`,
`prox_l2sq`, and `project_nonneg`. Each objective's gradient is the metric
(Riesz) gradient under the domain geometry, and the proximal step is taken in
the space metric.

**Linear algebra.** `cg`, `lsqr`, `lanczos_smallest`, `power_iteration`, and
`expm_multiply` operate on SpaceCore operators and spaces. They document their
mathematical preconditions and reject provably invalid inputs at entry; for
example `cg` expects a square Hermitian positive definite map `A : X -> X` with
respect to `X.inner`, and rejects an operator that is provably non-self-adjoint
in its geometry. These native solvers are intentionally small — a correctness
baseline and a substrate for space-aware algorithms rather than a full solver
suite; backend-specific fast paths and external adapters can be layered on top.

**External optimizer adapters.** `minimize_scipy`, `line_search_scipy`, and
`minimize_optax` (in `spacecore.optimize`) drive mature external optimizers from
a `Functional`. Because SpaceCore gradients are metric (Riesz) gradients while
NumPy/JAX optimizers expect coordinate gradients, each adapter applies the
`X.riesz(F.grad(x))` handoff for you — the identity on a Euclidean space and
mandatory on a weighted one. The external optimizer owns the loop; the SciPy
adapters require a real domain and `minimize_optax` needs a JAX backend
(`pip install spacecore[optax]`).

**Backends.** `NumpyOps` is always available. `JaxOps`, `TorchOps`, and
`CuPyOps` are exported only when their optional dependencies are installed.
Backend portability means SpaceCore uses the same abstract operations and data
model; it does not erase backend-specific dtype, device, sparse, tracing, or
autograd behavior. Cross-backend agreement is documented in the
[backend conformance matrix](https://pavlo3p.github.io/SpaceCore/design/backend_conformance.html)
and [deviation catalog](https://pavlo3p.github.io/SpaceCore/design/backend_deviations.html).

## Validation Policy

A `Context` carries a `check_level` that determines how aggressively spaces,
operators, functionals, and solver preconditions validate their inputs. The
ordered levels are `CHECK_LEVELS = ("none", "cheap", "standard", "strict")`:
`cheap` covers shape/dtype/backend/tree-structure, `standard` adds membership
and Hermitian checks, and `strict` adds bounded expensive probes. Checks are
opt-in per context, so hot paths can run unvalidated while development and tests
run strict.

```python
import numpy as np
import spacecore as sc

ctx = sc.Context(sc.NumpyOps(), dtype=np.float64, check_level="standard")
X = sc.DenseCoordinateSpace((2,), ctx)
A = sc.DenseLinOp(ctx.asarray([[2.0, 0.0], [0.0, 3.0]]), X, X, ctx)

try:
    A.apply(ctx.asarray([1.0, 2.0, 3.0]))  # wrong shape
except sc.SpaceValidationError as exc:
    print(exc)
```

Expected output:

```text
Expected shape (2,), got (3,)
```

## Batching

A space describes one element type. Batched computation is handled by vectorized
application methods such as `vapply`, `rvapply`, `vvalue`, and backend
vectorization. Batching does not change the mathematical domain or codomain of
an operator unless the operator itself is explicitly built over a stacked or
tree space. Iterative solvers are unbatched in this release; batched-input
invocations raise a clear shape error.

## Performance

The typed, validated API is designed to add little run-time overhead over
working with raw arrays: operator and functional hot paths use check-free fast
paths, and iterative solvers validate their inputs once at entry rather than on
every iteration. A unified benchmark framework is available via `python -m bench`.

## Documentation

- [Tutorials](https://pavlo3p.github.io/SpaceCore/tutorials/index.html)
- [Design notes](https://pavlo3p.github.io/SpaceCore/design/index.html)
- [API reference](https://pavlo3p.github.io/SpaceCore/api/index.html)
- [Release notes](https://pavlo3p.github.io/SpaceCore/release_notes.html)

## Project Status

SpaceCore is experimental `0.4.x` software. The `0.4.0` release stabilizes the
typed linear-algebra core as a validated algebra of structured mathematical
objects — a public check-policy, the dtype/scalar-field contract, the
`TreeSpace` direct-product abstraction with block-structured operators, an
everyday functional and proximal toolbox, external optimizer adapters, reusable
test generators, and a backend conformance matrix. Core abstractions are usable
for research and prototyping, but API details may still change before a stable
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
