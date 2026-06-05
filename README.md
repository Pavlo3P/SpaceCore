# SpaceCore

[![CI](https://github.com/Pavlo3P/SpaceCore/actions/workflows/ci.yml/badge.svg)](https://github.com/Pavlo3P/SpaceCore/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/spacecore.svg)](https://pypi.org/project/spacecore/)
[![Python](https://img.shields.io/pypi/pyversions/spacecore.svg)](https://pypi.org/project/spacecore/)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)

**Backend-agnostic vector spaces, linear operators, and iterative solvers for scientific computing.**

Write your algorithm once. Run it on NumPy for development, JAX for GPU acceleration and autodiff, or PyTorch for ML pipelines — without changing a line.

```python
import spacecore as sc
import numpy as np

# Define a space, a linear operator, and solve Ax = b
ctx = sc.Context(sc.NumpyOps(), dtype=np.float64)
X = sc.DenseCoordinateSpace((100,), ctx)
A = sc.DenseLinOp(np.random.randn(100, 100) @ np.random.randn(100, 100).T + np.eye(100), X, X, ctx)
b = ctx.asarray(np.random.randn(100))

result = sc.cg(A, b, tol=1e-8)
print(f"Converged in {result.num_iters} iterations.")
```

Same code on JAX with GPU?

```python
ctx = sc.Context(sc.JaxOps(), dtype=jnp.float64)
# ... build A and b the same way using jax arrays ...
result = sc.cg(A, b, tol=1e-8)   # runs on GPU, JIT-compiled
```

## Install

```bash
pip install spacecore                # core (numpy only)
pip install "spacecore[jax]"         # add JAX backend
pip install "spacecore[torch]"       # add PyTorch backend
pip install "spacecore[jax,torch]"   # both
```

Python 3.11+. Built on the [Python Array API](https://data-apis.org/array-api/) standard.

## What is SpaceCore for?

SpaceCore is for people writing numerical algorithms — optimization, inverse problems, eigensolvers, quantum simulation, computational geometry — who don't want to choose between NumPy, JAX, and PyTorch.

### Three things SpaceCore does well

**1. Matrix-free linear operators with algebra.** Write your operator once as `apply` and `adjoint` callables, then compose them:

```python
# An FFT-based convolution operator, never materialized as a matrix
K = sc.MatrixFreeLinOp(apply=fft_convolve, rapply=fft_convolve_adjoint, dom=X, cod=X, ctx=ctx)
grad = sc.MatrixFreeLinOp(apply=finite_diff, rapply=neg_div, dom=X, cod=Y, ctx=ctx)

# Build the regularized system operator using algebra
lam = 0.01
system = K.H @ K + lam * grad.H @ grad     # SumLinOp of ComposedLinOps
rhs = K.H.apply(b)

# Solve — no matrices were assembled
solution = sc.cg(system, rhs).x
```

**2. Cross-backend iterative solvers.** CG, LSQR, Lanczos, power iteration — all work uniformly across NumPy, JAX, and PyTorch. JAX backends JIT-compile:

```python
ctx = sc.Context(sc.JaxOps(), dtype=jnp.complex128)
A = build_hermitian_operator(ctx)

# Find the smallest eigenpair via Lanczos
result = sc.lanczos_smallest(A, initial_vector, max_iter=50)
print(f"E_0 = {result.eigenvalue}, converged={result.converged}")
```

**3. Custom Hilbert spaces with non-Euclidean geometry.** Subclass `DenseCoordinateSpace`, override `inner`, and every solver respects your geometry:

```python
class WeightedL2(sc.DenseCoordinateSpace):
    def __init__(self, shape, weights, ctx=None):
        super().__init__(shape, ctx)
        self.weights = self.ctx.asarray(weights)

    def inner(self, x, y):
        return self.ops.vdot(x, self.weights * y)

# CG, LSQR, Lanczos all use this inner product automatically
```

This is the basis for RKHS spaces, truncated Fock spaces (quantum many-body), function spaces with quadrature, and anything else where the geometry isn't `sum(x * y)`.

## Quick examples

### Conjugate gradient on a symmetric positive-definite system

```python
import spacecore as sc

ctx = sc.Context(sc.NumpyOps(), dtype=np.float64)
X = sc.DenseCoordinateSpace((1000,), ctx)
A = sc.DenseLinOp(make_spd_matrix(), X, X, ctx)
b = ctx.asarray(rhs)

result = sc.cg(A, b, tol=1e-10, maxiter=500)
print(f"x = {result.x}, residual = {result.residual_norm}")
```

### Least-squares with regularization

```python
# min ||Ax - b||^2 + λ||x||^2  via normal equations
I = sc.IdentityLinOp(X)
system = A.H @ A + lam * I
rhs = A.H.apply(b)
x_hat = sc.cg(system, rhs).x
```

### Smallest eigenpair of a Hermitian operator

```python
result = sc.lanczos_smallest(A, initial_vector, max_iter=100)
print(f"E_0 ≈ {result.eigenvalue}")
print(f"Krylov dimension used: {result.krylov_dim}")
print(f"Converged: {result.converged}")
```

### Building a custom operator

```python
class Convolution(sc.LinOp):
    def __init__(self, kernel, space, ctx):
        super().__init__(space, space, ctx)
        self.kernel = kernel

    def apply(self, x):
        return self.ops.real(self.ops.fft.ifft(self.ops.fft.fft(x) * self.ops.fft.fft(self.kernel)))

    def rapply(self, y):
        return self.ops.real(self.ops.fft.ifft(self.ops.fft.fft(y) * self.ops.conj(self.ops.fft.fft(self.kernel))))
```

This operator works on NumPy, JAX, and PyTorch backends without modification.

## How is SpaceCore different from...?

**...`scipy.sparse.linalg`?** SciPy's iterative solvers are great but tied to NumPy/SciPy. SpaceCore gives you the same algorithms across NumPy, JAX, and PyTorch, plus operator algebra (`A @ B + lam * I` actually returns a usable operator), plus first-class custom Hilbert spaces.

**...PyLops?** PyLops is excellent for inverse problems but assumes Euclidean vectors and is tied to NumPy/CuPy. SpaceCore handles non-Euclidean geometry (RKHS, weighted spaces, function spaces) and works on JAX/PyTorch for autodiff and ML pipelines.

**...QuTiP?** QuTiP is the standard for quantum optics on top of SciPy. SpaceCore lets you build the same quantum operators on JAX or PyTorch for GPU acceleration and gradient-based parameter learning. Less prebuilt, more composable.

**...`array_api_compat`?** That package gives you portable arrays. SpaceCore builds on top of it to give you portable *vector spaces, linear operators, and iterative algorithms* — the abstractions one level up from arrays.

## Documentation

[//]: # (- **[Quick Start]&#40;https://pavlo3p.github.io/SpaceCore/quickstart.html&#41;** — 20-line introduction)
[//]: # (- **[Concepts]&#40;https://pavlo3p.github.io/SpaceCore/concepts.html&#41;** — Spaces, operators, contexts)
[//]: # (- **[Tutorials]&#40;https://pavlo3p.github.io/SpaceCore/tutorials/index.html&#41;** — Image deblurring, Jaynes-Cummings model, kernel ridge regression)
- **[API Reference](https://pavlo3p.github.io/SpaceCore/api/index.html)** — Full documentation

## Features at a glance

**Spaces.** `DenseCoordinateSpace`, `DenseVectorSpace`, `ElementwiseJordanSpace`, `EuclideanElementwiseJordanSpace`, `HermitianSpace`, `ProductSpace`, and `StackedSpace`. Generic dense spaces can use custom inner products; `DenseVectorSpace` has no Jordan capability by default; real Euclidean elementwise spaces get the Euclidean-Jordan capability.

**Linear operators.** `DenseLinOp`, `SparseLinOp`, `DiagonalLinOp`, `MatrixFreeLinOp`, plus operator algebra (`A @ B`, `A + B`, `2 * A`, `A.H`, `IdentityLinOp`, `ZeroLinOp`).

**Functionals.** `LinearFunctional`, `QuadraticForm`, with `value`, `grad`, `hess_apply`, and `compose(linop)` for pull-back.

**Iterative solvers.** `cg`, `lsqr`, `lanczos_smallest`, `power_iteration`.

**Backends.** NumPy (always), JAX (`spacecore[jax]`), PyTorch (`spacecore[torch]`), CuPy (`spacecore[cupy]`). Adding a backend is ~100 LOC; the registry is public.

## Project status

**v0.3 alpha.** API may still change in minor ways. Core abstractions are stable. Suitable for research code; not yet recommended for production deployment.

The library is being developed in the open and is looking for early users and feedback. If you try it on your problem, please open an issue with what worked and what didn't — that's the single most valuable contribution right now.

## Contributing

Bug reports, feature requests, and PRs welcome. See [CONTRIBUTING.md](CONTRIBUTING.md).

Specific areas where help is wanted:

- **Tutorials.** If SpaceCore solves your problem, a notebook example helps everyone.
- **Backends.** CuPy and Dask integration is partial; adding a new backend is well-scoped (~100 LOC).
- **Performance.** Cross-backend benchmarks on real workloads.
- **Documentation.** Concept pages, FAQ, gotchas.

## License

Apache 2.0. See [LICENSE](LICENSE).

## Citation

If SpaceCore is useful in your research, a citation is appreciated:

```bibtex
@software{spacecore,
  author = {Pavlo, Pelikh},
  title = {SpaceCore: Backend-agnostic vector spaces and linear operators},
  url = {https://github.com/Pavlo3P/SpaceCore},
  year = {2026},
}
