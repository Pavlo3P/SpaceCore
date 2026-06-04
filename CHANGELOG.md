# Changelog

All notable changes to SpaceCore are documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and the project adheres to [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Removed

- Removed the deprecated `Space.eigh` alias introduced by the prior spectrum
  migration release. Use `spectral_decompose` when both eigenvalues and a
  reconstruction frame are needed, or `spectrum` for eigenvalues only.

## [0.2.0]

SpaceCore 0.2.0 is a major API expansion. The backend layer now sits on the
Array API standard. Operators gained a lazy algebra with adjoint views,
composition, sums, and scaling. A new `Functional` hierarchy provides
scalar-valued maps with gradients and pull-backs. A new `spacecore.linalg`
module ships four JIT-compatible iterative solvers. Spaces, operators, and
functionals share a single validation pattern via `checked_method`, and the
public API is documented to numpydoc standard with doctest coverage.

This release introduces breaking changes; see [Migration](#migration-from-01x).

### Added

#### Backend

- Migrated `BackendOps` to the Array API standard via `array-api-compat`.
- `CuPyOps` and the `cupy` backend family as an optional install
  (`pip install 'spacecore[cupy]'`).
- `BackendOps.is_complex_dtype` for backend-aware complex detection.
- `BackendOps.real_dtype` for extracting the real dtype matching a complex one.
- Broadened backend coverage for array creation, dtype conversion, sparse
  conversion, indexing, reductions, linear algebra, loop primitives
  (`fori_loop`, `while_loop`, `cond`), tree helpers, and vectorized mapping.
- JAX pytree registration for operator, space, and functional types so they
  pass through `jax.jit`, `jax.vmap`, and `jax.grad` boundaries.

#### Context and checking

- Public free-function API in `spacecore._contextual`: `set_context`,
  `get_context`, `resolve_context_priority`, `register_ops`, and the
  resolution-policy accessors.
- Extended `checked_method` to support validation against `self` and multiple
  input argument positions.
- Reusable space-validation checks: backend, dtype, shape, Hermitian,
  square-matrix, product-structure, and product-component checks. Documented
  at `docs/source/design/checking_policy.rst`.

#### Spaces

- `BatchSpace` for batched elements with explicit batch shape and batch-axis
  metadata.

#### Linear operators

- Lazy operator algebra:
  - `A @ B` composes operators.
  - `A + B` sums operators.
  - `alpha * A` scales an operator.
  - `A.H` returns a cached adjoint view satisfying `A.H.H is A`.
  - Algebraic simplification eliminates `I`, `Zero`, `alpha = 0`, `alpha = 1`,
    and flattens nested sums.
- New operator types: `IdentityLinOp`, `ZeroLinOp`, `MatrixFreeLinOp`,
  `DiagonalLinOp`.
- Structural `LinOp.is_hermitian()` reporting `True`, `False`, or `None`
  (unknown) without applying incorrect Euclidean assumptions for custom space
  geometries.
- `LinOp.to_dense()` for materializing operators as backend arrays.
- Product-structured operators and batched lifting:
  - `ProductLinOp`
  - `BlockDiagonalLinOp`
  - `StackedLinOp`
  - `SumToSingleLinOp`
  - `vapply` / `rvapply` paths for batched operator application.

#### Functionals

- `Functional` as an abstract base for scalar-valued maps on spaces, with
  `value`, `grad`, `hess_apply`, and batched counterparts.
- Linear functionals: `LinearFunctional`, `InnerProductFunctional`,
  `MatrixFreeLinearFunctional`.
- Quadratic forms: `QuadraticForm`, `LinOpQuadraticForm`.
- `Functional.compose` and `ComposedFunctional` for pull-backs along linear
  operators, with specializations that preserve the concrete functional type
  when possible.

#### Linear algebra

The `spacecore.linalg` module is new in 0.2.0. It provides JIT-compatible
iterative solvers and structured result types.

- Iterative solvers:
  - `cg` for Hermitian positive-definite systems.
  - `lsqr` for rectangular least-squares problems.
  - `power_iteration` for dominant-eigenpair estimates of a `LinOp` or
    `QuadraticForm`.
  - `lanczos_smallest` for smallest-Ritz-eigenpair estimates of Hermitian
    operators.
  - `expm_multiply` for Krylov matrix-exponential actions `exp(t A) v` on
    Hermitian operators, with complex `t` supported for Schrodinger-type
    evolution.
- Structured result types `CGResult`, `LSQRResult`, `PowerIterationResult`,
  `LanczosResult`, and `ExpmMultiplyResult`, each carrying convergence
  diagnostics and a compact `__repr__`.
- Solvers are geometry-aware: norms, inner products, and the default initial
  vector use `Space.inner` and `Space.norm` rather than assuming Euclidean
  geometry. This makes the solvers correct on custom inner products such as
  RKHS or weighted spaces.

#### Documentation

- Numpydoc-standard public docstrings with runnable doctests for solvers,
  spaces, operators, functionals, backends, and contextual helpers.
- API reference pages for backend ops, spaces, linear operators, functionals,
  and linear algebra.
- JAX integration design note at `docs/source/design/jax_integration.rst`
  covering trace-time operator algebra and recommended JIT usage.
- Tutorials for backend operations, linear operators, and matrix-free linalg
  workflows.

#### Tooling

- Optional dependency groups: `[jax]`, `[torch]`, `[cupy]`, `[examples]`,
  `[docs]`, `[dev]`.
- Explicit `__all__` at the top level covering new backends, operators,
  functionals, solvers, result types, validation checks, and contextual
  helpers.
- CI runs a JIT-traceability audit in `--check` mode and enforces a 70%
  coverage floor via `pytest-cov`.
- Cross-backend tests covering NumPy, JAX, Torch, and optional CuPy.

### Changed

- Restructured `_contextual` to hide implementation details while preserving
  the public API via free functions.
- Replaced manual `if self._enable_checks` guards with `checked_method` across
  `Space`, `LinOp`, and `Functional`. Inline guards are now reserved for
  non-membership checks such as dense-array assertions and custom output-shape
  checks.
- Improved `VectorSpace`, `HermitianSpace`, and `ProductSpace` conversion
  behavior, validation, batching support, and docstrings.
- Improved linear-operator equality, representation, conversion, and JAX
  pytree behavior.
- `spacecore.__version__` now resolves from package metadata via
  `importlib.metadata` instead of a hand-maintained constant.
- Bumped the package version to `0.2.0`.

### Fixed

- `LinOp.__eq__` returns `NotImplemented` instead of raising
  `NotImplementedError` on the base class, so `op == None` and
  `op in some_list` no longer raise.
- `DenseLinOp.is_hermitian` and `SparseLinOp.is_hermitian` return `None` for
  custom space geometries instead of applying an incorrect Euclidean
  matrix-symmetry test.

### Migration from 0.1.x

- `BackendOps.eps` is now a method `eps(dtype)` rather than a property.
  Callers must pass a dtype, typically `ctx.dtype`.
- The implementation attribute `DenseLinOp.A` is now a `cached_property`
  backed by `_A`. The public attribute access `op.A` is unchanged.
- `LinOp.__eq__` returns `NotImplemented` rather than raising; downstream code
  relying on the exception should be updated to handle the new behavior.
- Several module-internal helpers in `spacecore._contextual` moved to private
  modules. Use the public functions re-exported from `spacecore._contextual`
  (`set_context`, `get_context`, `resolve_context_priority`, `register_ops`,
  `set_resolution_policy`, and the dtype-policy accessors) rather than
  importing from internal modules.

### Known limitations

- `cg`, `lsqr`, and `power_iteration` do not structurally validate operator
  properties (positive-definiteness, full Hermiticity) and may silently
  produce incorrect results on inputs that violate their preconditions. See
  each function's `Notes` section for details.
- Operator algebra runs Python-level simplification at construction time. For
  maximum JIT efficiency, assemble operator expressions outside the
  `jax.jit` boundary; see the JAX integration design note.
- `MatrixFreeLinOp` stores its callables in pytree auxiliary data.
  Constructing one inside a JIT-traced function with a new lambda each call
  triggers retracing. Construct outside the traced region with a stable
  callable reference.
- The CuPy backend is provided as a preview. Coverage of non-standard
  operations and sparse handling may evolve in a subsequent release.

[0.2.0]: https://github.com/Pavlo3P/SpaceCore/releases/tag/v0.2.0
