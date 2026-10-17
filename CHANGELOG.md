# Changelog

All notable changes to SpaceCore are documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and the project adheres to [Semantic Versioning](https://semver.org/).

## [0.5.0] — unreleased (date set at tagging)

### Changed

- **Cores are the extension point for every object** (breaking for subclasses). Each
  public operation of spaces, operators and functionals is a checked wrapper over a
  `_<op>_core`: `axpy`, `norm`, `riesz`, `riesz_inverse`, `flatten`, `unflatten`,
  `flatten_batch`, `unflatten_batch`, `add_batch`, `scale_batch`, `pair`, `vpair`,
  `star`, `jordan`, `spectrum`, `spectral_decompose`, `from_spectrum`,
  `spectral_apply`, `trace`, `determinant` on spaces; `grad`, `value_and_grad`,
  `derivative_at`, `value_and_derivative`, `differential`, `vgrad`, `prox` on
  functionals (joining `add`, `scale`, `inner`, `apply`, `rapply`, `value`, ...). A
  family implements the cores; `_star_core`, the Jordan cores and
  `_flatten_core`/`_unflatten_core` are the abstract methods. Composites call their
  children's cores, so a sum, product or operator checks once at its own boundary.
  Subclasses that counted or replaced a public batch method of a concrete space now
  override the core.
- An undecorated public override still works: `install_public_checks` (run from
  `ContextBound.__init_subclass__` for spaces, operators and functionals) installs
  it as the class's core and restores the base's checked method. An override that
  calls `super().<same name>` is re-checked in place instead.
- **Spaces over the reals reject a provably non-real multiplier** (breaking): `scale`,
  `axpy` and `scale_batch` raise `SpaceValidationError` on every real space, not only
  `HermitianSpace`; traced scalars pass.
- New checks at standard level: `unflatten` checks its output; `riesz`,
  `riesz_inverse` and `norm` check their input (an element or a leading-axis batch);
  `star` and the Jordan operations check inputs and outputs.
- `checked_method` finds a checked argument passed by keyword (`f.prox(x=..., step=...)`)
  and records the core it calls (`core=`).
- `CuPyOps` is `NumpyOps` with the CuPy/cupyx handles and `array_api_compat.cupy`
  (not yet run on a GPU). `TorchOps.xp` is a per-instance cached property.
- A broken optional backend (any import error, not only `ImportError`) warns and is
  skipped instead of breaking `import spacecore`.
- `TreeSpace` and `StackedSpace` build their capability subclasses from one list of
  capability sets; the classes keep their names (`TreeInnerProductSpace` is an alias
  of `_TreeInnerProductSpace`), and `_*EuclideanJordanStarSpace` no longer subclasses
  `_*EuclideanJordanAlgebraSpace`. Every stacked capability subclass checks its base.
- Every linalg solver returns one `LinalgResult(x, converged, num_iters, residual_norm,
  eigenvalue=None, normal_residual_norm=None)`, replacing `CGResult`, `LSQRResult`,
  `PowerIterationResult`, `LanczosResult` and `ExpmMultiplyResult`. Renamed fields:
  `eigenvector` and `result` are `x`, `krylov_dim` is `num_iters`, and
  `residual_estimate` is `residual_norm`.

### Added

- Spaces, operators and functionals are hashable, consistently with `==`: the hash
  covers type, backend family, dtype and shape (spaces) or endpoint spaces
  (mappings), never array values. `_repr_html_` renders the text repr in notebooks.
- `set_traced_check_level` / `get_traced_check_level`: a process-wide cap on check
  levels while a backend traces (`jax.jit`, `jax.vmap`, `torch.compile`), default
  `"none"`, so objects built at any level compile. `"cheap"` keeps shape and dtype
  checks, which run once at trace time. Backends report tracing through the new
  `BackendOps.is_tracing()`.
- `cg(..., preconditioner=K)`: preconditioned CG with an operator `K` on `A.domain`.
- `InnerProductSpace.dual_norm(xi)`, the norm of a covector.
- The metric-free pairing layer: `CoordinateSpace.pair(xi, x)` and `vpair(xi, xs)`
  pair a coordinate covector with elements without any inner product;
  `Functional.differential(x)` returns the covector of the derivative (natively for
  coordinate functionals).
- `OperatorInnerProduct(M, M_inverse=None)`: a metric given by a self-adjoint
  positive-definite operator on the Euclidean coordinates, with `M` as the Riesz
  map and an optional, possibly inexact, inverse. Block, Kronecker and matrix-free
  metrics work with the existing metric adjoints, and CG in such a space is
  preconditioned CG.
- Optional `Functional.prox` rules in the existing mapping table, Hilbert-space
  `conjugate_prox` via Moreau, positive scaling, and finite constant shifts.
  Rules cover the squared space norm, coordinate L1 in diagonal geometry, and
  indicators. Unsupported sums, products, pullbacks, and geometries raise
  `CapabilityError`; no optimization loop or dense fallback is introduced.
- `Set`, `NonnegativeOrthant`, and `IndicatorFunctional` connect membership,
  metric projection, and extended-real indicator values. Custom sets explicitly
  declare convexity before supplying an indicator.
- Acceptance tests run unchanged proximal gradient and PDHG across two metrics
  and two representations, including JAX compilation with dynamic steps, and
  check independent KKT conditions.

### Fixed

- Pickling a `TreeSpace` or `StackedSpace` (their `__new__` dispatches on arguments).
- `power_iteration` on a complex Hermitian operator under `jax.jit`/`lax.while_loop`:
  the eigenvalue carry is real, as the Rayleigh quotient is; `cg`'s breakdown
  tolerance is real too (ADR-015's real solver diagnostics).
- `StackedSpace` of a complex `ElementwiseJordanSpace` in a real context: the guard
  now checks the converted base.
- `torch.compile` with the traced check level at `"cheap"`: space check caches are
  not filled while tracing.
- The documentation build imports `spacecore` with `torch` mocked.
- `InnerProductSpace.geometry` is never `None`: it defaults to Euclidean, and a native
  space states a non-Euclidean metric as its own `InnerProduct` (inner product and Riesz
  maps on its native elements). `MatrixFreeLinOp` no longer warns about coordinate
  adjoints on such a space. `least_squares` keeps native members native.
- `Element` builds no dense form at `check_level="none"`, and `densify` is an
  optional capability: without it, fallback arithmetic raises `CapabilityError`
  instead of failing obscurely.
- Proximal helpers accept dynamic scalar steps at `check_level="none"` and use
  real thresholds for complex diagonal geometry. Thresholding works on Torch
  without passing Python scalars to its tensor-only `maximum` primitive.
- Default Krylov initialization uses native `ones()` when available and asks
  for an explicit initial element otherwise. Non-coordinate spaces require an
  explicit iteration budget, rather than failing with a missing `shape`.
- **Custom `InnerProduct` implementations must supply Riesz maps** (breaking).
  Missing maps raise `CapabilityError`; only `EuclideanInnerProduct` supplies
  identity maps by default, preventing silent misuse of a non-Euclidean metric.

- Backend array typing: native dense results for NumPy, JAX, and Torch operations
  and explicit contexts; portable protocols accept native positional operators;
  optional backend exports and forwarded operations are visible to editors.
  Validation decorators preserve callable signatures. Wheels include `py.typed`.
- Scalar results from elementwise backend operations, contractions, and
  NumPy/CuPy `logsumexp` are promoted to zero-dimensional arrays, matching the
  existing array-result contract.

### Changed

- **Space cores** (breaking for custom spaces): a vector space implements `_add_core` and
  `_scale_core`, and an inner-product space `_inner_core`, instead of the public `add`,
  `scale` and `inner`, which the base classes now own and check. The name-based core
  lookups in operator algebra, functionals and solvers are gone.

### Removed

- `spacecore.linalg._utils.SpaceCoreOps`, `resolve_core`, `resolve_apply` and
  `resolve_rapply`: solvers call the operators' and spaces' cores directly, and an
  overridden public method is already the core. `HermitianSpace.scale` and
  `scale_batch` (the base wrappers now carry the real-scalar guard).
- The batched-Riesz `vmap` fallbacks in `metric_rvapply` and `StackedSpace`: Riesz maps
  must broadcast over leading batch axes, as all built-in geometries do.
- **`ComplexField`, the `Field` base class and `Space.scalars`** (breaking). Nothing
  used them once functionals became real-valued; `RealField` is the one scalar
  field and absorbs `Field`'s behaviour. A space's `scalar_field` still declares
  real or complex coefficients.
- **`StackedLinOp` and `SumToSingleLinOp`** (breaking). `BlockMatrixLinOp` covers
  both: a side given as a plain one-element list or tuple is a single space, not a
  one-leaf tree, and the containers of rows and of each row are the codomain and
  domain trees (a NamedTuple of rows gives NamedTuple elements). Write
  `BlockMatrixLinOp.stack((A1, A2))` for `x -> (A1 x, A2 x)` and
  `BlockMatrixLinOp.from_operators((A1, A2))` for `(x1, x2) -> A1 x1 + A2 x2`; the
  adjoint of each is the other. A one-column or one-row `BlockMatrixLinOp` now has a
  plain endpoint where it used to have a one-leaf `TreeSpace`. The flat dense fast
  paths of the two classes went with them: they did the same matrix products as
  `DenseLinOp`'s core and saved only per-block call overhead.
- **Legacy `BlockDiagonalLinOp(dom, cod, parts, ctx)` construction** (breaking).
  `BlockDiagonalLinOp(blocks, ctx)` takes the block tree only; its tree is the element
  structure, so a NamedTuple of blocks gives NamedTuple elements. The tree operators'
  `_*_unchecked` methods are now the standard `_*_core` cores, and the
  `_apply_unchecked` fallback in `TreeLinOp` is gone.
- **`spacecore.optimize`** (breaking): `minimize_scipy`, `line_search_scipy`,
  `minimize_optax` and `OptaxResult` moved to the separate
  [`spacecore-optimize`](https://github.com/Pavlo3P/spacecore-optimize) package
  (`import spacecore_optimize as spo`), which depends on SpaceCore. The `optax`
  install extra, `docs/source/api/optimize.rst` and `tests/optimize/` go with
  them. SpaceCore keeps the geometry handoff they use (`Functional.grad`,
  `X.riesz`) and no optimization loops.
- **Functional catalogue** (breaking): `least_squares`, `NegativeEntropyFunctional`,
  `KLDivergenceFunctional`, `HuberFunctional`, `SpectralFunctional`, `spectralize`,
  `eigenvalue_space`, `NuclearNormFunctional`, `generalized_shrinkage`, `prox_l1`,
  `prox_l2sq`, `project_nonneg`, `RealifiedFunctional` and `realify` moved to
  `spacecore-optimize` (`spo.<Name>`).
  SpaceCore keeps the norms used by the prox rules (`LpNormFunctional`,
  `SquaredL2NormFunctional`, with the `L1NormFunctional` wrapper), the
  prox/conjugate machinery, indicators and sets, and the coordinate-gradient base
  `_CoordinateFunctional`.
- Notebooks, examples and the benchmark suite (`tutorials/`, `notebooks/`,
  `examples/`, `bench/`) move out of the package repository; the docs keep
  their tutorials pages (`docs/source/tutorials/`), and the
  `examples` install extra is gone.
- **The ADR-016 benchmarked-spec dispatch layer** (breaking): `spacecore.kernels.specs`,
  `dispatch`, `dispatch_mode`, `set_dispatch_mode`, `get_dispatch_mode`,
  `should_consult_dispatch`, `KernelSpec`, `KernelCost`, `KernelRegistry`, `registry`,
  the dispatch errors and the memory-budget knobs, together with the ADR-022
  `CachedStackParts` cache. Dispatch was off by default; its batched specs applied only
  to eager NumPy, where they ran mostly slower than the plain per-block loops, and
  never under `jax.jit`. The bench harness now runs the
  `baseline` regime only, and tutorial 09 covers fusion alone.
- **The `spacecore.kernels` package.** Each check-free core (`_apply_core`,
  `_rapply_core`, `_value_core`, ...) is now a method on the class it served, so the
  duck-typed free functions, their binding lines and the lazy imports that broke
  the `linop` ↔ `kernels` cycle are gone. The fast paths are unchanged.
- **`spacecore.opfamily`** (breaking): `OperatorFamily`, `FunctionalScaledOperator`
  and `make_functional_scaled_operator` are gone, and `Functional * LinOp` raises
  `TypeError`. Nothing used the functional-weighted map; write `F(x) * A` for a
  frozen member.
- **The `enable_checks` boolean shim** (breaking): `check_level` accepts only
  `"none"`/`"cheap"`/`"standard"`/`"strict"` or `None`, and the
  `normalize_check_level(enable_checks=...)`, `level_to_enabled`,
  `enabled_to_level` and `ContextBound._enable_checks` bridges are gone.
- **`checked_method(arg_pos=...)`** (breaking); use `arg_positions`.
- **`enforce_convert_policy`** (breaking); `ContextBound.convert` normalizes the
  target context itself.
- **Entry-point backend discovery** (breaking): `discover_entry_point_ops` and the
  `spacecore.backends` entry-point group. `available_ops` lists NumPy and the
  importable built-in backends.
- **`minimize_optax(progress_callback=...)`** (breaking). It only replayed the
  recorded history after the loop; read `result.history` instead.
- **`power_iteration(check_every=...)`** (breaking); it was already ignored.
- **`ShiftedFunctional`** (breaking). `F + c` is now the sum of `F` and a
  `ConstantFunctional`; a sum skips its constant terms' zero gradients, so the
  shift costs no extra gradient add.
- **`BackendOps.free_memory_bytes`** (breaking) and the `psutil` dependency. Only
  the removed dispatcher read the memory budget.
- **The core-kernel registry** (breaking): `CoreKernelSet`, `core_kernels`,
  `register_core_kernels`, `get_core_kernels` and `core_kernel_names`. Operator
  classes bind their cores as class attributes (`_apply_core = dense_apply_core`).

### Changed

- **Checks follow the check level throughout geometry and solvers.** `WeightedInnerProduct`
  reads weight values (real, finite, positive) only from `standard` upward, so building a
  space costs nothing at `none` and traced weights work under `jit`; weights may be any
  shape that broadcasts to the space (a 0-d array scales the Euclidean metric), and two
  weightings of one metric compare equal. `cg`, `lanczos_smallest`, `expm_multiply` and
  `power_iteration` probe self-adjointness only from `standard` upward.
- **Self-adjointness is decided structurally where possible.** Adjoint-palindromic
  compositions (`R.H @ R`, `L.H @ V @ L` with `V` self-adjoint), square block matrices with
  `A_ij# == A_ji` and self-adjoint diagonal blocks, and diagonal operators on a weighted
  space answer `is_hermitian()` without probes. `MatrixFreeLinOp(..., hermitian=True)`
  declares it, probed once at `strict`.
- **`OperatorInnerProduct`.** Without `M_inverse` it reports no Riesz maps, so matrix-backed
  operators refuse the space at construction instead of failing at `rapply`. At `strict`,
  positivity is relative (smallest eigenvalue estimate above `1e-12` times the largest) and a
  supplied `M_inverse` must invert `M` on a probe vector to `1e-6`.
- **Functionals are strictly real-valued** (breaking). Their codomain is always
  `RealField`, even on complex domains; the `cod` and `real_linear` arguments and the
  functional `scalar_field` property are gone. With checks on (`cheap` and above),
  complex outputs, coefficients and offsets raise `TypeError` through `RealField`
  membership, including in batched and fused evaluations and subclass overrides.
  Functionals pair through the new `InnerProductSpace.real_inner`
  (`Re<x, y>`), which the space norm and the solvers also use; no functional
  applies a real part of its own.

- **`LinOp` and `Functional` own their checked public methods.** `apply`, `rapply`,
  `vapply`, `rvapply`, `value` and `vvalue` validate on the base class and call the
  check-free cores (`_apply_core`, `_value_core`, ...), with the vmap fallback in the
  default batched cores. A subclass implements the cores; overriding a public method
  still works. See the new "Extending SpaceCore" design note.
- The map algebra moved from `spacecore/mapping.py` into the `spacecore.mapping`
  package; `from spacecore.mapping import ...` is unchanged.
- **Field declarations replace hand-written pytree, convert and equality hooks.** A
  class lists its constructor fields once (`_fields`, with `_dynamic` naming the
  children); `PyTreeNode` derives `tree_flatten`/`tree_unflatten`, `ContextBound`
  derives `_convert`, and `Mapping` derives `__eq__`. Coordinate functionals that
  compared by identity (`SquaredL2NormFunctional`, `LpNormFunctional`, `HuberFunctional`,
  the entropy functionals, `IndicatorFunctional`) now compare by their fields. A
  `PyTreeNode` that neither declares fields nor flattens itself refuses construction.
  The pytree `aux` of these classes is now a tuple of `(field, value)` pairs.
- Lanczos and `expm_multiply` now retain native elements through a private basis
  adapter and space algebra, with dense fast paths and fixed-shape JAX pytree
  support. No coordinate conversion is required for custom spaces.
  **Lanczos zero-start behavior changed**: a zero initial element raises eagerly;
  under JAX tracing it returns a NaN eigenvalue and `converged=False` instead of
  inventing a coordinate vector. The exponential action of zero returns zero.

- The `Scaled`/`Sum`/`Zero` lazy nodes and the `make_sum`/`make_scaled` factories are
  written once in `spacecore.mapping` and mixed into the `LinOp` and `Functional` node
  classes, which keep only evaluation. A functional sum now also requires its terms to
  share a codomain field, as an operator sum already did.
- Functionals now expose a `Field` codomain, defaulting to `domain.scalars`,
  and validate scalar output through ordinary space membership. Explicit field
  codomains can differ from the domain's coefficient field; exact batched
  scalar checks remain unchanged.
- `LinOp` rejects a `Field` codomain at construction, preserving the separate
  functional and Riesz-gradient contract from ADR-010.
- **Spaces reject integer/Boolean storage and complex coefficients over real
  storage at construction** (breaking). The guard lives on `Space.__init__`, so
  it applies to every space rather than only the field-backed numerical ones. A
  space over an integer or Boolean dtype now raises; represent such data in a
  floating dtype, or model it outside the vector-space hierarchy.

- **`LinOp` endpoints are bound to `VectorSpace`** rather than `CoordinateSpace`.
  A linear operator needs only `zeros`/`add`/`scale` from its endpoints;
  coordinates are a representation, not part of the contract. Operators can now
  be defined over any vector space, including one with no `shape`, `size`,
  `flatten` or `stacked`. Concrete operators are unaffected — `SparseLinOp` and
  `DiagonalLinOp` declare `LinOp[CoordinateSpace, CoordinateSpace]` themselves.

  `VectorSpace` and not `InnerProductSpace`: the two are siblings, so binding to
  the latter would have *narrowed* the accepted set and dropped bare `TreeSpace`
  and `StackedSpace`.

- **Operations that need more than a vector space now say so.** Materialization
  (`to_dense`, `to_matrix`, `fuse`, basis probes) requires coordinates, batching
  (`vapply`, `rvapply`) requires the batched coordinate surface, and `rapply`
  requires an inner product for its metric-adjoint contract. Each raises
  `CapabilityError` naming the operation and the missing capability, where
  previously an unsupported endpoint surfaced as an `AttributeError`.

- `MatrixFreeLinOp`'s strict adjoint-consistency probe no longer requires
  coordinates. It builds its probe element from the space's own `ones` when one
  exists, so a non-coordinate space is checked too, and is skipped only when no
  non-zero element can be built at all.

- Operator endpoint compatibility no longer compares `shape`. Type identity,
  context agreement and convert-equality already decide it, and
  `CoordinateSpace` equality compares shape.

### Added

- `Element`, a formal member of a space: context-bound, a pytree, checked
  against its space once at construction, with arithmetic that stays in the
  space. Representations are subclasses — `DenseElement` and the factored
  Hermitian `LowRankElement` — and a pair without its own rule densifies. `A @ x`
  applies an operator to an element.
- `Mapping`, the common parent of `LinOp` and `Functional`. It owns the operator
  overloads: each family registers `add`, `mul` and `matmul` rules, and
  subtraction, negation and `sum()` are derived. `F @ A` now composes a
  functional with an operator, and `A + 0` returns `A`.
- Public `Field`, `RealField`, and `ComplexField` scalar spaces with Euclidean
  geometry, and `Space.scalars` for accessing the coefficient field on the
  same backend. Real coefficients over complex storage use the corresponding
  real dtype without changing the owning space's context.

- `CapabilityError`, exported from `spacecore` and `spacecore.space`, raised when
  a space lacks a capability the requested operation needs.

## [0.4.3] — 2026-09-12

### Removed

- **`jax_pytree_class` is removed from the public API** (breaking). Pytree
  registration is no longer a JAX-named decorator applied by hand, but a
  backend-neutral registry: containers inherit `PyTreeNode` and each backend
  installs its own tree protocol (see *Added*). No replacement decorator is
  exported — a class becomes a pytree node by being a concrete `LinOp`,
  `Functional`, or one of the three pytree `Space` types. Code that decorated its
  own class should inherit `spacecore.backend.PyTreeNode` instead.
- **`SpectralLpNormFunctional` is removed** (breaking). A per-formula spectral
  class duplicates the value, the gradient, the pytree methods and the validation
  of its coordinate twin — it re-checked `p >= 1` that `LpNormFunctional` already
  enforces. The Schatten `p`-norm is now the spectral *lift* of the coordinate
  `p`-norm:

  ```python
  # before
  f = sc.SpectralLpNormFunctional(X, p)
  # after
  f = sc.spectralize(X, lambda s: sc.LpNormFunctional(s, p))
  ```

  `NuclearNormFunctional(X)` is unchanged as a named constructor and now returns a
  `SpectralFunctional` whose `base` is `LpNormFunctional(s, 1.0)`; code reading
  `f.p` should read `f.base.p`.

### Added

- **Backend-neutral pytree registration.** `PyTreeNode` (the capability mixin
  owning the `tree_flatten` / `tree_unflatten` contract), `PyTreeRegistry`, and
  `BackendOps.install_pytree_protocol()` — a plugin hook, no-op by default, that a
  backend implements to teach its transform machinery to see inside SpaceCore
  containers. The registry wires the class × backend cross-product with
  back-fill in both directions, so registration no longer depends on import order.
  `LinOp` and `Functional` now carry the capability on their bases, so every
  concrete subclass — including user-defined ones — is registered automatically.
- **Torch is now a transform-capable backend.** `TorchOps.install_pytree_protocol`
  registers containers with `torch.utils._pytree`, so a Torch-backed operator can
  flow through `torch.compile` and `torch.func` as a traced argument rather than an
  opaque leaf. One registration also covers `torch.utils._cxx_pytree`.
- **`SpectralFunctional`, `spectralize`, `eigenvalue_space`** — lift **any**
  symmetric coordinate functional onto a Jordan spectrum (Lewis's theorem), instead
  of hand-writing a class per formula. `spectralize(X, NegativeEntropyFunctional)`
  is the von Neumann entropy; `SquaredL2NormFunctional` lifts to the squared
  Frobenius norm, `HuberFunctional` to its spectral analogue. The base functional
  must be symmetric — eigenvalues have no canonical order, so a non-symmetric `f`
  makes `f(lambda(X))` ill-defined rather than merely mis-differentiated.
- **`RealifiedFunctional` and `realify`** — view a complex-domain functional over
  stacked real coordinates `(Re v, Im v)`, for optimizers that assume a real vector
  space. The real gradient is `(Re g_v, Im g_v)` with `g_v` the *coordinate*
  gradient, so the metric correction stays where ADR-010 put it. `realify` is a
  no-op on an already-real domain.
- **`OpsRegistry`** — the backend registry extracted out of `Contextual`, which now
  holds ambient policy only. Instantiable, so registration is testable without the
  process-wide singleton; `register_ops` still raises `ContextConflictError` on a
  duplicate family.
- **`BackendOps.complex_dtype`** — the inverse of `real_dtype`. Deriving it at the
  call site is not portable: NumPy promotes `float32` against a Python complex to
  `complex128`, while JAX and Torch give `complex64`.

- **`ProductFunctional` and `make_functional_product`** — the functional algebra
  becomes multiplicative. `F * G` is the pointwise product `F(x) * G(x)` on a
  shared domain, with the product-rule Riesz gradient

  ```
  grad(F·G)(x) = conj(G(x))·grad F(x) + conj(F(x))·grad G(x)
  ```

  combined through the domain's own `scale`/`add` (a domain element may be a
  pytree), and a `value_and_grad` that evaluates each factor once. Deliberately
  **binary**: the product rule is a two-factor law, and `(F*G)*H` expresses the
  n-ary case at the same cost. The conjugations are identities for the usual
  real-valued factors.
- **`ConstantFunctional` and `make_constant_functional`** — the constant map
  `x -> c`, with zero gradient. This is the embedding of a scalar into the
  functional algebra, previously unrepresentable: the algebra had a zero element
  and an affine shift but no constant node. `make_constant_functional` collapses
  `c = 0` to `ZeroFunctional` so the additive identity keeps one representation.
- **`spacecore.opfamily`: `OperatorFamily`, `FunctionalScaledOperator`,
  `make_functional_scaled_operator`** — `F * A` for a `Functional` and a `LinOp`
  is the functional-weighted map `m(x) = F(x) A x`. That map is **not linear**
  (both the scale and the direction move with `x`), so it is deliberately *not* a
  `LinOp`; it is a point-indexed *family* `x -> A_x`, of which a `LinOp` is the
  constant case. Freezing the point recovers linearity, and each point carries
  two different operators:

  - `m.at(x)` — the frozen member `F(x) · A`, an ordinary `ScaledLinOp` that
    composes, sums and has an adjoint;
  - `m.linearize_at(x)` — the derivative `Dm(x)[h] = <grad F(x), h>·Ax + F(x)·A h`,
    with metric adjoint `<Ax, w>_Y·grad F(x) + conj(F(x))·A^# w`, for Newton-type
    steps.

  They differ by exactly a rank-one term and coincide only when `F` is constant.
  A `ConstantFunctional` weight collapses to a plain `ScaledLinOp`, so the linear
  case is never forced through the non-linear type. The module is top-level
  because it depends on both `linop` and `functional`, and neither depends on it.
- **`checked_method(out_scalar=True)` / `out_batched_scalar=True`** — the codomain
  check for a `Functional`. `out_space=` names an attribute holding a `Space`, and
  a functional's codomain is the scalar *field*, reported only as a string, so the
  decorator that guards every `LinOp` output had nothing to bind to and the check
  was hand-written per subclass. `out_scalar` asserts `shape == ()`;
  `out_batched_scalar` asserts `(N,)` with `N` read from the input named by
  `in_space`, which it therefore requires.
- **`Space.scalar_field`, `Space.declared_scalar_field`, `Space.check_scalar`** —
  the field of scalars a space is closed under, *declared* rather than inferred
  from the dtype. It defaults to `field`; `HermitianSpace` declares `"real"`,
  because complex Hermitian matrices have complex entries but form a **real**
  vector space (`i·H` is anti-Hermitian).

### Changed

- **`check_level` is a property of the bound object, not of `Context`** (breaking).
  `Context(ops, dtype=..., check_level=...)` and the deprecated `enable_checks=`
  argument are gone; `Context` is now exactly `(ops, dtype)`. Pass `check_level=`
  to the space / operator / functional constructor, or set it ambiently with
  `set_check_level` / `use_check_level`. Two contexts differing only in strictness
  are now correctly the same context.
- **`spacecore._contextual` is now `spacecore.contextual`** (breaking), and
  `Context` moved out of `spacecore.backend`. The top-level `spacecore.Context`
  re-export is unchanged.
- **Ambient context and check level are scoped with `contextvars`.**
  `use_context` / `use_check_level` install a `ContextVar` override unwound by
  `Token`, so nesting is exact and concurrent threads or async tasks cannot clobber
  one another. `set_context` / `set_check_level` still write the process-wide
  baseline. The backend registry deliberately stays global — scoping it would make
  a backend registered inside a `with` block vanish on exit.
- **`available_ops()` is memoized and returns a tuple** rather than a list.
  Discovery attempts a real import per optional backend and scans entry-point
  metadata; it ran three times per `import spacecore`. A tuple because the cached
  value is shared.

- **`Functional.__mul__` / `__rmul__` now dispatch on the operand type.** A scalar
  still gives `ScaledFunctional`; a `Functional` now gives the pointwise product
  and a `LinOp` the functional-weighted family. Previously both returned
  `NotImplemented`, so `F * G` and `F * A` raised `TypeError`. Operands that are
  neither scalar-like nor `Functional` nor `LinOp` still defer to the reflected
  operation.
- **Functional outputs are checked as scalars at `standard` and above.** 17
  `value` and 3 `vvalue` implementations carry the new decorator flags, replacing
  four hand-written `_checks_at_least("standard")` / `_check_scalar_shape` bodies
  with one implementation. The level matches what those call sites already used —
  `cheap` deliberately does not run it. Cost is ~0.5 µs per `value` call, and
  nothing at `check_level="none"`, which still short-circuits before any check.
  `MatrixFreeLinearFunctional.vvalue` keeps its own check: it permits several
  leading batch axes, which is broader than the single-axis contract `vvalue`
  documents. `RealifiedFunctional.value` is checked on its output only — its input
  is validated against the *complex* domain by the inner functional.
- **`_check_scalar_shape` distinguishes single from batched output** in its error
  message ("Expected scalar output" vs "Expected scalar batch output"). It said
  "batch" unconditionally, which was harmless while it guarded four batched paths
  and misleading now that it guards ~20 mostly single-element ones.
- **`HermitianSpace.scale` / `scale_batch` reject a non-real multiplier**
  (breaking, at `standard` and above). `scale(1j, H)` returned a skew-Hermitian
  array still typed as an element of `Herm(n)`; the error surfaced at some later
  membership check, or never at `check_level="none"`. Only a *provably* non-real
  multiplier is refused, so a traced scalar under `jax.jit` still passes. `field`
  remains dtype-derived and still drives equality and repr: a real-dtype and a
  complex-dtype `Herm(n)` are genuinely different spaces.

### Fixed

- **A present-but-broken optional backend no longer aborts `import spacecore`.**
  `spacecore.backend` eagerly imported the JAX subpackage to reach
  `jax_pytree_class`, so an installed-but-unimportable backend raising anything
  other than `ModuleNotFoundError` — a shadowed `cupy`, a partially-installed
  `jax` — propagated out of the import. Discovery is now the only path into a
  backend package, and it warns and skips instead.
- **One broken backend now warns once, not once per discovery call.**

- **`ComposedFunctional` now has a gradient.** `F.compose(A).grad(x)` raised
  `NotImplementedError`: the node implemented `value` but not the chain rule,
  though `LinOp.rapply` already provides the metric adjoint it needs. Added
  `grad`, a fused `value_and_grad`, and a batched `vgrad`, with cores registered
  in the `composed-functional` kernel set:

  ```
  grad(F o A)(x) = A^#(grad F(A x))
  ```

  `rapply` **is** `A^#` (ADR-009), so no Riesz map is applied on top of it —
  that would count the geometry twice — and no explicit conjugation appears,
  because the adjoint identity absorbs it. `value_and_grad` applies `A` once and
  shares the image, where the inherited default applied it twice. The typed
  specializations in `make_functional_composed` (`InnerProductFunctional`,
  `LinOpQuadraticForm`) were unaffected, and are used as a cross-check on the
  generic node.
- **`scalar_eq` no longer swallows every exception.** It wrapped its comparison in
  a bare `except Exception: return False`, so a raising `__eq__` was silently
  reported as inequality. Narrowed to `TypeError` — the base class of JAX's
  `TracerBoolConversionError` and of any "cannot reduce to a concrete bool"
  failure — which keeps the intended verdict for an abstract scalar (undecidable,
  so canonicalization is skipped and the expression tree stays unfolded but
  correct) while letting a genuinely broken `__eq__` propagate.

## [0.4.2] — 2026-07-01

### Added

- **Jordan spectral primitives** on every Jordan-algebra space: `trace`,
  `determinant`, and the algebra identity `unit()`, derived from the spectrum and
  batch-axis preserving. Hermitian `trace` reads the diagonal (no eigendecomposition);
  tree/stacked spaces are direct sums, so `trace` is additive and `determinant`
  multiplicative (#56).
- **Lazy functional algebra** mirroring `LinOp`: `SumFunctional`, `ScaledFunctional`,
  `ShiftedFunctional` (affine `F + c`), and `ZeroFunctional`, wired to
  `Functional.__add__` / `__sub__` / `__neg__` / `__mul__` / `__rmul__` and the
  `make_functional_sum` / `make_scaled_functional` / `make_shifted_functional`
  factories. Gradients combine through the domain's `add`/`scale` (metric-correct on
  tree/stacked/weighted domains), and a complex scalar conjugates on the gradient
  side (#60).
- **`Functional.value_and_grad`** — a fused evaluator (base default returns
  `(value, grad)`; subclasses may override with a single-pass AD) — and
  `value`/`grad` now accept auxiliary `*args, **kwargs` (#51).
- **Structure-preserving tree spectra**: `TreeSpace.spectrum(x, structured=True)`
  returns a `treedef`-shaped tree of per-leaf eigenvalues, and
  `TreeSpectralDecomposition` carries the `treedef` with a `to_tree()` view. The
  flat `spectrum(x)` is unchanged (#58).

### Changed

- **`minimize_optax` is now a compiled, convergence-aware driver** (breaking): the
  whole loop runs inside `jax.jit(jax.lax.while_loop(...))` with the fused
  `value_and_grad` cached once per iteration; on-device `grad_norm <= tol` and
  finiteness stopping; four-column progress logging (iteration · value · ΔF ·
  grad norm) + a preallocated history buffer with post-hoc `progress_callback`
  replay; an optional `project` retraction hook; and a rich `OptaxResult`
  (success/status/message/iters/`nfev`/`njev`/`n_linesearch_steps`/history/timing/
  `x_element`). The old fixed-`steps` eager loop is removed (#54).

## [0.4.1] — 2026-06-28

### Added

- The **`bench` benchmark submodule** is brought under version control and
  aligned to the agreed 0.4.1 micro-surface (ADR-023,
  `docs/dev/0.4.1-bench-surface.md`): probes measure per-call SpaceCore overhead
  against a hand-optimal pure-array-library bare, across `space` / `linop` /
  `functional` only (linalg, the synthetic kernel-comparison probes,
  `check_member`, tree-space, and generated-linop probes removed). New
  **configuration-regime axis** (`baseline` / `dispatch` / `dispatch_cache` /
  `verify`) times the same probe under the ADR-016 dispatch and ADR-022 caching
  toggles, recording a within-run `regime_speedup`; block operators are measured
  uniform **and** ragged. The interactive HTML dashboard gains a problem-size
  range slider, a backend/family/status/size/check/search/speedup filter set
  that drives a fully **filter-reactive** summary and diagnosis section, a
  bottom-of-page tag legend, and zero-count-status hiding. Every backend runs in
  float64 for a fair comparison against the NumPy reference (JAX via
  `enable_jax_x64`, Torch via `enable_torch_x64`); the sole exception is Apple
  MPS (float32-only hardware), where the device probe builds a float32 case and
  the correctness gate widens accordingly. The float64-aware tolerance is shared
  by the verdict and diagnosis layers. **JAX is benchmarked only at
  `check_level="none"`** and **jits both the SpaceCore call and the bare
  reference**, comparing their post-compile steady state (the warmup absorbs
  compilation, so the speedup is jitted-vs-jitted with compilation excluded);
  each side's compile latency is recorded and reported separately (two
  `sc compile` / `bare compile` columns). The pair is resolved **symmetrically** —
  if either side is not jittable, both are timed eagerly, so a comparison is never
  eager-vs-jitted. Tooling only — `spacecore` never imports `bench`.
- ADR-016 optimized-kernel **dispatch** is accepted and implemented (off by
  default). `KernelSpec` gains `dispatch_key`, `priority`, and an optional
  shape-only `cost` estimator (`KernelCost`); a spec that names a `dispatch_key`
  with `rtol == atol == 0` is *dispatch-eligible*. `KernelRegistry` indexes the
  eligible specs by key (descending `priority`) and rejects two eligible specs
  that share a key at equal priority (`DispatchAmbiguityError`). A single
  `spacecore.kernels.dispatch(key, *args, generic=…, ctx=…)` entry point selects
  the first applicable, affordable spec or runs the inline `generic` fallback.
  `dispatch_mode` (`off`/`on`/`verify`) is settable process-globally
  (`set_dispatch_mode`) and per-scope (`dispatch_mode(...)` context manager);
  `check_level="strict"` implies `verify`. A materializing fast path is gated by
  a memory budget computed from `BackendOps.free_memory_bytes()` and
  `set_memory_budget_fraction` — no estimate or no budget means no fuse. The
  composed apply chain (`linop.composed.apply`) and block-diagonal apply
  (`linop.block_diagonal.apply`) call sites are wired through `dispatch`; with
  dispatch off they are result-identical to the prior inline paths. The two
  `0.4.0` catalog specs stay explicit-entry (no `dispatch_key`); activating a
  routed spec under either key is a benchmark-gated follow-up.
- Three dispatch-eligible algebraic-optimization kernels (exact, `rtol=atol=0`):
  `composed-zero-annihilation` (a chain containing a zero map collapses to
  `codomain.zeros()`) and `composed-identity-elision` (skip identity leaves)
  under `linop.composed.apply`, and `block-diagonal-uniform-dense-batched`
  (uniform flat-dense blocks fold into one batched `matmul`) under
  `linop.block_diagonal.apply`. The block kernel is *materializing*: it carries a
  shape-only `KernelCost` and the dispatcher gates it on the memory budget. All
  three route only under `dispatch_mode("on"/"verify")`; dispatch stays off by
  default. Each has a correctness reference and a `python -m bench` probe.
- `NumpyOps.free_memory_bytes()` reports available system RAM via `psutil` (now a
  required core dependency), so the dispatcher's memory gate can size
  materializing fast paths on the CPU backend.
- Five more dispatch-eligible batched-matmul kernels (exact, `rtol=atol=0`,
  NumPy-only until cross-backend bit-exactness is verified) extend the catalog to
  the adjoint and batched directions, each wired through `dispatch` at its own new
  key with a shape-only `KernelCost` and a `Class::method` correctness reference:
  `block-diagonal-uniform-dense-batched-rapply` / `-vapply` / `-rvapply` (under
  `linop.block_diagonal.rapply` / `.vapply` / `.rvapply`) and the broadcast-no-sum
  folds `stacked-uniform-dense-batched-apply` /
  `sum-to-single-uniform-dense-batched-rapply` (under `linop.stacked.apply` /
  `linop.sum_to_single.rapply`). The adjoint folds exploit the Euclidean-flat
  adjoint and so guard on `EUCLIDEAN_FLAT`; the batched folds use the dense core's
  transpose-on-right orientation. All five reuse one shared
  `spacecore.kernels.specs._batched` helper (stack uniform flat-dense matrices into
  a single batched `matmul`) instead of duplicating the fold. The
  `BlockDiagonalLinOp` (`rapply`/`vapply`/`rvapply`), `StackedLinOp` (`apply`) and
  `SumToSingleLinOp` (`rapply`) call sites are wired through `dispatch`;
  dispatch-off is result-identical to the prior inline paths. Off by default.
- `SparseLinOp` is confirmed to need no dispatch spec: every direction is already a
  single optimal backend call — one SpMV for `apply`/`rapply` and one batched SpMV
  over the stacked right-hand side for `vapply`/`rvapply` — so the reserved
  `linop.matvec.sparse` key stays inert (a spec would add only dispatch overhead).
- ADR-021 **lazy-operator-algebra `fuse()`** (Tier-2 explicit simplification):
  `LinOp.fuse(*, materialize=False)` collapses each maximal subtree of dense
  operators into a single materialized `DenseLinOp` — a composition becomes the
  matrix product `M_A @ M_B`, a scalar folds into the matrix, a sum of dense terms
  is added into one matrix, an adjoint fuses its operand, and block/tree operators
  fuse each component (so a composed-dense block becomes foldable by the
  block-diagonal dispatch spec). It is mathematically equal to the original up to
  floating-point rounding (fusion reassociates the arithmetic) and is
  adjoint-consistent on any geometry — the shared middle-space Riesz maps cancel,
  so fusion is not restricted to Euclidean spaces. A matrix-free operand is
  **never** densified by the default `fuse()`; `fuse(materialize=True)` is the
  explicit opt-in that densifies it (via the `to_dense` basis probe) so an
  enclosing expression can collapse. Lives in `spacecore.linop`, separate from the
  ADR-016 dispatch layer.
- ADR-022 **materialized-form cache** for the uniform batched folds: the
  input-independent stacked block-matrix array `ops.stack([matrix(p) for p in
  parts])` is now built once and reused across applies instead of re-stacked every
  call. `BlockDiagonalLinOp`, `StackedLinOp`, and `SumToSingleLinOp` carry their
  `parts` in a `spacecore.kernels.CachedStackParts` tuple that memoizes the stack
  per matrix accessor (`_A2` apply, `_A2H` adjoint, `_A2T`/`_A2H.T` batched), so a
  routed fold pays one `matmul` with no re-stack. The memo is built lazily on first
  optimized use (NumPy-only, and only while dispatch is `on`/`verify`), so the
  default `off` path is untouched. It is a derived value: excluded from operator
  identity (`__eq__`/`__hash__`) and from the pytree (`tree_flatten` re-normalizes
  `parts` to a plain tuple, so a round-trip rebuilds an empty cache), and a
  matrix-free operand is never cache-materialized (the fold is inapplicable). The
  dispatcher stays stateless. Dispatch-decision caching remains out of scope
  (ADR-022 §"Decision caching").

### Fixed

- Corrected stale `correctness_ref` node ids on two shipped `KernelSpec`s
  (`composed-chain-apply`, `block-diagonal-dense-apply`) that named non-existent
  module-level test nodes instead of the real `Class::method` ids (ADR-016
  requires the reference pin an existing test).

## [0.4.0] — 2026-06-24

SpaceCore 0.4.0 stabilizes the typed linear-algebra core as a validated
algebra of structured mathematical objects. It ships a public check-policy,
ADR-015 Stage 1 dtype/field contract, the `TreeSpace` finite direct-product
abstraction, block-structured LinOps on tree domains, an everyday functional
and proximal toolbox, external optimizer adapters, reusable test generators,
and a full backend conformance matrix with deviation catalog.

### Added

- ADR-018 **external optimizer adapters** in the new `spacecore.optimize`
  subpackage: `minimize_scipy(F, x0, *, method="L-BFGS-B", jac=True, **kw)` and
  `line_search_scipy(F, x, d, **kw)` drive `scipy.optimize`, and
  `minimize_optax(F, x0, optimizer, *, steps, callback=None)` runs the canonical
  optax update loop with pytree pass-through. Each adapter evaluates the
  objective through `F.value` and converts the metric (Riesz) gradient `F.grad`
  to the coordinate gradient external optimizers expect with
  `X.riesz(F.grad(x))` — the identity on a Euclidean space and mandatory on a
  weighted one, defusing the study's silent metric-gradient trap once and
  centrally. The external optimizer owns iteration, line search, and
  convergence; the SciPy adapters reject complex domains and document the
  structure/geometry/field lost at the boundary. `minimize_scipy` returns the
  SciPy `OptimizeResult` with an added `x_element` field (the minimizer
  unflattened into `F.domain`); `minimize_optax` requires a JAX-backed domain and
  the optional `optax` extra (`pip install spacecore[optax]`). Per ADR-018,
  jaxopt/BlackJAX/pymanopt seams stay as tutorials, not shipped adapters.
- ADR-019 everyday functional and proximal toolbox in the new
  `spacecore.functional.tools` subpackage (re-exported from
  `spacecore.functional` and the top-level `spacecore` namespace): named
  constructors over the existing `Functional` machinery, with no new core type
  hierarchy. `least_squares(A, b, *, weights=None, scale=0.5)` returns a
  `LinOpQuadraticForm` for `scale·‖Ax−b‖²` (default `½‖Ax−b‖²`, with optional
  diagonal residual weights). New battery functionals `SquaredL2NormFunctional`,
  `LpNormFunctional`, `L1NormFunctional`, `NegativeEntropyFunctional`,
  `KLDivergenceFunctional`, and `HuberFunctional` are coordinate objectives whose
  gradients are the metric (Riesz) gradient under the domain geometry.
  `SpectralLpNormFunctional` (with the `NuclearNormFunctional` wrapper) is the
  spectral analogue — the Schatten-`p` norm of a Jordan spectrum (`HermitianSpace`
  eigenvalues), a spectral function whose gradient is reconstructed through the
  ADR-012 `from_spectrum` API; on an elementwise Jordan space it coincides with
  `LpNormFunctional`. A closed-form proximal primitive
  `generalized_shrinkage(X, *, c, x0, eps, lam=0.0, nonneg=False)` solves the
  separable forward–backward subproblem in the space metric (per-coordinate
  threshold `τᵢ = λ/(2 ε wᵢ)` on a diagonal metric; it **raises** on a
  non-diagonal metric rather than returning a wrong separable answer), with the
  named wrappers `prox_l1`, `prox_l2sq`, and `project_nonneg`. The
  `IndicatorFunctional`/`project_C` surface reserved by ADR-019 is deferred to
  ADR-020 (`Set`).
- `HermitianSpace.eig_to_dense` (and therefore `from_spectrum`, `psd_proj`, and
  `spectral_apply`) now symmetrizes the `U diag(·) U^*` reconstruction before the
  membership check, so a zero-tolerance Hermitian space no longer rejects its own
  spectral reconstruction over a few-ULP floating-point skew.
- Public `check_level` policy literal (`"none"`, `"cheap"`, `"standard"`,
  `"strict"`) with `CHECK_LEVELS` ordering and `_checks_at_least` dispatch
  across spaces, LinOps, functionals, and solver preconditions.
- `Space.field` exposing a `Literal["real", "complex"]` mathematical
  contract derived from the context dtype; capability guards now consult
  `Space.field` instead of inspecting precision-bearing dtypes.
- `TreeSpace` finite direct-product space organized by an `optree`
  definition; `TreeElement` ordered-leaf binding; `TreeSpace.from_leaf_spaces`
  flat-tuple shortcut.
- `BlockDiagonalLinOp` and `BlockMatrixLinOp` over `TreeSpace` domains
  including metric-adjoint behavior; `TreeLinOp` base class for tree-shaped
  operators.
- Reusable test generators under `tests/generators/` for spaces, LinOps,
  functionals, and linalg references; contributor guide at
  `docs/dev/contributing/linop_generators.md`.
- `spacecore.kernels` subpackage with the optimized-kernel registration
  policy. Two demonstration kernels ship in 0.4.0: `composed-chain-apply`
  (skips the per-link `@checked_method` wrapper for a flat chain of LinOps)
  and `block-diagonal-dense-apply` (tight ``ops.matmul`` loop over dense
  block leaves). Both have correctness references and bench cases. No
  dispatch or fusion is wired in 0.4.0: the ADR-016 dispatch mechanism is
  implemented but ships **off by default** and dormant, with no production
  routing (see Unreleased).
- Unified benchmark framework at `python -m bench` (subcommands `run`,
  `compare`, `plot`, `summary`, `list`) with generator-driven probes in
  `bench/_operations.py`, peak-memory recording in
  `bench/harness.py:measure_peak_memory`, fixed seeds `(0, 1, 2, 3)`, and
  a self-contained interactive Plotly dashboard at `bench/_dashboard.py`.
- Kernel policy doc at `docs/source/design/kernels_policy.rst`.
- Backend conformance matrix at `docs/source/design/backend_conformance.rst`
  with per-op tolerance harness in `tests/backend/_conformance.py`,
  systematic NumpyOps reference (`tests/backend/test_conformance_numpy.py`),
  cross-backend parity (`tests/backend/test_conformance_cross_backend.py`),
  and dedicated modules for optional args, conversion, dtype promotion,
  field consistency, vmap, and JIT.
- Operator apply cores are organized as a *core-kernel* layer in the
  `spacecore.kernels` subpackage instead of inline in each operator class. The
  check-free cores of `apply`/`rapply`/`vapply`/`rvapply` for **every** operator
  with a fast path — the composite algebra (`ComposedLinOp`, `ScaledLinOp`,
  `SumLinOp`, the adjoint view, `IdentityLinOp`, `ZeroLinOp`, `MatrixFreeLinOp`)
  and the concrete leaves (`DenseLinOp`, `DiagonalLinOp`, `SparseLinOp`) — now
  live as concrete functions in the kernels subpackage (`kernels/algebra.py`,
  `kernels/dense.py`, `kernels/diagonal.py`, `kernels/sparse.py`). Operators
  bind them by declaring the `@core_kernels("...")` class decorator (rules in
  `spacecore/kernels/_core.py`); the base `LinOp` cores remain the generic
  fallback for operators without a registered kernel. Binding is static
  (class-definition time), so routing through the kernel registry costs nothing
  per call — leaf-operator apply latency is unchanged. The lazy-algebra cores
  additionally validate membership only once at the boundary instead of
  re-validating every intermediate link, and `ComposedLinOp` caches a flattened
  `_apply_chain` at construction so a deep `A @ B @ C @ ...` runs one loop rather
  than re-walking the binary tree (a depth-16 composition applies ~9x faster than
  the per-link-checked path and scales flat with depth). All results are
  numerically identical. Public API: `core_kernels`, `CoreKernelSet`,
  `register_core_kernels`, `get_core_kernels`, `core_kernel_names` from
  `spacecore.kernels`.
- Iterative linalg solvers (`cg`, `lanczos_smallest`, `lsqr`, `power_iteration`)
  consume the check-free operator/space cores in their hot loops. They validate
  the operator and right-hand side once, at entry, then run the iteration through
  the resolved cores — eliminating the per-iteration membership validation that
  dominated eager-backend runtime (CG at `check_level="standard"` is ~3.4x faster
  on NumPy and now matches `check_level="none"`). Resolution is safe: a core is
  used only when it is consistent with the public method (`linalg/_utils.py`
  `resolve_core`/`SpaceCoreOps`), so a user space that overrides `inner` with a
  custom geometry without overriding `_inner_core` keeps its override. Results are
  numerically identical.
- The `spacecore.kernels` subpackage is reorganized into two subpackages by kind:
  `spacecore.kernels.core` (the check-free apply/eval cores + the `core_kernels`
  binding rules) and `spacecore.kernels.specs` (the benchmarked `KernelSpec`
  layer). Public names are re-exported from `spacecore.kernels`, so
  `spacecore.kernels.core_kernels`, `spacecore.kernels.CoreKernelSet`, and
  `spacecore.kernels.KernelSpec`/`registry` resolve unchanged.
- The same core-kernel organization now covers the `spacecore.functional`
  submodule. The check-free `value`/`grad`/`vvalue`/`vgrad` cores for
  `InnerProductFunctional`, `MatrixFreeLinearFunctional`, `LinearFunctional`,
  `LinOpQuadraticForm`, and `ComposedFunctional` live in
  `spacecore/kernels/functional.py`; the functionals bind them via
  `@core_kernels("...")`. `CoreKernelSet` gained `value`/`grad`/`vvalue`/`vgrad`
  fields alongside the LinOp `apply`/`rapply`/`vapply`/`rvapply` ones, and the
  base `Functional` carries the generic core fallbacks. Composite functionals now
  reach their operands' cores instead of re-validating intermediates — e.g.
  `LinOpQuadraticForm.value` validates its input once rather than once per
  sub-term (`Q.apply` + `linear.value`), and `ComposedFunctional` evaluates
  `F._value_core(A._apply_core(x))`. Results are numerically identical.
- `BackendOps.hstack`, `vstack`, `dstack`, and `column_stack` array-stacking
  helpers delegating to the backend's native routines, alongside the existing
  `stack`.
- `BackendOps.vectorize` for elementwise vectorization of a scalar Python
  function over array arguments. Delegates to the backend's native
  `vectorize` (NumPy, JAX, CuPy) and uses a portable Python-loop fallback on
  backends without one (Torch). Closes the previously unimplemented
  `ops.vectorize` fallback used by `spectral_apply`.
- Backend deviation catalog at `docs/source/design/backend_deviations.rst`.
- Batching test policy at `docs/source/design/batching_test_policy.rst`.

### Changed

- Spaces, LinOps, and functionals dispatch optional checks via
  `_checks_at_least`; `cheap` covers shape/dtype/backend/tree-structure,
  `standard` adds membership and Hermitian checks, `strict` adds bounded
  expensive probes (matrix-free adjoint identity, CG positive-curvature
  probe).
- `Context.dtype` is documented as the representation default; `Space.field`
  is the mathematical contract derived from it.

### Removed

- `ProductSpace` was removed in favor of `TreeSpace`. Tuple-style products
  use `TreeSpace.from_leaf_spaces((X1, X2, ...))`; nested / dict /
  namedtuple structures use `TreeSpace(template, leaf_spaces)` or
  `TreeSpace.from_template`.
- `ProductLinOp` was renamed to `TreeLinOp`. `ProductStructure`,
  `TupleStructure`, `PytreeStructure`, `ProductStructureCheck`, and
  `ProductComponentCheck` were removed; tree-structure handling and leaf
  validation are owned by `TreeSpace`.
- Conversion (`ctx.asarray` and construction helpers) refuses silent
  complex-to-real narrowing per ADR-015 Stage 1.

### Deprecated

- `enable_checks=True/False` is deprecated in favor of `check_level`:
  `True` maps to `"standard"`, `False` maps to `"none"`. Passing both
  raises `TypeError`. `enable_checks` continues to work in 0.4.0 but will
  be removed in a future release.

### Fixed

- Corrected the matrix-free adjoint contract so `MatrixFreeLinOp` and its
  adjoint view use user-supplied forward and reverse callables directly,
  without applying matrix-backed Riesz-map adjoint corrections.
- `ComposedLinOp`, `ScaledLinOp`, and `SumLinOp` now implement structural
  `is_hermitian()` inference. A Gram product `R.H @ R` (or `L @ L.H`) reports
  `True` in any geometry, a real-scaled operator propagates its operand's
  verdict, and a sum of provably-Hermitian terms reports `True`. The checks
  are cheap, conservative, and never assert `False`, so the normal operator
  `A.H @ A + lam * Identity` is now correctly recognized as self-adjoint
  instead of reporting `None`.
- `cg` now rejects an operator that is *provably* non-self-adjoint in its
  geometry (`A.is_hermitian() is False`) at entry with a clear `ValueError`,
  matching the guard already used by `power_iteration`, `lanczos_smallest`,
  and `expm_multiply`. Previously, at the default check level, `cg` would
  silently accept (for example) a symmetric matrix on a weighted
  inner-product space — which is not self-adjoint under the weighted inner
  product — and return a confusing `converged=False` result. Operators with
  unknown Hermiticity (`is_hermitian() is None`, e.g. matrix-free) are still
  accepted unchecked.

### Known limitations

- Iterative solvers (`cg`, `lsqr`, `lanczos_smallest`, `power_iteration`,
  `expm_multiply`) remain unbatched. Batched-input invocations raise a
  clear shape error; explicit batched solver entry points are deferred to
  0.5.0.
- ADR-015 Stage 2 (operand-dtype-preserving `Context.asarray`, opt-in
  exact dtype membership, operand-dtype solver workspaces) is deferred to
  0.5.0.
- Strict check level currently inherits the standard space-membership
  semantics; additional spectral / metric probes at the space layer are
  deferred to 0.5.0.

## Older releases

0.3.1 and earlier: [docs/dev/changelog-archive.md](docs/dev/changelog-archive.md).
