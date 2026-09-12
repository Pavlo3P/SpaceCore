# Changelog

All notable changes to SpaceCore are documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and the project adheres to [Semantic Versioning](https://semver.org/).

## [Unreleased]

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

## [0.3.1] — 2026-06-10

SpaceCore 0.3.1 is a release-candidate stabilization release for the `0.3.x`
API. It focuses on documentation consistency, tutorial execution, release
artifact checks, and public API audit cleanup. It does not add new solver
families or SDPLab-specific downstream integration.

### Documentation

- Reworked API reference landing pages for backend, context, spaces, linear
  operators, functionals, and linalg.
- Added design notes for context ownership, batching, and capability dispatch.
- Clarified conversion and dtype policy documentation for explicit target
  contexts.
- Clarified adjoint documentation to distinguish coordinate transpose,
  Euclidean adjoint, and metric/Riesz-represented adjoint behavior.

### Examples and Tutorials

- Added a SpaceCore-only weighted Tikhonov worked example demonstrating
  weighted spaces, metric adjoints, lazy operator algebra, CG, and an
  independent dense NumPy reference solve.
- Integrated the weighted Tikhonov example into tests and documentation.

### Testing and CI

- Documentation CI now builds Sphinx with warnings as errors.
- Release-candidate checks include full tests, strict docs build, public API
  audit, artifact build, `twine check`, clean wheel installation, and smoke
  testing.

### Known limitations

- Optional backend behavior depends on installed optional dependencies. CuPy is
  not required for the core release-candidate gate.
- The advanced regularized OT tutorial is an illustrative SpaceCore/JAX/Optax
  example, not a claim that SpaceCore ships a production OT solver.

## [0.3.0]

SpaceCore 0.3.0 is a breaking release for the unstable `0.x` series. Space
capabilities are now derived from actual structure, dtype, and inner product,
and conversions rebuild spaces through public factories so stale capabilities
are not retained.

### Migration

| 0.2.x | 0.3.0 |
| --- | --- |
| `space.eigh(x)` | `space.spectral_decompose(x)` for eigenvalues and frame |
| `space.eigh(x)` | `space.spectrum(x)` for eigenvalues only |
| `sc.VectorSpace((n,))` | `sc.DenseVectorSpace((n,))` |
| `sc.VectorSpace((d, d))` | `sc.DenseCoordinateSpace((d, d))` |
| `ProductInnerProductSpace(...)` | `ProductSpace(...)` |
| `ProductStarSpace(...)` | `ProductSpace(...)` |
| `ProductJordanAlgebraSpace(...)` | `ProductSpace(...)` |
| `ProductEuclideanJordanAlgebraSpace(...)` | `ProductSpace(...)` |
| `StackedInnerProductSpace(...)` | `StackedSpace(...)` |
| `StackedStarSpace(...)` | `StackedSpace(...)` |
| `StackedJordanAlgebraSpace(...)` | `StackedSpace(...)` |
| `StackedEuclideanJordanAlgebraSpace(...)` | `StackedSpace(...)` |
| `BatchSpace` and `space.batch(...)` | leading-axis batched arrays with `vapply(...)` / `rvapply(...)` |
| `op.vapply(xs, batch_space=...)` | `op.vapply(xs)` |
| global context conversion policies | explicit `Context` construction and `obj.convert(ctx)` |
| global dtype preservation policies | target-context dtype during explicit conversion |

Prominent `eigh` replacement:

```python
space.eigh(x)
# -> space.spectral_decompose(x)  # eigenvalues and frame
# -> space.spectrum(x)            # eigenvalues only
```

### Added

- `spectrum`, `spectral_decompose`, and `from_spectrum` as the public spectral
  contract for Jordan spaces.
- `ElementwiseJordanSpace` for real or complex elementwise Jordan algebras.
- `EuclideanElementwiseJordanSpace` for real Euclidean elementwise Jordan
  algebras.
- Jordan capability hierarchy separating `JordanAlgebraSpace` from
  `EuclideanJordanAlgebraSpace`.
- `ProductStructure`, `TupleStructure`, `PytreeStructure`, and
  `ProductSpace.from_template` for structured product elements.
- `ProductSpectralDecomposition` for product spectral data independent of
  product element structure.
- `StackedSpace` for leading-axis repeated leaf spaces.
- Vectorizable axis-aware validation checks.
- `InnerProduct.validate_for(space)` and construction-time validation for
  `WeightedInnerProduct`.
- `scripts/api_audit.py` for repository and downstream migration audits.

### Changed

- `VectorSpace` is an abstract linear-capability base.
- Previous concrete `VectorSpace` use cases moved to `DenseVectorSpace` for
  one-dimensional dense vectors and `DenseCoordinateSpace` for generic dense
  coordinate arrays.
- `DenseVectorSpace` is now a plain one-dimensional vector space with star and
  no Jordan capability by default.
- Elementwise Euclidean-Jordan capability is selected only for real dtype with
  `EuclideanInnerProduct`.
- `ProductSpace(...)` and `StackedSpace(...)` are the only public product and
  stacked constructors; they auto-dispatch to private implementation classes.
- `convert()` for elementwise, product, and stacked spaces recomputes
  capabilities through public factories.
- `ProductSpace.spectral_decompose` returns explicit product spectral data
  rather than routing decompositions through element structure adapters.

### Removed

- Removed `eigh` from spaces. Use `spectral_decompose` when both eigenvalues
  and a reconstruction frame are needed, or `spectrum` for eigenvalues only.
- Removed public specialized product and stacked constructors from the public
  API.
- Removed `BatchSpace`, `Space.batch`, and `batch_space=` arguments from public
  batching APIs. Use leading-axis vectorization through `vapply` and `rvapply`.
- Removed global context-policy and dtype-policy APIs. Conversion now follows
  the requested target `Context` directly.

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

[0.3.1]: https://github.com/Pavlo3P/SpaceCore/releases/tag/v0.3.1
[0.3.0]: https://github.com/Pavlo3P/SpaceCore/releases/tag/v0.3.0
[0.2.0]: https://github.com/Pavlo3P/SpaceCore/releases/tag/v0.2.0
