# Current Development State

This document is rewritten at each release. It is the canonical, committed record
of what SpaceCore is actively working on, which design questions remain open, and
the forward roadmap. Where this document and any uncommitted planning note
disagree, this document wins.

## Now

### `0.5.0a` real functional contract — 2026-10-07

All Functionals have a real codomain and produce real values, including on
complex domains. Complex outputs, codomains, scaling coefficients and offsets
are rejected even at `check_level="none"`. Inner-product functionals and
Hermitian quadratic forms use the real pairing. Batch, fused, conversion and
pytree paths preserve the contract; see ADR-010.

### `0.5.0a` simplification pass — 2026-10-06

The package lost layers without changing the Backend → Space → Mapping
(LinOp / Functional) → Element division:

- `LinOp` and `Functional` own their checked public methods; subclasses write only
  the check-free cores (`_apply_core`, `_value_core`, ...). The `kernels` package
  that held those cores as free functions is gone.
- A class declares its constructor `_fields` once; pytree flattening, `convert`
  and `Mapping` equality derive from it (`docs/source/design/extending.rst`).
- `BlockMatrixLinOp` covers stacking and summing (`stack()`, `from_operators()`);
  `StackedLinOp`, `SumToSingleLinOp`, `spacecore.opfamily`, the legacy
  `BlockDiagonalLinOp(dom, cod, parts, ctx)` form and `_check_policy.py` are gone.
- The map algebra lives in the `spacecore.mapping` package.
- The optimizer adapters moved to the separate `spacecore-optimize` package; the
  notebooks, tutorial notebooks, examples and the `bench` suite left the
  repository (the rendered tutorial pages stay in the docs).

### `0.5.0a` proximal milestone — 2026-09-28

The first-order toolbox now has optional functional prox rules, Hilbert conjugate prox, and the minimal
ADR-020 set/indicator/projection surface. Tests exercise
unchanged proximal-gradient/PDHG loops across dense/native elements and
Euclidean/weighted geometry, including compiled JAX. Unsupported operations
raise explicit capability errors. See `docs/source/design/proximal_contract.rst`.

Next: downstream adoption, demand-driven additional rules, and measured
performance improvements/regression tracking. Broad performance parity is not
established. The release train below is historical planning: ADR-016 dispatch,
the ADR-018 adapters and the ADR-023 bench have since left the package, and native
functionals and Krylov paths have advanced.

### Earlier release planning

`0.4.0` is implemented bar the Phase J/M closeout (backend-conformance matrix
completion, deviation catalog, doc-build hygiene). Active work is the post-`0.4.0`
interim line — ADR-016 dispatch (mechanism now implemented, off by default),
ADR-018 optimizer adapters, ADR-019 everyday toolbox (see Roadmap).

Milestone tracking: [SpaceCore milestones](https://github.com/Pavlo3P/SpaceCore/milestones).

## Open questions

Design decisions not yet settled. Do not open unsolicited PRs for these unless an
issue already defines the work or a maintainer has agreed on the design.

- **Mixed precision** — which operations may combine precisions without explicit
  conversion?
- **Solver workspaces** — when should vector workspaces follow operand dtype
  rather than context dtype?
- **Backend promotion** — which NumPy/JAX/Torch/CuPy promotion differences are
  contractual?
- **Cross-backend dtype compatibility** — which dtype pairs represent the same
  portable precision? Do not relax `Context.__eq__`/`Space.__eq__` across
  precision without recording this decision.
- **Batched solver entry points** — CG/LSQR/Lanczos/power-iteration batching is
  unsupported; choose between an explicit batched API and a documented user loop.
  Today only a clear shape error is guaranteed.
- **Strict-check scope** — exhaustive basis-based adjoint, metric
  positive-definiteness, spectral, and cross-backend checks remain conformance
  follow-ups, not implicit checks in numerical hot loops.

## Roadmap

Targets are tentative; the release **gates** are the contract, not the dates. Each
ADR moves from Proposed to Accepted before its implementation lands. No
optimization changes dispatch, and no demand-gated abstraction is built, before
its ADR is accepted.

### Interim line (post-`0.4.0`)

Turn three recorded design decisions into code:

- **ADR-016 — optimized-kernel dispatch.** *Removed in 0.5.0a*: its batched specs
  ran mostly slower than the plain loops on eager NumPy and never under `jax.jit`;
  the check-free cores it sat on are now methods of their classes.
- **ADR-018 — external optimizer adapters.** *Moved out in 0.5.0a* to the
  `spacecore-optimize` package (`minimize_scipy`, `line_search_scipy`,
  `minimize_optax`).
- **ADR-019 — everyday toolbox.** Battery functionals and a closed-form
  proximal/projection primitive.

### `0.5.0` — Ergonomics, identity, dtype, and new abstractions

- **Object identity.** Restore `__hash__` consistently across the abstract bases
  (defining `__eq__` left value objects unhashable); add an optional `_repr_html_`
  for notebooks.
- **Ergonomics.** Small, explicit wrappers/decorators for repeated jobs —
  check-level logic, domain/codomain compatibility, scalar-field/dtype checks,
  context-preserving construction, bare-array→element coercion, batched
  `vapply`/`rvapply` fallback, functional gradient/pull-back validation. The goal
  is readability, never magical dispatch; Riesz wrappers stay honest (ADR-009:
  matrix-free `rapply` is never wrapped).
- **Dtype Stage 2 (ADR-015).** Dtype-preserving `ctx.asarray`; field-level
  `DTypeCheck` as default with exact-dtype opt-in; operand-driven solver
  workspaces.
- **ADR-017 — tensor-product spaces.** `TensorProductSpace`, tensor-shaped
  elements, Kronecker/tensor LinOps, and adjoint/flattening rules, kept distinct
  from `TreeSpace` direct products.
- **ADR-020 — sets and projection.** A `Set` abstraction owning an ambient space
  and a metric projector, with a typed `C` for indicators/projection.

### `0.6.0` — Interoperability, linalg, and performance

- **LinearOperator interop adapters** — `to_scipy_linear_operator`/
  `from_scipy_linear_operator`, `to_pylops`/`from_pylops`; evaluate CoLA/Pyxu and
  implement only where the mapping is mathematically honest. Preserve application
  and Euclidean adjoint behavior; document the lost information (structured
  elements, non-Euclidean geometry, backend context, generalized adjoints,
  scalar-field/dtype). Distinct from the ADR-018 optimizer adapters.
- **Linalg** — a prioritized survey (MINRES, GMRES, BiCGSTAB, Arnoldi, randomized
  range finder/SVD) against real demand; a common preconditioner protocol added to
  the iterative methods; LOBPCG and the highest-priority survey methods; and a
  design note on non-Euclidean preconditioning (how preconditioners transform
  under a Riesz map `R_X`).
- **Performance** — measure abstraction overhead against the stable generators;
  add a specialized path for block / Kronecker / tensor-product operators only
  where a benchmark under `jax.jit` shows a gain, each behind a correctness
  reference; add performance-regression tracking to CI (the benchmark suite now
  lives outside this repository).

### `0.7.0` — Documentation and tutorials

- **Mathematical tutorials** (referenced) — Jordan algebras and standard examples;
  generalized adjoints and Riesz maps; dtype defaults vs scalar-field contract;
  non-Euclidean preconditioning; structured/block/tensor/matrix-free operators.
  Each cites standard books, courses, or foundational papers.
- **Extension tutorials** — implement a custom space, a matrix-free LinOp, a
  functional with a specific gradient, and a generic backend-independent algorithm,
  driven off the `0.5.0` ergonomics helpers so the tutorials show the easy path.
- **Application / comparison tutorials** — PDE / inverse problems, optimal
  transport, manifold optimization, conic/PDHG (and RKHS if non-coordinate support
  is mature). Each compares SpaceCore against an established library on code
  complexity, mathematical expressiveness, backend portability, runtime, and
  limitations, reusing the `0.6.0` adapters. Logo and documentation navigation
  finalized here.

### `1.0.0` — Stable public API

**Requirements** — core abstractions exercised in SpaceCore and SDPLab; backend
contracts stable and documented with a current deviation catalog; extension APIs
demonstrated by tutorials outside the core package; reliable interoperability
adapters; benchmarks and regression tracking running in CI; a defined
deprecation/compatibility policy.

**Final work** — freeze the public API; remove any abandoned experimental surface
(the `enable_checks` shim, `opfamily`, the `kernels` package and the legacy
`BlockDiagonalLinOp` form already went in 0.5.0a); complete the typing
(`docs/dev/typing-audit.md`, `uvx pyright` baseline), naming, and documentation
audits; run full downstream migration and artifact verification.

## Continuous parallel work

Throughout the train:

- Maintain `CHANGELOG.md`, migration notes, and `docs/source/release_notes.rst`;
  publish patch releases for regressions.
- Validate changes against SDPLab.
- Benchmark before and after every optimization; never optimize on intuition.
- Reject abstractions that hide problem-specific mathematics.
- Update this document at the **start** of each release, before writing code —
  including the open-question list.
- Write a new ADR whenever a non-trivial design decision is made.
