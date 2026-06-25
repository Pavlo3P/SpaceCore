# Current Development State

This document is rewritten at each release. It is the canonical, committed record
of what SpaceCore is actively working on, which design questions remain open, and
the forward roadmap. Where this document and any uncommitted planning note
disagree, this document wins.

## Now

`0.4.0` is implemented bar the Phase J/M closeout (backend-conformance matrix
completion, deviation catalog, doc-build hygiene). Active work is the post-`0.4.0`
interim line — ADR-016 dispatch (mechanism now implemented, off by default),
ADR-018 optimizer adapters, ADR-019 everyday toolbox (see Roadmap).

Milestone tracking: [SpaceCore milestones](https://github.com/Pavlo3P/SpaceCore/milestones).

## Open questions

Design decisions not yet settled. Do not open unsolicited PRs for these unless an
issue already defines the work or a maintainer has agreed on the design.

- **Dispatch** — ADR-016 is accepted and the structural-dispatch mechanism is
  implemented (`dispatch_key`/`priority`/`cost`, the registry index, the
  `dispatch()` entry point with `off`/`on`/`verify` modes and the memory gate),
  off by default. Three dispatch-eligible algebraic specs now route at the wired
  call sites (`composed-zero-annihilation`, `composed-identity-elision`,
  `block-diagonal-uniform-dense-batched`); `NumpyOps.free_memory_bytes()` reports
  RAM via optional `psutil`. Remaining: per-backend `free_memory_bytes()` for
  GPU backends; cross-backend bit-exactness of the batched block fold (NumPy-only
  today); whether build-time materialization may ever be auto-triggered or stays
  explicit opt-in.
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

- **ADR-016 — optimized-kernel dispatch.** *Implemented* (off by default):
  `dispatch_key`/`priority`/`cost` on `KernelSpec`, the registry index, the
  `dispatch()` entry point with its `dispatch_mode` (`off`/`on`/`verify`) and
  shape-only memory gate, the composed / block-diagonal call sites rewired, and
  three dispatch-eligible algebraic specs (zero annihilation, identity elision,
  uniform-dense batched block matmul) each with an `rtol=atol=0` correctness
  reference and a bench probe. Remaining: GPU-backend `free_memory_bytes()` and
  cross-backend bit-exactness of the batched block fold (NumPy-only today).
- **ADR-018 — external optimizer adapters.** *Implemented*: the
  `spacecore.optimize` subpackage ships `minimize_scipy`, `line_search_scipy`,
  and `minimize_optax`, each taking a `Functional` and performing the
  metric→coordinate gradient handoff (`X.riesz(F.grad(x))`) centrally. The SciPy
  adapters reject complex domains; `minimize_optax` requires a JAX-backed domain
  and the optional `optax` extra. jaxopt/BlackJAX/pymanopt stay as tutorials.
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
  add optimized kernels for block / Kronecker / tensor-product operators, each
  behind a correctness reference and a benchmark; add performance-regression
  tracking to CI on top of `python -m bench`.

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

**Final work** — freeze the public API; remove the `enable_checks` deprecation
shim and any abandoned experimental surface; complete the typing
(`docs/dev/typing-audit.md`, `uvx pyright` baseline), naming, and documentation
audits; run full downstream migration and artifact verification. Keep the `1.0.0`
definition in sync with `vision.md`.

## Continuous parallel work

Throughout the train:

- Maintain `CHANGELOG.md`, migration notes, and `docs/source/release_notes.rst`;
  publish patch releases for regressions.
- Validate changes against SDPLab.
- Benchmark before and after every optimization; never optimize on intuition.
- Reject abstractions that hide problem-specific mathematics (the core invariant
  from `vision.md`).
- Update this document at the **start** of each release, before writing code —
  including the open-question list.
- Write a new ADR whenever a non-trivial design decision is made.
