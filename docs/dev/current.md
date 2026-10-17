# Current Development State

This document is rewritten at each release. It is the canonical, committed record
of what SpaceCore is actively working on, which design questions remain open, and
the forward roadmap. Where this document and any uncommitted planning note
disagree, this document wins.

## Now

### `0.5.0` — 2026-10-17 (release prepared, not tagged)

Every public operation of spaces, operators and functionals is a checked wrapper
over a check-free `_<op>_core` that families implement; an undecorated public
override becomes the class's core (`install_public_checks`), composites call
cores, and the solvers run on cores directly (`SpaceCoreOps` is gone). Spaces,
operators and functionals are hashable. Checks inside `jax.jit` / `torch.compile`
follow `set_traced_check_level` (default `"none"`). Verified locally on
Python 3.11, 3.12 and 3.13 (suite, doctests, coverage 94%, pyright/mypy typing
tests, ruff, docstring and JIT audits, `sphinx -W`). Not yet verified: the CuPy
backend on a GPU, and SDPLab, whose `master` still targets SpaceCore 0.4.2 and
does not import on 0.4.3 or later (`jax_pytree_class`).

History of the 0.5.0a work: `CHANGELOG.md`. Next: downstream adoption, demand-driven
additional prox rules, and measured performance work; broad performance parity is not
established.

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

### `0.5.0` — Ergonomics, identity, dtype, and new abstractions

| Item | State in 0.5.0 |
|---|---|
| **Object identity** | Shipped: spaces, operators and functionals hash consistently with `==`; `_repr_html_`. |
| **Ergonomics** | Shipped as the core rule: every public operation is a checked wrapper over a `_<op>_core`, one `install_public_checks` mechanism for all objects, the traced check-level cap. |
| **ADR-020 — sets and projection** | Shipped (minimal surface: `Set`, `NonnegativeOrthant`, `IndicatorFunctional`). |
| **ADR-017 — tensor-product spaces** | Not in 0.5.0. Design proposed in ADR-017 (2026-10-17), awaiting acceptance and a named workload. |
| **Dtype Stage 2 (ADR-015)** | Safe part shipped: solver scalar diagnostics are real for complex operators (`cg`, `power_iteration` joined Lanczos and `expm_multiply`); `ctx.asarray` already returns an input of the context dtype without copying. Open: field-level `DTypeCheck` by default and dtype-preserving `ctx.asarray` for other dtypes, which ADR-015 keeps strict until a workload justifies them. |

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

**Final work** — freeze the public API; remove any abandoned experimental surface;
complete the typing (`uvx pyright` baseline), naming, and documentation
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
