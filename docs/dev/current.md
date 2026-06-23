# Current Development State

This document is rewritten at each release. It records what maintainers are
actively stabilizing, which design questions remain open, and what contributors
should treat as next work rather than current contract.

## Now

SpaceCore has implemented the `0.4.0` structural work recorded in the phase
notes below. The active focus is the post-`0.4.0` interim line described under
**Next**: turning three recorded design decisions — optimized-kernel dispatch
(ADR-016), external optimizer adapters (ADR-018), and the everyday toolbox
(ADR-019) — into code before the `0.5.0` ergonomics release.

The `0.4.0` Phase B check-policy implementation is complete. Public contexts
use `check_level` with `none`, `cheap`, `standard`, and `strict`; deprecated
`enable_checks` values map to `none` or `standard`.

The `0.4.0` Phase C Stage 1 dtype/field contract is implemented. `Context.dtype`
is the default array representation dtype, while `Space.field` is the derived
real/complex mathematical contract. There is no constructor-level field
override in 0.4.0 because ADR-015 did not authorize one. Exact dtype membership
remains strict, and conversion refuses silent complex-to-real narrowing.

ADR-015 Stage 2 is deferred to `0.5.0`. In particular, `Context.asarray` still
uses the context dtype by default, exact dtype checking is not opt-in, and
solver vector workspaces are not generally operand-dtype driven.

The `0.4.0` Phase D tree-space contract is implemented. `TreeSpace` represents
a finite direct product organized by an `optree` definition, `TreeElement`
optionally binds ordered leaves to that space, and raw matching Python trees
are the normal element representation. `TreeSpace` is the only structured
finite direct-product abstraction. Tuple products use
`TreeSpace.from_leaf_spaces(...)`.

The `0.4.0` Phase I functional and iterative-linalg reference suite is in place.
Functional generators cover analytic values, Riesz gradients, pull-backs,
conversion, weighted geometry, and supported TreeSpace domains. CG, LSQR,
Lanczos, and power iteration have small direct-reference cases, including
metric-adjoint and current workspace-dtype behavior.

The `0.4.0` Phase H LinOp reference suite is in place. It covers every concrete
public LinOp family, Euclidean and weighted metric adjoints, algebraic laws,
conversion, all four check levels for batching, TreeSpace block operators, and
supported NumPy/JAX/Torch/CuPy contexts with explicit sparse-backend skips.

Milestone tracking: [0.3.2 milestone placeholder](https://github.com/Pavlo3P/SpaceCore/milestones).

## Open questions

- Dispatch architecture is now designed in ADR-016 (broader structural dispatch
  over the benchmarked-spec layer, Proposed) and slated for the interim line.
  Remaining open: the per-backend memory-budget source, and whether build-time
  materialization may ever be auto-triggered or stays explicit opt-in.
- Tensor-product spaces: the direct-vs-tensor boundary is settled in ADR-017
  (Deferred); implementation is slated for `0.5.0`.
- Mixed precision: which operations may combine precisions without explicit conversion?
- Solver workspaces: when should vector workspaces follow operand dtype rather than context dtype?
- Backend promotion: which NumPy, JAX, Torch, and CuPy promotion differences are contractual?
- Cross-backend dtype compatibility: which dtype pairs represent the same portable precision?
- Batching limitations and batch-conformance boundaries.
- Batched CG, LSQR, Lanczos, and power-iteration entry points are not currently
  supported. A follow-up must choose between explicit batched solver APIs and a
  documented user-level loop; this phase only guarantees a clear shape error.
- Strict runtime checking intentionally uses bounded probes. Exhaustive
  basis-based adjoint, metric-positive-definiteness, spectral, batched/single,
  and cross-backend checks remain follow-up conformance work in Phases H-J,
  rather than implicit checks in numerical hot loops.
- Phase G generated space coverage therefore treats `strict` as the current
  superset of `standard` space membership. There is no space-local strict-only
  spectral or metric membership check yet; generated tests exercise explicit
  spectral and SPD reference checks without adding them to hot-path validation.

Do not open unsolicited PRs for these unsettled questions unless an issue
already defines the desired work or a maintainer has agreed on the design.

## Next

Between `0.4.0` and `0.5.0`, an interim line implements three recorded design
decisions. Each ADR moves from Proposed to Accepted before its implementation
lands:

- **ADR-016 — optimized-kernel dispatch.** Implement the broader structural-
  dispatch system over the benchmarked-spec layer: `dispatch_key`/`priority` on
  `KernelSpec`, the registry index, the dispatcher with its `dispatch_mode`
  (`off`/`on`/`verify`) and shape-only cost/memory gate, then rewire the approved
  hot call sites. Each activated kernel keeps a `rtol=atol=0` correctness
  reference and a before/after benchmark probe.
- **ADR-018 — external optimizer adapters.** SciPy/optax adapters taking a
  `Functional`, with the metric→coordinate gradient handoff.
- **ADR-019 — everyday toolbox.** Battery functionals and the closed-form
  proximal/projection primitive.

ADR-017 (tensor-product spaces) and ADR-020 (sets and projection) are left for
`0.5.0`, alongside the ergonomics, identity, and dtype Stage 2 work.
