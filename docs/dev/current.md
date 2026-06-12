# Current Development State

This document is rewritten at each release. It records what maintainers are
actively stabilizing, which design questions remain open, and what contributors
should treat as next work rather than current contract.

## Now

SpaceCore is preparing a `0.3.2` contributor-infrastructure release. Current
work is focused on architecture ADRs, contributor documentation, GitHub
templates and labels, `0.4.0` issue decomposition, the matrix-free `LinOp`
adjoint-contract fix, and documentation of the current batching model.

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

Milestone tracking: [0.3.2 milestone placeholder](https://github.com/Pavlo3P/SpaceCore/milestones).

## Open questions

- Dispatch architecture.
- Tensor-product spaces.
- Mixed precision: which operations may combine precisions without explicit conversion?
- Solver workspaces: when should vector workspaces follow operand dtype rather than context dtype?
- Backend promotion: which NumPy, JAX, Torch, and CuPy promotion differences are contractual?
- Cross-backend dtype compatibility: which dtype pairs represent the same portable precision?
- Batching limitations and batch-conformance boundaries.
- Strict runtime checking intentionally uses bounded probes. Exhaustive
  basis-based adjoint, metric-positive-definiteness, spectral, batched/single,
  and cross-backend checks remain follow-up conformance work in Phases H-J,
  rather than implicit checks in numerical hot loops.

Do not open unsolicited PRs for these unsettled questions unless an issue
already defines the desired work or a maintainer has agreed on the design.

## Next

`0.4.0` is tentatively focused on test infrastructure: reusable generators for
spaces, LinOps, functionals, and linalg; backend conformance; batching
conformance; check-policy migration; dtype/field Stage 1; and block and
tree-structured testing.
