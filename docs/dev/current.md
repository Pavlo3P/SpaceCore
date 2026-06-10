# Current Development State

This document is rewritten at each release. It records what maintainers are
actively stabilizing, which design questions remain open, and what contributors
should treat as next work rather than current contract.

## Now

SpaceCore is preparing a `0.3.2` contributor-infrastructure release. Current
work is focused on architecture ADRs, contributor documentation, GitHub
templates and labels, `0.4.0` issue decomposition, the matrix-free `LinOp`
adjoint-contract fix, and documentation of the current batching model.

Milestone tracking: [0.3.2 milestone placeholder](https://github.com/Pavlo3P/SpaceCore/milestones).

## Open questions

- Dtype defaults versus scalar-field contract.
- Dispatch architecture.
- Tensor-product spaces.
- Mixed precision and cross-precision compatibility.
- Batching limitations and batch-conformance boundaries.

Do not open unsolicited PRs for these unsettled questions unless an issue
already defines the desired work or a maintainer has agreed on the design.

## Next

`0.4.0` is tentatively focused on test infrastructure: reusable generators for
spaces, LinOps, functionals, and linalg; backend conformance; batching
conformance; check-policy migration; dtype/field Stage 1; and block, product,
and tree-structured testing.
