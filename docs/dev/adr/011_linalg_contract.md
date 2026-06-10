# ADR-011: Linalg contract

## Status

Accepted

## Context

SpaceCore linalg routines must work for dense, sparse, matrix-free, structured, Euclidean, and non-Euclidean operators without materializing matrices by default.

## Current design

Implemented iterative routines include `cg`, `lsqr`, `power_iteration`, `lanczos_smallest`, and `expm_multiply`. They operate on `LinOp` or, for power iteration, a quadratic-form Hessian action. They use `LinOp.apply`, `.H.apply`, space `add`, `scale`, `inner`, and `norm`. Backend loops, conditionals, indexing, eigendecompositions of small projected matrices, and array constructors are delegated to `BackendOps`.

Correctness assumptions are explicit. CG requires square Hermitian positive-definite operators in the domain geometry. LSQR uses the metric adjoint for normal-equation residuals. Power iteration, Lanczos, and exponential multiply require self-adjoint actions; known non-Hermitian structure is rejected, unknown matrix-free structure is trusted.

## Decision

Linalg routines are operator-driven and geometry-aware. They must not require dense materialization of the input operator and must rely on backend primitives for loop and array behavior.

## Rationale

This lets the same algorithms run on dense, sparse, lazy, product, and matrix-free operators while preserving non-Euclidean correctness.

## Alternatives considered

Delegating all linalg to backend dense solvers was rejected because it would lose structure and matrix-free support. Reimplementing backend control flow ad hoc inside each solver was rejected because JAX/Torch/NumPy need different loop semantics.

## Consequences

New solvers must state their mathematical assumptions in terms of spaces and geometry. They must use `A.H` for adjoint products and backend loop primitives for JIT-compatible backends. Solver workspaces should be driven by operand/operator dtype; see [ADR-015](015_dtype_default_vs_scalar_field.md).

## Contributor invariants

- Solvers use space operations for vector arithmetic and geometry.
- Solvers do not materialize `A` unless the routine explicitly documents that choice.
- Hermitian, positive-definite, and residual assumptions are with respect to the declared space inner products.
- Backend-specific loop, indexing, and small dense linalg behavior goes through `BackendOps`.
