# ADR-007: LinOp contract

## Status

Accepted

## Context

Linear operators are the central abstraction connecting spaces, geometry, batching, and linalg.

## Current design

`LinOp` owns a domain space, codomain space, and context. Construction resolves one context and converts both endpoint spaces to it. Subclasses implement `apply(x)` and `rapply(y)`. `rapply` is the Hermitian adjoint with respect to the declared space geometries. `.H` returns a cached adjoint view. Algebraic operations build lazy sums, scalar multiples, differences, and compositions. `to_dense` and `to_matrix` are explicit materialization helpers for small problems and tests.

Validation uses `checked_method` and endpoint membership checks when `ctx.enable_checks` is true. Conversion rebuilds spaces and owned data or operands in a target context.

## Decision

The LinOp contract is mathematical: `A : X -> Y` with `apply : X -> Y` and `rapply : Y -> X` satisfying the inner-product adjoint identity. Algebra must preserve domain/codomain compatibility and context compatibility.

## Rationale

Solvers and functionals can rely on `LinOp` without knowing whether the operator is dense, sparse, matrix-free, or structured.

## Alternatives considered

A matrix-only operator API was rejected because matrix-free and structured operators are first-class. Returning raw transpose matrices for adjoints was rejected because non-Euclidean spaces require metric adjoints.

## Consequences

New LinOp subclasses must implement true `rapply`, batching when they can improve on `ops.vmap`, conversion, and pytree flattening where applicable. See [ADR-009](009_metric_adjoint.md) for metric adjoints and [ADR-008](008_linop_subclasses.md) for implemented families.

## Contributor invariants

- Domain and codomain spaces are part of operator identity.
- `rapply` must be the geometry-aware adjoint, not merely a coordinate transpose.
- `.H.apply(y)` must be equivalent to `rapply(y)`.
- Algebraic operators must preserve compatible contexts and spaces.
- Validation must check inputs and outputs against the declared endpoint spaces when enabled.
