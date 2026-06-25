# ADR-008: LinOp subclasses

## Status

Accepted

## Context

SpaceCore implements several LinOp storage and composition families. Contributors need to know which subclasses own coordinates and which rely on user callables.

## Current design

Matrix-backed operators include `DenseLinOp`, `SparseLinOp`, and `DiagonalLinOp`. They own dense tensors, sparse coordinate matrices, or diagonal arrays and implement coordinate forward actions plus metric-aware adjoints. `DenseLinOp` stores shape `cod.shape + dom.shape`; `SparseLinOp` stores a 2D sparse matrix and is limited to coordinate spaces; `DiagonalLinOp` stores one diagonal element per space coordinate.

Matrix-free operators include `MatrixFreeLinOp`, which trusts user-supplied callables. `IdentityLinOp` and `ZeroLinOp` are structural operators with no matrix storage. Lazy algebraic operators include `ComposedLinOp`, `SumLinOp`, and `ScaledLinOp`. Tree-structured operators include `TreeLinOp`, `BlockDiagonalLinOp`, `StackedLinOp`, and `SumToSingleLinOp`.

## Decision

Coordinate-backed operators own coordinate representations and may optimize dense, sparse, weighted, flat, and batched paths. Matrix-free operators trust supplied callables and only validate membership. Lazy and product operators delegate to their operands and preserve structure.

## Rationale

This keeps high-performance coordinate cases efficient while preserving matrix-free and structured composition as first-class APIs.

## Alternatives considered

Forcing every operator to expose a stored matrix was rejected because it breaks matrix-free and lazy use cases. Eagerly simplifying all algebra into dense matrices was rejected because it destroys sparsity, structure, and backend tracing.

## Consequences

New structured operators should avoid materializing unless explicitly requested. Tree operators must preserve `TreeSpace` definitions and deterministic leaf order. Coordinate-backed operators must implement conversion for stored arrays. Matrix-free conversion cannot rewrite backend-specific Python callables.

## Contributor invariants

- Matrix-backed subclasses own and validate their coordinate storage shape.
- Matrix-free `apply` and `rapply` callables are trusted as the mathematical contract.
- Lazy algebra must not reorder or densify operands unless a documented factory rule says so.
- Product operators must preserve domain and codomain product structures in `apply`, `rapply`, `vapply`, and `rvapply`.
