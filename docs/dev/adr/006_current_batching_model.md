# ADR-006: Current batching model

## Status

Accepted

## Context

Batching is current SpaceCore behavior. It must be documented before 0.4.0 tests are generated so contributors do not accidentally remove or misstate it.

## Current design

A space describes one mathematical element. Batched evaluation is represented by leading axes on values, not by changing `domain` or `codomain`. `_batching` contains shared helpers for batched membership checks and batched inner products. `checked_method(..., in_batched=True, out_batched=True)` validates trailing element shape while allowing leading dimensions.

`LinOp.vapply(xs)` applies `apply` over a leading batch axis. `LinOp.rvapply(ys)` applies the adjoint over a leading batch axis. The base implementation uses `ops.vmap`; dense, sparse, diagonal, algebraic, and product-structured operators provide specialized batched paths where useful. Functionals similarly expose `vvalue` and selected `vgrad` methods. `StackedSpace` is separate: it makes a fixed stack part of one element rather than a transient evaluation batch.

## Decision

Batching stays as explicit vectorized methods (`vapply`, `rvapply`, `vvalue`, `vgrad`) and space batch helpers. `apply` and `rapply` remain single-element methods.

## Rationale

This preserves the mathematical meaning of a space while supporting efficient backend vectorization and specialized dense/product fast paths.

## Alternatives considered

Removing batching was rejected because current operators, tests, and materialization paths rely on it. Treating every leading axis as implicit batching in `apply` was rejected because it blurs single-element membership and makes product/pytree structure ambiguous.

## Consequences

0.4.0 tests must cover leading-axis validation, product-structured batches, backend fallback loops, native backend vectorization, metric `rvapply`, and consistency between single and batched methods. Known limitations include a leading-axis-only convention, backend-dependent performance, and fallback warnings where NumPy-style loops are used.

## Contributor invariants

- `apply` and `rapply` are single-element contracts.
- `vapply` and `rvapply` own batched LinOp behavior.
- Batched validation must allow leading axes but still enforce trailing element shape, backend, dtype, and product component structure.
- Backend context conversion must preserve batched behavior and not bypass metric adjoints.
