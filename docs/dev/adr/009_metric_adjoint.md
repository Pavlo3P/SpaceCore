# ADR-009: Metric adjoint

## Status

Accepted

## Context

Non-Euclidean inner products make the Euclidean coordinate adjoint insufficient. This ADR records the release-gating rule for correct adjoints.

## Current design

Spaces expose Riesz maps through their `InnerProduct` geometry. Matrix-backed `DenseLinOp`, `SparseLinOp`, and `DiagonalLinOp` first compute the Euclidean coordinate adjoint and then adapt it to the declared domain and codomain geometry. The formula is:

```text
A^sharp = R_X^{-1} A^dagger R_Y
```

where `A : X -> Y`, `A^dagger` is the Euclidean conjugate coordinate adjoint, and `A^sharp` is the metric adjoint. Fast paths exist for Euclidean and weighted diagonal geometries. General metric paths require usable Riesz maps. `InnerProduct.validate_for(space)` validates geometry storage such as weighted metrics.

`MatrixFreeLinOp` is different: its direct `rapply` callable is already the metric adjoint and must not be Riesz-wrapped. `MatrixFreeLinOp.from_coordinate_adjoint(...)` exists for the case where the user has a Euclidean coordinate adjoint and wants SpaceCore to wrap it.

## Decision

Matrix-backed operators are responsible for converting coordinate adjoints into metric adjoints. Matrix-free direct `rapply` is a true adjoint by contract. Type guards alone are not a correctness fix because the issue is geometry, not only class identity.

## Rationale

The adjoint identity must hold for the declared spaces. Riesz maps are the architectural boundary between coordinate storage and mathematical geometry.

## Alternatives considered

Using only type guards for Euclidean concrete spaces was rejected because custom spaces and weighted geometries need the same mathematical rule. Always Riesz-wrapping matrix-free `rapply` was rejected because it would double-wrap correct true adjoints and break user-supplied non-coordinate implementations.

## Consequences

Coordinate-backed operators must reject non-Euclidean spaces without Riesz maps. Matrix-free docs and tests must distinguish true adjoints from coordinate adjoints. Batched metric adjoints should use batched Riesz maps when possible and may fall back to vectorized scalar adjoints with a warning. See [ADR-004](004_inner_product_and_geometry.md) and [ADR-007](007_linop_contract.md).

## Contributor invariants

- Matrix-backed adjoints use `R_X^{-1} A^dagger R_Y`.
- `MatrixFreeLinOp(..., rapply=...)` receives a true metric adjoint and must not be automatically Riesz-wrapped.
- `from_coordinate_adjoint` is the explicit API for wrapping a Euclidean coordinate adjoint.
- `validate_for` and Riesz-map availability must be checked before accepting non-Euclidean coordinate-backed operators.
