# ADR-005: Space subclasses and capabilities

## Status

Accepted

## Context

Concrete spaces expose different mathematical operations. Contributors need a stable way to reason about which operations are present and how conversion affects them.

## Current design

Public concrete spaces include `DenseCoordinateSpace`, `DenseVectorSpace`, `ElementwiseJordanSpace`, `EuclideanElementwiseJordanSpace`, `HermitianSpace`, `ProductSpace`, and `StackedSpace`. Capability classes include `InnerProductSpace`, `StarSpace`, `JordanAlgebraSpace`, and `EuclideanJordanAlgebraSpace`.

`DenseCoordinateSpace` has inner-product geometry but no star operation. `DenseVectorSpace` adds elementwise conjugation. `ElementwiseJordanSpace` adds elementwise Jordan and spectral operations; when context and geometry are real Euclidean, construction dispatches to `EuclideanElementwiseJordanSpace`. `HermitianSpace` represents dense Hermitian matrices with Frobenius geometry and spectral calculus. `ProductSpace` and `StackedSpace` dynamically choose internal subclasses that preserve only capabilities shared by their components or base.

## Decision

Capabilities are represented by Python class membership, not by flags. Geometry belongs to inner-product-capable spaces. Product and stacked conversion rebuilds converted components and then recomputes the capability-specific class.

## Rationale

Class-based capabilities make missing operations fail naturally and keep capability composition explicit. Recomputing on conversion prevents stale capability promises after dtype or geometry changes.

## Alternatives considered

Single concrete classes with runtime feature flags were rejected because they make unsupported operations harder to detect. Forcing every dense space to expose star or Jordan operations was rejected because those operations are not always mathematically valid.

## Consequences

Code should test capabilities with `isinstance`, not with name checks or ad hoc attributes. New concrete spaces must be explicit about which capabilities they provide. Spectral operations belong only to Jordan-capable spaces; see [ADR-012](012_jordan_spectrum.md).

## Contributor invariants

- Public capabilities must match actual implemented methods.
- `convert()` must recompute product and stacked capabilities in the target context.
- Geometry conversion must convert stored geometry arrays such as weights.
- Missing capabilities must remain absent rather than becoming no-op placeholders.
