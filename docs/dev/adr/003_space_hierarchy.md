# ADR-003: Space hierarchy

## Status

Accepted

## Context

SpaceCore represents mathematical spaces separately from raw array containers. Contributors need to know which layer owns linear operations, coordinates, membership, and element construction.

## Current design

`Space` owns context and membership checks. `VectorSpace` adds abstract `zeros`, `add`, `scale`, and `axpy`. `CoordinateSpace` adds finite coordinate `shape`, `size`, `flatten`, `unflatten`, batch flattening, and `stacked`. `InnerProductSpace`, `StarSpace`, `JordanAlgebraSpace`, and `EuclideanJordanAlgebraSpace` add optional mathematical capabilities.

Concrete dense coordinate spaces construct zeros through their context, validate backend/shape/dtype, and flatten to a dense coordinate vector. `DenseVectorSpace` specializes one-dimensional dense coordinates. `TreeSpace` is a coordinate space whose elements are Python trees of coordinate-space leaves. `StackedSpace` represents a fixed leading-axis stack as one mathematical element.

## Decision

A space defines valid elements and operations on those elements; it is not just an array shape. Coordinate representation is exposed through `flatten`/`unflatten` for operators and linalg, while mathematical operations use `zeros`, `add`, `scale`, and capability methods.

## Rationale

Separating mathematical spaces from coordinate arrays lets product and pytree-structured elements participate in the same operator and solver contracts as dense vectors.

## Alternatives considered

Using only array shapes was rejected because it cannot represent product structure, custom geometry, Hermitian membership, or future tree spaces. Making all spaces dense arrays was rejected because product and structured elements need non-array containers.

## Consequences

New spaces must decide which capabilities they implement and must provide coordinate flattening if they are `CoordinateSpace`s. LinOps and solvers should depend on space methods, not on raw container assumptions. See [ADR-005](005_space_subclasses_and_capabilities.md) and [ADR-006](006_current_batching_model.md).

## Contributor invariants

- `Space` membership is the source of truth for valid elements.
- `VectorSpace` linear operations must preserve membership.
- `CoordinateSpace.flatten` and `unflatten` must be inverse coordinate representations for one element.
- Tree element structure must be handled through the `TreeSpace` optree definition, not by assuming tuples everywhere.
