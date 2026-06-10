# ADR-004: Inner product and geometry

## Status

Accepted

## Context

Adjoints, gradients, norms, and solver correctness depend on the chosen inner product. Treating geometry as a hidden implementation detail would make non-Euclidean spaces incorrect.

## Current design

`InnerProduct` is a separate geometry object with `inner`, `riesz`, `riesz_inverse`, `convert`, `validate_for`, and `is_euclidean`. `InnerProductSpace` delegates `inner`, `norm`, and Riesz maps to this geometry. Built-in geometries include Euclidean coordinate geometry and weighted diagonal geometry.

Spaces own a geometry instance. The same geometry type can be reused by different coordinate spaces and converted with the space context. Metric adjoints use Riesz maps, and linalg uses space inner products and norms. Euclidean spaces use identity Riesz maps; weighted spaces use multiplication/division by validated positive finite weights.

## Decision

Geometry remains a separate object instead of only methods on `Space`. It is part of the mathematical contract of a space and must be validated, converted, and preserved by spaces and operators.

## Rationale

A separate geometry object avoids duplicating inner-product implementations across space subclasses and makes metric changes explicit. It also lets matrix-backed operators implement the correct adjoint formula without guessing from concrete space types.

## Alternatives considered

Putting all inner-product code directly on each space was rejected because it couples geometry to storage classes and makes reuse harder. Treating non-Euclidean geometry as a solver option was rejected because adjoints and gradients would already be wrong before the solver sees them.

## Consequences

New geometries must provide Riesz maps if they are used with matrix-backed operators. Riesz maps should broadcast over leading batch axes for efficient batched adjoints. See [ADR-009](009_metric_adjoint.md) for adjoint details and [ADR-011](011_linalg_contract.md) for solver assumptions.

## Contributor invariants

- Geometry is part of mathematical correctness, not just implementation detail.
- `InnerProduct.validate_for(space)` must reject incompatible geometry/storage pairs.
- Non-Euclidean geometry used by coordinate-backed LinOps must provide usable Riesz maps.
- `norm(x)` must be induced by `inner(x, x)` and return a real magnitude.
