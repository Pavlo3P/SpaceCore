# ADR-020: Sets and metric projection

## Status

Proposed

## Context

Constrained problems recur throughout the `0.4.0` ergonomics study and the wider
roadmap: the non-negative spectral fit projected onto a cone (run-01 S19), sphere
and Stiefel feasibility (run-02 S13), and the box / simplex / ball constraints
that appear in everyday optimization. In every case the user hand-wrote the
projection, and on a non-Euclidean space that projection must respect the ambient
metric — the same correctness trap that ADR-019 records for proximal steps.

[ADR-019](019_everyday_toolbox.md) introduced `IndicatorFunctional(C)` and
`project_C`, but left `C` untyped: there is no object that owns a feasible set,
its ambient space, and its projector. This ADR adds that noun.

## Current design

There is no set abstraction. Constraints are expressed ad hoc, membership has no
common interface, and projection is hand-written per problem (S19 hand-rolled the
cone projection; S13 hand-wrote sphere and Stiefel projections). The indicator
functional and projection helpers reserved in ADR-019 have no typed `C` to range
over.

## Decision

Add a `Set` abstraction: a closed subset of an ambient space that owns its
ambient `Space` and provides a metric projection and optional membership.

```python
class Set:
    ambient: Space
    def project(self, x) -> Element        # nearest point in the ambient metric
    def contains(self, x) -> bool | None    # membership; None = cannot cheaply decide
```

Concrete sets cover the common cases: `NonnegativeOrthant(X)`, `Box(X, lo, hi)`,
`Ball(X, radius, center=None)`, `Simplex(X)`, affine sets / hyperplanes, and the
Jordan / second-order cones. `project` returns the nearest point under the
*ambient inner product* `X.inner`, not the Euclidean default unless the space is
Euclidean. As with the proximal primitive in ADR-019, the closed-form projection
is valid only for separable / diagonal metrics; where no closed form exists on a
non-diagonal metric the projection **raises** rather than returning an
approximate (wrong) point.

The set unifies the constraint surface reserved in ADR-019:
`project_C(x) == C.project(x)`, `IndicatorFunctional(C)` is the indicator `δ_C`,
and projection onto `C` *is* the proximal operator of that indicator. The three
must be mutually consistent.

### Scope boundary

A `Set` guarantees projection and optional membership only. It does **not** imply
tangent spaces, retractions, parallel transport, or manifold-optimization
protocols. A non-convex set such as the sphere or Stiefel manifold may supply a
projector (and thereby support projected steps), but that does not make it a
manifold object — Riemannian manifold support remains a demand-gated track (§7 of
the post-0.4.0 plan) and a tutorial concern per [ADR-018](018_external_optimizer_adapters.md).
Projection onto intersections of sets (which needs iterative schemes such as
Dykstra) is out of scope and would require its own decision.

## Rationale

A feasible set with a metric-correct projector is the missing noun behind
projected-gradient and proximal methods and behind constraint indicators.
Centralizing the projection in the set defuses the metric trap once, the same way
the optimizer adapters and the proximal primitive do. Modeling the
indicator / prox / projection identity explicitly keeps the optimization toolbox
coherent instead of growing three unrelated helpers.

## Alternatives considered

Keeping constraints as bare projection callables was rejected: it carries no
membership test, no ambient-space typing, and reproduces the metric-projection
trap at every call site. Modeling every set as a manifold was rejected as
overpromising — most constraints are plain closed convex sets needing only a
projector, and manifold protocols (tangent, retraction, transport) are a separate,
gated concern. Folding projection into `IndicatorFunctional` alone was rejected
because the set is the natural owner of both membership and projection, from which
the indicator and its prox derive.

## Consequences

The set abstraction depends on the space hierarchy ([ADR-003](003_space_hierarchy.md))
for its ambient space and on the geometry/Riesz design
([ADR-004](004_inner_product_and_geometry.md)) for the metric projection. The
`IndicatorFunctional(C)` and `project_C` reserved in [ADR-019](019_everyday_toolbox.md)
are defined over a `Set`. Projected-gradient and proximal call sites consume
`Set.project`. Closed-form projections share the diagonal-metric boundary rule
with the ADR-019 proximal primitive. Intersection projection, if ever needed, is a
later decision.

## Contributor invariants

- A `Set` owns an ambient `Space`; `project` returns an ambient element that is
  the nearest point under `X.inner`, not the Euclidean default unless the space is
  Euclidean.
- Closed-form projection is valid only for separable / diagonal metrics; a
  non-diagonal metric with no closed form must raise, never approximate (the same
  rule as the ADR-019 proximal primitive).
- `contains` may return `None` when membership cannot be cheaply decided.
- `C.project(x)` equals the proximal operator of `IndicatorFunctional(C)`; the
  set's projection and indicator must stay mutually consistent.
- A `Set` does not imply tangent spaces, retractions, or transport; manifold
  optimization is out of scope.
