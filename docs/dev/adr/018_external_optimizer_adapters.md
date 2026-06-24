# ADR-018: External optimizer adapters

## Status

Accepted (implemented)

## Context

SpaceCore is a substrate, not an optimization framework: per `vision.md` it
composes with mature external optimizers (SciPy, optax, jaxopt, BlackJAX) rather
than replacing them. The `0.4.0` ergonomics study (`docs/dev/ergonomics-run-01`
and `-run-02`) made external-optimizer interop the single most requested feature
(six of ten scenarios in run 2: S06, S10, S11, S13, S14, S15).

Two findings shape this ADR. First, every persona that handed a SpaceCore
objective to an external optimizer had to reinvent the same wrapper: marshal
elements to and from the flat or pytree representation the optimizer expects, and
expose `value`/`grad`. Second, and more dangerous, is a silent correctness trap:
SpaceCore gradients are metric (Riesz) gradients per [ADR-009](009_metric_adjoint.md)
and [ADR-010](010_functional_contract.md), but NumPy/JAX optimizers expect
*coordinate* gradients. On a non-Euclidean space, passing `F.grad(x)` directly is
mathematically wrong; run-01 S04 converged to the wrong fixed point and run-02
S11 measured a gradient-representation difference of `6.0e-01`. The correct
handoff is `X.riesz(F.grad(x))`, which no scenario discovered without difficulty.

## Current design

The `spacecore.optimize` subpackage ships the three committed adapters
(`minimize_scipy`, `line_search_scipy`, `minimize_optax`), re-exported from the
package root. A shared `coordinate_gradient(F, X, x)` helper performs the
`X.riesz(F.grad(x))` conversion once, centrally; the SciPy adapters flatten and
unflatten elements through `X.flatten`/`X.unflatten` and reject complex domains,
while the optax adapter passes the element pytree through unchanged and requires
a JAX-backed domain. The external optimizer owns the loop in every case.
`optax` is an optional dependency (`spacecore[optax]`), imported lazily.

Before this, there was no optimizer-interop surface: users hand-wrote `fun`/`jac`
closures, hand-flattened elements, and had to know to apply `X.riesz` on weighted
spaces — the silent metric-gradient trap the adapters now defuse.

## Decision

Ship a small set of thin adapter functions that take a SpaceCore `Functional`
and drive an external optimizer. The committed shipped surface is:

```python
minimize_scipy(F, x0, *, method="L-BFGS-B", jac=True, **kw)   # SciPy minimize fun/jac
minimize_optax(F, x0, optimizer, *, steps, callback=None)     # optax loop, pytree state
line_search_scipy(F, x, d, **kw)                              # SciPy line_search
```

Each adapter:

1. Evaluates the objective through `F.value` and the gradient through `F.grad`.
2. **Converts the metric gradient to a coordinate gradient with `X.riesz`
   before handing it to the optimizer.** On a Euclidean space this is the
   identity; on a weighted/non-Euclidean space it is mandatory. This conversion
   is the adapter's central responsibility.
3. Marshals between SpaceCore elements and the optimizer's representation —
   flatten/unflatten for SciPy's flat arrays; pytree pass-through for optax.
4. Owns no iteration logic: the external optimizer owns the loop, line search,
   and convergence; the adapter only translates the objective and geometry.

Per-library JAX and manifold seams — jaxopt LM / fixed-point / Anderson,
BlackJAX HMC mass matrices, pymanopt / manifold handoff — are **not** shipped as
adapters. They are delivered as tutorials, because each carries library-specific
ceremony (solver callbacks, `check_level="none"` under `jit`, manifold
projection/retraction) that does not generalize into a stable function.

## Rationale

Adapters as functions, not a framework, keep SpaceCore on the substrate side of
`vision.md`: the external optimizer remains the optimizer. SciPy and optax are
the two highest-leverage targets — the NumPy and JAX defaults — and both accept a
plain objective, so a single `Functional`-consuming adapter covers each. Putting
the `X.riesz` conversion inside the adapter defuses the metric-gradient trap once,
centrally, instead of in every user's wrapper.

## Alternatives considered

A native SpaceCore optimizer abstraction or solver protocol was rejected: it
duplicates mature ecosystems and contradicts `vision.md`. A generic `Problem`
wrapper in the pymanopt/Manopt style was rejected as more structure than the
substrate role warrants. Shipping adapters for jaxopt/BlackJAX/pymanopt was
rejected because the per-library surface is large and unstable; tutorials carry
that knowledge without committing public API. Leaving the metric handoff to user
code was rejected because the study proved it is a silent correctness trap.

## Consequences

Adapters depend on the functional contract ([ADR-010](010_functional_contract.md))
and the metric-adjoint/Riesz design ([ADR-009](009_metric_adjoint.md),
[ADR-004](004_inner_product_and_geometry.md)). They must document the information
lost at the boundary (structured elements, non-Euclidean geometry, backend
context). JAX adapters must document tracing constraints surfaced by the study —
notably that jitted external solves may require `Context(..., check_level="none")`
(run-02 S14). The `Functional`-input convention is shared with the proximal
toolbox in [ADR-019](019_everyday_toolbox.md); both rely on `F.grad` returning a
metric gradient.

## Contributor invariants

- Adapters convert a metric gradient to a coordinate gradient with `X.riesz`
  before passing it to a coordinate-gradient consumer; the Euclidean case is the
  identity, not a special-cased branch.
- Adapters do not own iteration, line search, or convergence — the external
  optimizer does.
- No new core optimizer or solver-protocol abstraction is introduced; adapters
  are functions over a `Functional` and its domain space.
- jaxopt, BlackJAX, and pymanopt seams stay as tutorials, not shipped adapters.
- Adapters document the geometry and structure lost at the external boundary.
