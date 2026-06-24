# ADR-019: Everyday functional and proximal toolbox

## Status

Accepted

## Context

The `0.4.0` ergonomics study found that SpaceCore composes well once the user
identifies the right object, but that the common daily objectives have no front
door. Every least-squares persona (run-01 S01, S04, S11) hand-expanded
`½‖Ax−b‖²` into `Q = A.H @ A` and `c = −Aᵀb` to fit `LinOpQuadraticForm`; run-01
S02 hand-wrote `Σ ρ log ρ` because there is no entropy functional; run-01 S01/S04
and run-02 S11 hand-wrote proximal/soft-threshold steps. The run-02 toolbox
recommendation was explicit: do **not** grow a deep `Functional` hierarchy; add a
small set of named helpers over the existing quadratic/composition machinery,
plus the few genuinely missing objects.

The proximal layer carries the study's most dangerous correctness finding
(run-01 S04): a hand-written Euclidean soft-threshold paired with a metric
gradient converged to the wrong fixed point (obj `0.26753` vs true `0.26558`).
A proximal step must be taken in the space metric; for a diagonal weighted metric
this is the per-coordinate threshold `τᵢ = λ/(2 ε wᵢ)`, which users do not derive
unaided.

## Current design

The toolbox ships as named constructors over the existing
[ADR-010](010_functional_contract.md) machinery, with no new core type
hierarchy:

- `least_squares(A, b, *, weights=None, scale=0.5)` returns a
  `LinOpQuadraticForm` (`Q = 2·scale·A.H @ A`, linear term `-2·scale·A.H(b)`,
  offset `scale·<b, b>_Y`); with the default `scale=0.5` this is `½‖Ax−b‖²` and
  `Q = A.H @ A`. Optional diagonal `weights` give the weighted objective
  `scale·<Ax−b, W(Ax−b)>_Y`.
- The battery functionals `SquaredL2NormFunctional`, `LpNormFunctional`
  (with the `L1NormFunctional` wrapper), `NegativeEntropyFunctional`,
  `KLDivergenceFunctional`, and `HuberFunctional` are thin coordinate
  functionals sharing a private `_CoordinateFunctional` base that Riesz-corrects
  the coordinate gradient once via `domain.riesz_inverse`, so every gradient is
  the metric gradient required by ADR-010.
- `SpectralLpNormFunctional` (with the `NuclearNormFunctional` wrapper) is the
  spectral analogue, applying the coordinate `p`-norm to a Jordan spectrum and
  reconstructing the spectral gradient through `from_spectrum`.
- `generalized_shrinkage(X, *, c, x0, eps, lam=0.0, nonneg=False)` is the
  closed-form forward–backward solver, with the named wrappers `prox_l1`,
  `prox_l2sq`, and `project_nonneg`. It folds the metric into the threshold
  (`τᵢ = λ/(2 ε wᵢ)` on a diagonal metric) and raises on a non-diagonal metric.

These constructors live in the `spacecore.functional.tools` subpackage and are
re-exported from `spacecore.functional` and the top-level `spacecore` namespace.

`IndicatorFunctional(C)` and `project_C` are **not** part of this implementation;
they range over the `Set` abstraction of [ADR-020](020_sets_and_projection.md)
and land with that ADR. The Set-free `project_nonneg` prox wrapper ships here.

## Decision

Add a small toolbox of named constructors over the existing machinery, plus one
closed-form proximal primitive. No new core `Functional` type hierarchy.

Battery functionals:

```python
least_squares(A, b, *, weights=None, scale=0.5) -> LinOpQuadraticForm
SquaredL2NormFunctional(X)                       # ½‖x‖²
LpNormFunctional(X, p)                            # ‖x‖_p, p >= 1
L1NormFunctional(X)                               # thin wrapper = LpNormFunctional(X, 1)
SpectralLpNormFunctional(X, p)                    # Schatten-p: ‖λ(X)‖_p over a Jordan spectrum
NuclearNormFunctional(X)                          # thin wrapper = SpectralLpNormFunctional(X, 1)
NegativeEntropyFunctional(X)                      # Σ xᵢ log xᵢ
KLDivergenceFunctional(target)                    # KL(x ‖ target)
IndicatorFunctional(C)  +  project_C              # C is a Set (ADR-020): box / simplex / nonneg / cone
HuberFunctional(X, delta)
```

`SpectralLpNormFunctional` is the spectral analogue of `LpNormFunctional`: the
coordinate `p`-norm of the [ADR-012](012_jordan_spectrum.md) Jordan spectrum
rather than of the coordinates (Schatten norms on a `HermitianSpace`; nuclear
norm at `p = 1`). It is a spectral function `f(λ(X))`, so its gradient is the
spectral function gradient `from_spectrum(∇f(λ), frame)` and it is built on the
`spectrum` / `spectral_decompose` / `from_spectrum` capabilities, not on backend
`eigh`. On an elementwise Jordan space (spectrum = coordinates) it coincides with
`LpNormFunctional`.

`least_squares` returns an existing `LinOpQuadraticForm`, not a new type. Norm
functionals are coordinate norms; `LpNormFunctional` is the general form and
`L1NormFunctional` is a discoverable wrapper. `SquaredL2NormFunctional` stays
distinct from `LpNormFunctional(X, 2)` because it is `½‖·‖₂²`, the form with the
clean shrinkage prox.

Proximal / projection primitive — one closed-form, separable solver of the
forward–backward subproblem, with named specializations:

```python
generalized_shrinkage(X, *, c, x0, eps, lam=0.0, nonneg=False) -> Element
#   argmin_x  <c, x>_X + eps * ||x - x0||²_X + lam * ||x||₁   (optionally x >= 0)
prox_l1(v, t, X)        # wrapper: soft-threshold
prox_l2sq(v, t, X)      # wrapper: shrinkage
project_nonneg(v, X)    # wrapper: nonnegative orthant
```

`c` is typed as a **metric gradient** (i.e. `F.grad(x)`), consistent with
[ADR-010](010_functional_contract.md). The primitive is metric-aware: under a
diagonal weighted metric `G = diag(w)` the per-coordinate threshold is
`τᵢ = λ/(2 ε wᵢ)`. The closed form is valid only for diagonal/Euclidean metrics;
on a non-diagonal metric the problem is not separable and the primitive **raises**
rather than returning a separable (wrong) answer.

`spectral_gap` / multi-eigenvalue `lanczos_smallest(k=...)` is a linalg-contract
extension under [ADR-011](011_linalg_contract.md), not part of this toolbox.

## Rationale

The study showed the demand is for discoverability, not for new abstractions, so
the toolbox is constructors and wrappers over what exists. One general shrinkage
primitive with named wrappers, rather than a zoo of independent prox helpers,
keeps the surface small while covering forward–backward splitting (S01),
ISTA/FISTA LASSO (S04), and non-negative fitting (S19). Putting the metric into
the primitive and refusing non-diagonal metrics turns the study's worst silent
correctness trap into either a correct result or an explicit error.

## Alternatives considered

A deep loss/functional class hierarchy was rejected against the `vision.md`
"keep SpaceCore small" principle and the explicit run-02 recommendation. A set of
independent, unrelated prox helpers was rejected in favor of the single
parametrized primitive plus thin wrappers. Silently returning the separable
result on a non-diagonal metric was rejected because it reproduces the S04 trap.
A new core type for least squares was rejected because `LinOpQuadraticForm`
already expresses it.

## Consequences

New functionals follow [ADR-010](010_functional_contract.md): `grad` returns a
domain element under the domain geometry. The proximal primitive and the metric
convention depend on [ADR-004](004_inner_product_and_geometry.md) and
[ADR-009](009_metric_adjoint.md). The `c`-is-a-metric-gradient convention is
shared with the optimizer adapters in [ADR-018](018_external_optimizer_adapters.md).
TV regularization, logistic/softplus/hinge losses, and Tikhonov helpers stay as
examples/tutorials, not shipped core surface. The `C` ranged over by
`IndicatorFunctional` and `project_C` is the `Set` abstraction defined in
[ADR-020](020_sets_and_projection.md); projection onto `C` is the prox of its
indicator.

## Contributor invariants

- Convenience constructors return existing types (`least_squares` returns a
  `LinOpQuadraticForm`); they do not introduce new core abstractions.
- Norm wrappers delegate to `LpNormFunctional`; `SquaredL2NormFunctional` remains
  distinct from the `p=2` norm.
- Proximal and projection operations are taken in the space metric. The
  closed-form path is valid only for separable/diagonal metrics and must raise on
  a non-diagonal metric, never approximate it.
- The shrinkage primitive's `c` argument is a metric gradient, consistent with
  `Functional.grad`.
- New functionals satisfy the [ADR-010](010_functional_contract.md) gradient
  contract.
