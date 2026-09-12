# ADR-010: Functional contract

## Status

Accepted

## Context

Functionals are scalar-valued maps on spaces. They share context and validation mechanics with LinOps but have different mathematical contracts.

## Current design

`Functional` owns a domain space, a scalar `Field` codomain, and a context. The codomain defaults to `domain.scalars`; an explicit `cod=` may independently select a real or complex field. Subclasses implement `value(x)`. `vvalue(xs)` evaluates over a leading batch axis through backend `vmap` unless overridden. `LinearFunctional` adds a Riesz-gradient contract through `representer`, `grad`, and `vgrad`. `InnerProductFunctional` stores a representer and evaluates `<c, x>`. `MatrixFreeLinearFunctional` trusts user evaluation callables and has no stored representer. `QuadraticForm` defines optional `grad`, `vgrad`, and `hess_apply`; `LinOpQuadraticForm` stores a Hermitian operator, optional linear term, and scalar offset. Pull-backs are built by composing with a LinOp and have specializations for inner-product and quadratic functionals.

## Decision

A functional is not a LinOp with a one-dimensional codomain. It owns a domain, a `Field` codomain and scalar-valued evaluation, and gradients are space elements with respect to the domain geometry when implemented.

`Field` inherits `InnerProductSpace` (and therefore `VectorSpace`), but not
`CoordinateSpace`. Since `LinOp` now accepts `VectorSpace` endpoints, the type
bound alone cannot maintain this separation. `LinOp.__init__` explicitly rejects
a `Field` codomain with an error citing ADR-010. Use a `Functional` for that map.

Scalar output validation uses `out_space="codomain"` and the field's
`ScalarShapeCheck`, active only at `standard` and `strict`. Exact batch output
validation deliberately retains `out_batched_scalar`: ordinary batched space
membership cannot require exactly `(N,)`. The matrix-free explicit batch
callback retains its existing check of all leading axes (ADR-006).

## Rationale

Scalar-valued objectives need gradients, Hessian actions, and pull-backs that are clearer as a separate abstraction than as degenerate linear operators.

## Alternatives considered

Representing all functionals as LinOps into a scalar space was rejected because it obscures Riesz gradients and quadratic objective structure. Requiring every functional to expose a gradient was rejected because matrix-free scalar maps may only support value evaluation.

## Consequences

New functionals must validate domain inputs and scalar output shape when checks are enabled. Gradients must be Riesz gradients satisfying the domain inner-product convention. Pull-back specializations may use `A.H` and therefore depend on [ADR-009](009_metric_adjoint.md).

## Contributor invariants

- `value` returns a scalar-like backend value in its `Field` codomain.
- Codomain conversion and pytree reconstruction preserve the mathematical field; a complex codomain must not promote a real domain to complex storage.
- `grad` returns a domain element when implemented.
- Matrix-free functionals do not imply a stored representer.
- Functional composition requires `A.codomain == F.domain`.
