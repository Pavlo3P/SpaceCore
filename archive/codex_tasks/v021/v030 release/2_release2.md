# Follow-up Task: Enforce truthful construction of Euclidean elementwise Jordan spaces

## Goal

Close the final capability-classification gap before releasing SpaceCore `0.3.0`.

`EuclideanElementwiseJordanSpace` is currently publicly constructible with configurations that do not satisfy the Euclidean-Jordan contract, including complex dtypes and non-Euclidean inner products.

Every instance of `EuclideanElementwiseJordanSpace` must satisfy:

```text
real dtype
exactly EuclideanInnerProduct
```

## Phase 1 — Add direct-construction validation

Add constructor validation to:

```python
EuclideanElementwiseJordanSpace
```

Validate after context resolution and inner-product resolution.

Require:

```python
not ctx.is_complex_dtype
type(inner_product) is EuclideanInnerProduct
```

Use the project’s canonical dtype/backend helpers rather than NumPy-specific checks.

Raise clear `TypeError` or `ValueError` messages:

```text
EuclideanElementwiseJordanSpace requires a real dtype.
EuclideanElementwiseJordanSpace requires EuclideanInnerProduct.
```

This validation must run for:

```python
EuclideanElementwiseJordanSpace(...)
_convert(...)
JAX pytree reconstruction
deserialization/reconstruction helpers
```

## Phase 2 — Centralize the invariant

Add a reusable internal predicate or validator, for example:

```python
def _is_euclidean_elementwise_jordan_compatible(ctx, inner_product) -> bool:
    ...
```

or:

```python
def _validate_euclidean_elementwise_jordan(ctx, inner_product) -> None:
    ...
```

Use the same logic in:

```text
EuclideanElementwiseJordanSpace direct construction
ElementwiseJordanSpace factory dispatch
convert() redispatch
product/stacked capability detection
```

There must be one source of truth for this compatibility condition.

## Phase 3 — Preserve factory redispatch

Keep automatic dispatch behavior:

```python
ElementwiseJordanSpace(real_euclidean_ctx)
    -> EuclideanElementwiseJordanSpace
```

```python
ElementwiseJordanSpace(complex_ctx)
    -> non-Euclidean ElementwiseJordanSpace
```

```python
ElementwiseJordanSpace(real_ctx, weighted_inner_product)
    -> non-Euclidean ElementwiseJordanSpace
```

On conversion, reconstruct through the public dispatch path so capabilities are recomputed:

```python
real_euclidean.convert(complex_ctx)
    -> ElementwiseJordanSpace
```

```python
complex_space.convert(real_ctx)
    -> EuclideanElementwiseJordanSpace
```

only when the conversion succeeds and the resulting inner product is Euclidean.

## Phase 4 — Add regression tests

Add direct-construction tests:

```python
EuclideanElementwiseJordanSpace(real_ctx)
```

accepts.

```python
EuclideanElementwiseJordanSpace(complex_ctx)
```

rejects.

```python
EuclideanElementwiseJordanSpace(
    real_ctx,
    inner_product=WeightedInnerProduct(...),
)
```

rejects.

```python
EuclideanElementwiseJordanSpace(
    real_ctx,
    inner_product=CustomInnerProduct(...),
)
```

rejects unless it is exactly the supported Euclidean implementation.

Also test:

* direct constructor and factory produce equal valid spaces;
* conversion never leaves a stale Euclidean-Jordan class;
* JAX pytree reconstruction revalidates the invariant;
* product and stacked spaces do not inherit Euclidean-Jordan capability from an invalid reconstruction path;
* every directly constructible `EuclideanJordanAlgebraSpace` satisfies the compatibility identity numerically.

## Phase 5 — Audit specialized public classes

Audit every public specialized capability class for the same issue:

```text
Can users directly construct an instance that violates the capability promised by its class?
```

At minimum inspect:

```text
EuclideanElementwiseJordanSpace
HermitianSpace
private/public product Euclidean-Jordan implementations
private/public stacked Euclidean-Jordan implementations
```

Add constructor validation where a capability depends on dtype, geometry, shape, or component capabilities.

## Acceptance criteria

* Complex `EuclideanElementwiseJordanSpace` construction fails immediately.
* Weighted or custom-inner-product `EuclideanElementwiseJordanSpace` construction fails immediately.
* Valid real Euclidean direct construction succeeds.
* Factory dispatch and direct construction use one compatibility validator.
* `convert()` always recomputes the correct capability class.
* Pytree and reconstruction paths cannot bypass validation.
* Product and stacked capability dispatch remains truthful.
* Every directly constructible `EuclideanJordanAlgebraSpace` satisfies its defining compatibility identity.
* Full tests, audits, documentation build, wheel build, and installation smoke tests pass.
