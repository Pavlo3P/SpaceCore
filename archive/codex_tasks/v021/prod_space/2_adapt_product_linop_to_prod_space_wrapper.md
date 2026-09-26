# Task: Make product LinOps compatible with tree-structured ProductSpace elements

`ProductSpace` now supports non-tuple element representations through structure adapters. However, product linear operators still assume tuple/list indexing and return tuples directly. This breaks `ProductSpace.from_template(...)` with dataclass/pytree elements.

## Problem

The following product LinOps still hard-code tuple semantics:

* `BlockDiagonalLinOp`
* `StackedLinOp`
* `SumToSingleLinOp`
* any other `spacecore/linop/product/*` class using `x[i]`, `tuple(...)`, or `zip(..., x)` directly

Observed failures:

* `BlockDiagonalLinOp.apply(State(...))` fails because `State` is not subscriptable.
* `StackedLinOp.apply(...)` returns a tuple even when codomain expects a dataclass element.
* `SumToSingleLinOp` has analogous tuple assumptions.

## Required fix

Route all product LinOp component access through the associated `ProductSpace` structure boundary.

Use helpers already present on `ProductSpace`, such as:

```python
parts = product_space._components(x)
x = product_space._from_components(parts)
```

Do not introduce a second structure abstraction inside LinOp.

## Rules

* If a LinOp domain is a `ProductSpace`, unpack inputs through `dom._components(x)`.
* If a LinOp codomain is a `ProductSpace`, repack outputs through `cod._from_components(parts)`.
* Scalar or single-space outputs should remain unchanged.
* Tuple behavior must remain identical.
* `flatten`/`unflatten` and solvers must remain untouched.
* Batch semantics remain “structure of batched components,” not “batch of structures.”

## Tests

Add `tests/linop/test_product_structure.py`.

Test with both tuple elements and a registered pytree/dataclass element.

Cover:

* `BlockDiagonalLinOp.apply`
* `BlockDiagonalLinOp.rapply` if applicable
* `StackedLinOp.apply`
* `SumToSingleLinOp.apply`
* adjoint/H behavior if those classes expose it
* NumPy and JAX contexts where supported

Acceptance criteria:

* Product LinOps accept dataclass/pytree elements when their domain is a structured `ProductSpace`.
* Product LinOps return dataclass/pytree elements when their codomain is a structured `ProductSpace`.
* Tuple path remains unchanged.
* Full relevant test suite passes.
