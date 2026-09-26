# Task: Finish ProductSpace structured-element documentation cleanup

The previous docs cleanup mostly landed, but two stale tuple-only API documentation issues remain. Fix them so the public API consistently reflects structured `ProductSpace` elements.

## Problem

`ProductSpace` no longer necessarily uses tuple elements. Tuple is only the default representation. Structured pytrees/dataclasses are supported through the product structure adapter.

However, some docstrings/type annotations still imply tuple-only product elements.

Known issues:

1. `ProductStructureCheck` docstring is stale:

```python
Check that a product-space value is a tuple of the right length.
```

This is now wrong. It should describe validation against the configured `ProductSpace` structure.

2. Public `ProductSpace` method annotations still advertise tuple-only inputs/outputs, for example:

```python
def zeros(self) -> Tuple[Any, ...]
def add(self, x: Tuple[Any, ...], y: Tuple[Any, ...]) -> Tuple[Any, ...]
def unflatten(self, v: DenseArray) -> Tuple[Any, ...]
```

These are now misleading because the return type may be a dataclass/pytree product element.

## Required fix

### 1. Fix stale checker docstring

Update `ProductStructureCheck` wording to something like:

```python
Check that a value is a valid product element for the configured ProductSpace structure.
```

Do not mention tuple unless explicitly discussing `TupleStructure`.

### 2. Fix `ProductSpace` type annotations

Replace tuple-only annotations on public methods with representation-neutral ones.

Preferred minimal fix:

```python
ProductElement = Any
```

Then use:

```python
def zeros(self) -> ProductElement: ...
def ones(self) -> ProductElement: ...
def add(self, x: ProductElement, y: ProductElement) -> ProductElement: ...
def scale(self, a: ScalarLike, x: ProductElement) -> ProductElement: ...
def inner(self, x: ProductElement, y: ProductElement) -> Scalar: ...
def flatten(self, x: ProductElement) -> DenseArray: ...
def unflatten(self, v: DenseArray) -> ProductElement: ...
def riesz(self, x: ProductElement) -> ProductElement: ...
def riesz_inverse(self, x: ProductElement) -> ProductElement: ...
```

Apply the same idea to batch methods if they still say `Tuple[...]`.

Do not over-engineer generics unless the project already has a clear typing pattern for element representations.

## Constraints

* Documentation/type-hint cleanup only.
* Do not change runtime behavior.
* Do not change tuple default behavior.
* Do not weaken validation.
* Do not touch solver logic.

## Tests/checks

Run:

```bash
pytest tests/spaces/test_product_structure.py
pytest tests/linops/test_product_structure.py
```

If available, also run existing type/docstring/public API checks.

If `scripts/docstring_audit.py` fails only because `numpydoc` is missing, report that explicitly rather than treating it as a code failure.

## Acceptance criteria

* No public `ProductSpace` method annotation claims tuple-only elements unless the method specifically belongs to `TupleStructure`.
* `ProductStructureCheck` docstring is representation-neutral.
* Existing structured-product tests still pass.
* No implementation behavior changes.
