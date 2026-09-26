# Follow-up Task: Complete the Space hierarchy refactor and reorganize it into folders

## Goal

Finish the hierarchy refactor by:

1. making capability inheritance truthful;
2. separating generic coordinate spaces from Euclidean Jordan spaces;
3. completing the repository-wide migration from the old concrete `VectorSpace`;
4. reorganizing the space implementation into clear folders by responsibility.

## Phase 1 — Reorganize the space package

Restructure `spacecore/space/` approximately as:

```text
spacecore/space/
├── base/
│   ├── _space.py
│   ├── _vector.py
│   ├── _coordinate.py
│   ├── _inner_product.py
│   ├── _star.py
│   └── _jordan.py
├── concrete/
│   ├── _dense_vector.py
│   ├── _dense_coordinate.py
│   ├── _hermitian.py
│   ├── _product.py
│   └── _stacked.py
├── checks/
│   ├── _base.py
│   ├── _coordinate.py
│   └── _product.py
└── __init__.py
```

Keep public imports stable through `spacecore.space` and top-level `spacecore`.

## Phase 2 — Fix capability truthfulness

Update concrete inheritance so classes only expose capabilities they always support.

### `ProductSpace`

Make `ProductSpace` inherit only the capabilities guaranteed for every product:

```python
class ProductSpace(CoordinateSpace):
    ...
```

Add explicit specialized product classes or constructors for stronger contracts:

```python
ProductInnerProductSpace
ProductStarSpace
ProductJordanAlgebraSpace
ProductEuclideanJordanAlgebraSpace
```

Each specialized class must validate at construction that every component supports the required capability.

Examples:

```python
ProductStarSpace(spaces)
```

requires:

```python
all(isinstance(space, StarSpace) for space in spaces)
```

```python
ProductEuclideanJordanAlgebraSpace(spaces)
```

requires every component to be a `EuclideanJordanAlgebraSpace`.

Implement operations componentwise.

### `StackedSpace`

Apply the same principle:

```python
class StackedSpace(CoordinateSpace):
    ...
```

Add specialized stacked variants or factory dispatch based on the base-space capability:

```python
StackedInnerProductSpace
StackedStarSpace
StackedJordanAlgebraSpace
StackedEuclideanJordanAlgebraSpace
```

The returned stacked class must expose exactly the capabilities supported by the base space.

Add tests checking both:

```python
isinstance(space, Capability)
```

and successful method execution.

## Phase 3 — Separate dense coordinate and Jordan behavior

Refactor `DenseCoordinateSpace` so it provides only generic finite-coordinate behavior.

Its baseline inheritance should be:

```python
class DenseCoordinateSpace(
    CoordinateSpace,
    InnerProductSpace,
):
    ...
```

Create a separate concrete class for elementwise Jordan structure, for example:

```python
class ElementwiseJordanSpace(
    DenseCoordinateSpace,
    StarSpace,
    EuclideanJordanAlgebraSpace,
):
    ...
```

Use this class only when its inner product is compatible with the elementwise Jordan product.

Validate compatibility at construction, or restrict supported inner products to explicitly compatible implementations.

Keep:

```python
DenseVectorSpace
```

as the plain one-dimensional specialization. Decide explicitly whether it is:

```python
DenseVectorSpace(DenseCoordinateSpace)
```

or:

```python
DenseVectorSpace(ElementwiseJordanSpace)
```

based on the guaranteed inner-product contract.

Add tests for real, complex, Euclidean, and weighted cases.

## Phase 4 — Define inner-product compatibility

Add an explicit capability or validation mechanism for Jordan-compatible inner products.

For each `EuclideanJordanAlgebraSpace`, test:

```python
inner(jordan(x, y), z)
==
inner(y, jordan(x, z))
```

within backend tolerance.

Reject or avoid constructing `EuclideanJordanAlgebraSpace` instances when this identity is not guaranteed.

Document which inner products are compatible with:

* elementwise Jordan multiplication;
* Hermitian Jordan multiplication;
* product Jordan algebras.

## Phase 5 — Complete repository-wide migration

Replace every remaining concrete use of:

```python
VectorSpace(...)
```

with the appropriate concrete class:

```python
DenseVectorSpace(...)
DenseCoordinateSpace(...)
ElementwiseJordanSpace(...)
```

Search and update:

```text
README.md
docs/
examples/
benchmarks/
scripts/
tutorials/
tests/
notebooks/
type annotations
docstrings
```

Keep `VectorSpace` exclusively as the abstract mathematical base.

Add a repository test that rejects direct instantiation patterns such as:

```python
VectorSpace(...)
```

outside intentional abstract-class tests.

## Phase 6 — Update consumers and runtime checks

Audit all consumers and require the weakest accurate capability:

```text
membership-only code              -> Space
linear operations                 -> VectorSpace
coordinate materialization        -> CoordinateSpace
inner products and adjoints       -> InnerProductSpace
star operations                   -> StarSpace
spectral/Jordan operations        -> JordanAlgebraSpace
Euclidean Jordan algorithms       -> EuclideanJordanAlgebraSpace
plain-vector optimized operations -> DenseVectorSpace
```

Replace late `AttributeError` failures with early, explicit capability errors.

## Tests

Add tests covering:

1. `ProductSpace` exposes only baseline coordinate capability.
2. Specialized product classes validate component capabilities.
3. `StackedSpace` capability dispatch matches the base space.
4. Generic `DenseCoordinateSpace` does not claim Jordan structure.
5. Euclidean Jordan compatibility holds for every declared Euclidean Jordan space.
6. Incompatible weighted complex coordinate spaces are rejected or classified without Jordan capability.
7. No repository example directly instantiates abstract `VectorSpace`.
8. Public imports remain stable.
9. Full test suite passes.
10. Documentation builds without stale class names.

## Acceptance criteria

* Capability inheritance is truthful for every concrete space.
* Product and stacked spaces expose only supported operations.
* `DenseCoordinateSpace` represents generic coordinates without overstating algebraic structure.
* Every `EuclideanJordanAlgebraSpace` satisfies the compatibility identity.
* All old concrete `VectorSpace(...)` usages are migrated.
* The implementation is organized into folders by abstraction and responsibility.
* Full tests and documentation build pass.
