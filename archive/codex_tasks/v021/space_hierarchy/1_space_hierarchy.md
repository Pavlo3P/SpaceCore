# Task: Refactor the SpaceCore space hierarchy

## Goal

Split the current monolithic `Space` abstraction into composable mathematical capabilities.

The resulting hierarchy must distinguish:

* general spaces;
* abstract vector spaces;
* finite coordinate spaces;
* inner-product spaces;
* spaces with a canonical star operation;
* Jordan-algebra spaces;
* concrete plain dense vectors.

## Target classes

Implement:

```python
Space
VectorSpace
CoordinateSpace
InnerProductSpace
StarSpace
JordanAlgebraSpace
EuclideanJordanAlgebraSpace
DenseVectorSpace
```

## Phase 1 — Refactor `Space`

Keep on `Space`:

```python
ctx
checks
member_checks()
_check_member(x)
check_member(x)
_convert(new_ctx)
```

Move the existing linear, coordinate, metric, batching, and spectral methods into the capability classes below.

## Phase 2 — Add abstract `VectorSpace`

Create:

```python
class VectorSpace(Space):
    @abstractmethod
    def zeros(self): ...

    @abstractmethod
    def add(self, x, y): ...

    @abstractmethod
    def scale(self, a, x): ...

    def axpy(self, a, x, y):
        return self.add(self.scale(a, x), y)
```

Update `LinOp` domains and codomains, functionals, and generic linear algorithms to use `VectorSpace`.

## Phase 3 — Add `CoordinateSpace`

Create:

```python
class CoordinateSpace(VectorSpace):
    shape: tuple[int, ...]

    @property
    def size(self) -> int: ...

    @abstractmethod
    def flatten(self, x): ...

    @abstractmethod
    def unflatten(self, v): ...

    def flatten_batch(self, xs): ...
    def unflatten_batch(self, vs): ...
    def add_batch(self, x, y): ...
    def scale_batch(self, a, x): ...
    def stacked(self, count): ...
```

Move into `CoordinateSpace` the existing implementations of:

```python
shape
size
flatten_batch
unflatten_batch
add_batch
scale_batch
stacked
```

Use `CoordinateSpace` for all finite coordinate representations, including:

```python
(n,)
(m, n)
(d1, ..., dk)
```

Update shape-dependent checks and coordinate-based algorithms to require `CoordinateSpace`.

## Phase 4 — Rename the current concrete `VectorSpace`

Rename the current concrete plain-vector implementation to:

```python
DenseVectorSpace
```

Restrict `DenseVectorSpace` elements to one-dimensional dense arrays:

```python
shape == (n,)
```

Move generic multidimensional dense-array behavior into `CoordinateSpace` or an internal reusable implementation helper.

Update imports, tests, examples, documentation, and annotations:

```text
VectorSpace       = abstract mathematical vector space
CoordinateSpace   = finite coordinate vector space
DenseVectorSpace  = concrete plain one-dimensional dense vectors
```

## Phase 5 — Add `InnerProductSpace`

Create:

```python
class InnerProductSpace(VectorSpace):
    geometry: InnerProduct

    def inner(self, x, y):
        return self.geometry.inner(self.ops, x, y)

    def riesz(self, x):
        return self.geometry.riesz(self.ops, x)

    def riesz_inverse(self, x):
        return self.geometry.riesz_inverse(self.ops, x)

    def norm(self, x):
        value = self.ops.real(self.inner(x, x))
        return self.ops.sqrt(value)

    @property
    def is_euclidean(self):
        return self.geometry.is_euclidean
```

Move into `InnerProductSpace`:

```python
geometry
inner
riesz
riesz_inverse
norm
is_euclidean
```

Update metric-aware adjoints and metric-dependent algorithms to require `InnerProductSpace`.

## Phase 6 — Add `StarSpace`

Create:

```python
class StarSpace(Space):
    @abstractmethod
    def star(self, x): ...
```

Implement `star` for supported spaces:

```python
DenseVectorSpace.star(x)       # elementwise conjugation
HermitianSpace.star(x)         # identity
ProductSpace.star(x)           # componentwise
StackedSpace.star(x)           # delegate to base space
```

For a coordinate space of square dense arrays, implement conjugate transpose through the appropriate concrete structured space.

Add tests for:

```python
star(star(x)) == x
```

and conjugate-linearity.

## Phase 7 — Add Jordan-algebra capabilities

Create:

```python
class JordanAlgebraSpace(VectorSpace):
    @abstractmethod
    def jordan(self, x, y): ...

    @abstractmethod
    def spectrum(self, x): ...

    @abstractmethod
    def spectral_decompose(self, x): ...

    @abstractmethod
    def from_spectrum(self, eigvals, frame): ...

    def spectral_apply(self, x, f): ...
```

Move the current spectral API from `Space` into `JordanAlgebraSpace`.

Rename:

```python
apply(x, f)
```

to:

```python
spectral_apply(x, f)
```

Create:

```python
class EuclideanJordanAlgebraSpace(
    JordanAlgebraSpace,
    InnerProductSpace,
):
    ...
```

Add tests for:

```python
inner(jordan(x, y), z)
==
inner(y, jordan(x, z))
```

## Phase 8 — Update concrete inheritance

Update concrete classes approximately as follows:

```python
class DenseVectorSpace(
    CoordinateSpace,
    InnerProductSpace,
    StarSpace,
    EuclideanJordanAlgebraSpace,
):
    ...
```

```python
class HermitianSpace(
    CoordinateSpace,
    InnerProductSpace,
    StarSpace,
    EuclideanJordanAlgebraSpace,
):
    ...
```

Other dense multidimensional spaces should inherit directly from the capabilities they support, primarily `CoordinateSpace` and optionally `InnerProductSpace`, `StarSpace`, or `JordanAlgebraSpace`.

Update `ProductSpace` and `StackedSpace` to delegate supported capability operations to their components or base space.

## Phase 9 — Update consumers

Use the weakest required capability:

```text
membership only                  -> Space
linear operations                -> VectorSpace
flattening / dense coordinates   -> CoordinateSpace
metric-aware operations          -> InnerProductSpace
spectral operations              -> JordanAlgebraSpace
plain-vector optimizations       -> DenseVectorSpace
```

Update runtime checks, annotations, and error messages accordingly.

## Tests

Add tests covering:

1. Minimal `Space` membership behavior.
2. Abstract `VectorSpace` without coordinates.
3. `CoordinateSpace` with vector, matrix-shaped, and tensor-shaped elements.
4. Coordinate flattening, batching, shape, size, and stacking.
5. `DenseVectorSpace((n,))`.
6. Rejection of multidimensional `DenseVectorSpace` shapes.
7. Non-coordinate `InnerProductSpace`.
8. Star involution and conjugate-linearity.
9. Jordan product and spectral operations.
10. Product and stacked delegation.
11. Consumer capability requirements.
12. Existing numerical behavior.
13. Full suite and documentation build.

## Acceptance criteria

* `Space` owns general membership and context behavior.
* `VectorSpace` owns abstract linear operations.
* `CoordinateSpace` supports arbitrary finite coordinate shapes.
* `DenseVectorSpace` represents only plain one-dimensional dense vectors.
* `InnerProductSpace` owns geometry and Riesz maps.
* `StarSpace` owns canonical involution.
* `JordanAlgebraSpace` owns Jordan and spectral functionality.
* Existing consumers use the weakest appropriate capability.
* Full tests and documentation build pass.
