# Release Task: Complete SpaceCore 0.3.0

## Goal

Prepare the repository for the **0.3.0** release.

This is a breaking minor release within the unstable `0.x` series. The space hierarchy was refactored into capability-based abstractions, `VectorSpace` became an abstract base, the Jordan/spectral contract replaced `eigh`, and product spaces gained structured-element support.

The release must:

* correct capability classification;
* recompute capabilities after context conversion;
* fix structured-product spectral reconstruction;
* validate inner-product implementations;
* consolidate product and stacked construction;
* document every breaking change with a migration path;
* verify source and built artifacts.

The central invariant is:

> A space’s capabilities must be derived from its actual structure, dtype, and inner product. Capabilities must never be asserted incorrectly or retained stale after `convert()`.

## Priority 1 — Versioning and release metadata

Set the release version to:

```python
# spacecore/_version.py
__version__ = "0.3.0"
```

Configure `pyproject.toml` to read the version from this single source of truth.

Update:

```text
pyproject.toml
CHANGELOG.md
docs/source/release_notes.rst
spacecore.__version__
```

The changelog must contain:

### Added

Document:

* `spectrum`;
* `spectral_decompose`;
* `from_spectrum`;
* Jordan capability hierarchy;
* `ProductStructure`;
* `ProductSpace.from_template`;
* `StackedSpace`;
* vectorizable axis-aware checks.

### Changed and Removed

Document:

* `VectorSpace` is now abstract;
* the previous concrete `VectorSpace` use cases moved to `DenseVectorSpace` and `DenseCoordinateSpace`;
* `eigh` was removed;
* specialized public product/stacked constructors were replaced by auto-dispatching factories.

### Migration

Add an explicit old-to-new migration table.

Place the `eigh` replacement prominently near the top:

```text
space.eigh(x)
    -> space.spectral_decompose(x)  # eigenvalues and frame
    -> space.spectrum(x)            # eigenvalues only
```

Verify:

```text
pytest from source tree
pytest against installed wheel
spacecore.__version__ == "0.3.0"
wheel filename contains spacecore-0.3.0
```

## Priority 2 — Correct elementwise Jordan capability classification

### Problem

For complex coordinates with the sesquilinear inner product

```text
inner(x, y) = sum(conj(x) * y)
```

and elementwise Jordan product

```text
jordan(x, y) = x * y
```

the Euclidean-Jordan compatibility identity does not generally hold:

```python
inner(jordan(x, y), z) == inner(y, jordan(x, z))
```

Therefore complex elementwise spaces must not claim `EuclideanJordanAlgebraSpace`.

### Refactor

Implement the following separation:

```python
class DenseVectorSpace(
    DenseCoordinateSpace,
    StarSpace,
):
    """Plain one-dimensional dense vectors with no Jordan capability by default."""
```

`DenseVectorSpace` must support valid Euclidean, weighted, and custom inner products.

Users requiring elementwise Jordan structure must construct:

```python
class ElementwiseJordanSpace(
    DenseCoordinateSpace,
    StarSpace,
    JordanAlgebraSpace,
):
    """Elementwise Jordan algebra for real or complex coordinates."""
```

Add:

```python
class EuclideanElementwiseJordanSpace(
    ElementwiseJordanSpace,
    EuclideanJordanAlgebraSpace,
):
    """Real elementwise Euclidean Jordan algebra."""
```

Select `EuclideanElementwiseJordanSpace` only when both hold:

```text
dtype is real
inner product is EuclideanInnerProduct
```

Verify that its MRO is valid and that the correct `InnerProductSpace`, `JordanAlgebraSpace`, and `VectorSpace` implementations are used.

### Capability redispatch on conversion

Every `convert(new_ctx)` implementation must reconstruct through the appropriate public factory or dispatch mechanism.

For example:

```python
real_euclidean_jordan.convert(complex_ctx)
```

must return an `ElementwiseJordanSpace` without the Euclidean-Jordan capability.

Converting from complex to real may be invalid or lossy. Require restoration of the Euclidean-Jordan capability only when:

* conversion succeeds;
* the resulting dtype is genuinely real;
* the resulting inner product is Euclidean.

### Tests

Verify:

1. complex `ElementwiseJordanSpace` is not an `EuclideanJordanAlgebraSpace`;
2. real Euclidean elementwise spaces are;
3. the compatibility identity holds numerically for every declared Euclidean elementwise Jordan space;
4. real-to-complex conversion removes the Euclidean-Jordan capability;
5. successful valid conversion to a real Euclidean context restores it;
6. `DenseVectorSpace` exposes no Jordan methods by default.

## Priority 2b — Fix structured `ProductSpace` spectral decomposition

### Problem

`ProductSpace.spectral_decompose` currently routes per-component spectral data through the product’s element-structure adapter.

This is invalid for `PytreeStructure`, because spectral data is not an element of the product space and does not share the element pytree structure.

### Add an explicit spectral-data type

Introduce an internal value type such as:

```python
@dataclass(frozen=True)
class ProductSpectralDecomposition:
    eigvals: tuple[Any, ...]
    frames: tuple[Any, ...]
```

Use it as the return type of:

```python
ProductSpace.spectral_decompose(x)
```

Implement:

```python
ProductSpace.from_spectrum(decomposition)
```

or an equivalently explicit signature consuming the same structured value.

Spectral decomposition must:

* call `spectral_decompose` independently for each component;
* store component eigenvalues and frames in plain ordered tuples;
* avoid `_from_components`, `to_components`, `tree_unflatten`, and all element-structure adapters;
* validate component count during reconstruction.

Remove heuristic handling based on tuple detection or dense-array detection.

### Tests

Add round-trip tests:

```python
decomposition = product.spectral_decompose(x)
reconstructed = product.from_spectrum(decomposition)
```

Cover:

* `TupleStructure`;
* `ProductSpace.from_template` using `PytreeStructure`;
* a mixed elementwise-vector and Hermitian product;
* NumPy;
* JAX;
* JAX pytree registration and JIT-compatible usage where applicable.

## Priority 3 — Validate inner-product implementations

### Add validation contract

Add:

```python
class InnerProduct:
    def validate_for(self, space: InnerProductSpace) -> None:
        ...
```

Call this validation:

* when constructing an `InnerProductSpace`;
* after converting the inner product;
* while reconstructing a space during `convert()`.

Specific inner products may require stronger capabilities.

For example:

```python
class WeightedInnerProduct(InnerProduct):
    def validate_for(self, space: InnerProductSpace) -> None:
        if not isinstance(space, CoordinateSpace):
            raise TypeError(...)
```

### Weighted inner-product validation

Validate that weights:

* have exactly the coordinate core shape;
* have no leading batch dimensions;
* are real;
* are finite;
* are strictly positive;
* use a compatible backend/context;
* remain valid after conversion.

Raise clear construction-time errors.

### Tests

Cover:

```text
negative weights
zero weights
complex weights
non-finite weights
incorrect shape
batched weights
backend/context mismatch
valid positive weights
successful context conversion
failed validation after conversion
```

## Priority 4 — Consolidate product and stacked construction

Make these the only public constructors:

```python
ProductSpace(...)
StackedSpace(...)
```

They must compute and preserve the complete valid capability set.

Move specialized implementation classes to private names:

```python
_ProductInnerProductSpace
_ProductStarSpace
_ProductJordanAlgebraSpace
_ProductEuclideanJordanAlgebraSpace
...

_StackedInnerProductSpace
_StackedStarSpace
_StackedJordanAlgebraSpace
_StackedEuclideanJordanAlgebraSpace
...
```

### Dispatch design

Use:

* one capability-detection helper;
* one product intersection rule;
* one stacked identity rule;
* one `capability_set -> concrete class` lookup.

Specialized private classes must contain only their declared mixin composition and capability-specific method implementations. Keep dispatch deterministic and inspectable.

### Conversion

`convert()` must:

1. convert component or base spaces;
2. recompute their capabilities;
3. reconstruct through `ProductSpace(...)` or `StackedSpace(...)`;
4. select the new runtime class from the recomputed capability set.

### Equality

Equivalent mathematical spaces must compare equal regardless of the internal construction path.

Verify equality after:

* construction;
* conversion;
* product reconstruction;
* stacked reconstruction;
* pytree round-trip.

## Priority 5 — Strengthen capability-registry tests

Add parameterized tests over every supported capability combination.

For products:

```python
capabilities(product)
==
intersection(capabilities(component) for component in components)
```

For stacks:

```python
capabilities(stacked)
==
capabilities(base)
```

Cover:

* coordinate-only;
* inner-product;
* star;
* Jordan;
* Euclidean-Jordan;
* all valid combinations;
* real and complex dtype;
* Euclidean and weighted inner products;
* Hermitian spaces;
* elementwise Jordan spaces;
* products and nested products;
* stacks and nested stacks.

For every combination test:

```text
construction
method availability
conversion and capability redispatch
equality
JAX pytree round-trip
stacked(...)
ProductSpace.from_template(...)
```

Unsupported methods must be absent through normal attribute lookup.

## Priority 6 — API and documentation cleanup

Update:

```text
README.md
docs/source/api/spaces.rst
docs/source/tutorials/spaces.rst
docs/source/design/geometry.rst
docs/source/release_notes.rst
tutorial notebooks
CHANGELOG.md
```

Document the final class meanings:

```text
DenseCoordinateSpace
    generic finite dense coordinates of any shape

DenseVectorSpace
    plain one-dimensional dense vectors; no Jordan capability by default

ElementwiseJordanSpace
    elementwise Jordan algebra for real or complex coordinates

EuclideanElementwiseJordanSpace
    real Euclidean elementwise Jordan algebra

HermitianSpace
    Hermitian Euclidean Jordan algebra
```

Add the migration table:

```text
0.2.x                          0.3.0
-----------------------------  ---------------------------------------------
sc.VectorSpace((n,))           sc.DenseVectorSpace((n,))
sc.VectorSpace((d, d))         sc.DenseCoordinateSpace((d, d))
space.eigh(x)                  space.spectral_decompose(x)
                               space.spectrum(x)
ProductInnerProductSpace(...)  ProductSpace(...)
StackedInnerProductSpace(...)  StackedSpace(...)
```

Remove public documentation references to private specialized product/stacked implementation classes.

## Priority 7 — Repository-wide API migration audit

Add a scripted API-audit tool that scans source files, documentation, notebooks, examples, benchmarks, and downstream repositories.

It must detect active uses of:

```text
VectorSpace(
.eigh(
ProductInnerProductSpace(
StackedInnerProductSpace(
other removed public specialized constructors
```

The audit should:

* parse Python files where practical;
* inspect notebook code cells;
* distinguish active code from prose where possible;
* report file and line or cell location;
* support scanning SpaceCore and configured downstream repositories.

Run it against:

```text
SpaceCore
SDPLab
```

Migrate downstream usages before tagging.

## Priority 8 — Release hygiene and artifact verification

Run:

```text
pytest
ruff check
docstring audit
JIT audit
Sphinx build with warnings treated as errors
wheel build
source-distribution build
fresh-environment wheel installation
fresh-environment source-distribution installation
```

### Backend coverage

Run the test suite with:

```text
NumPy
JAX
Torch, when supported by the project
```

Ensure backend-specific coverage includes:

* complex Jordan classification;
* dtype-changing conversion;
* JAX pytree product spectral round-trip;
* weighted inner-product validation.

### Benchmark/tooling checks

Verify that `bench/` imports and executes against the final hierarchy.

### Artifact cleanliness

Ensure source distributions and wheels exclude:

```text
__pycache__
.pytest_cache
__MACOSX
compiled bytecode
local build outputs
temporary audit files
```

### CI

Add or verify CI jobs for:

```text
Python 3.11
Python 3.12
NumPy backend
JAX backend
wheel-install smoke test
source-distribution smoke test
documentation build
repository API audit
```

## Acceptance criteria

* Source-tree, wheel, and source-distribution installations report `0.3.0`.
* `DenseVectorSpace` is a plain one-dimensional vector space without Jordan capability by default.
* `DenseVectorSpace` supports valid Euclidean, weighted, and custom inner products.
* Complex elementwise Jordan spaces are classified truthfully.
* Every declared `EuclideanJordanAlgebraSpace` satisfies its compatibility identity.
* Capabilities are recomputed after every conversion.
* Real-to-complex conversion removes invalid Euclidean-Jordan capability.
* Product spectral decomposition round-trips for tuple and pytree structures on NumPy and JAX.
* Spectral data uses an explicit structure independent of product elements.
* Invalid weighted inner products fail during construction and conversion.
* Product and stacked factories preserve complete valid capabilities.
* Equivalent reconstructed and converted spaces compare equal.
* Only final public constructors are exported.
* Every breaking change has a documented migration path.
* Full tests, linting, audits, documentation, artifact installation checks, benchmark smoke tests, and downstream API audits pass.

## Report

For each priority, report:

* files changed;
* implementation decisions;
* tests added;
* tests passing;
* remaining limitations.

Explicitly confirm:

* the complex elementwise Euclidean-Jordan misclassification is fixed;
* `DenseVectorSpace` has no Jordan capability by default;
* capabilities redispatch after conversion;
* the explicit `ProductSpectralDecomposition` round-trip passes for pytree products;
* weighted inner-product validation rejects every invalid case;
* dispatch uses one data-driven lookup without per-capability branching;
* the repository-wide API audit is clean for SpaceCore and SDPLab;
* any MRO conflict encountered and how it was resolved.
