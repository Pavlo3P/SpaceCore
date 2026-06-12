# Contributor architecture guide

SpaceCore is organized around mathematical contracts first and array containers second. A contributor should decide which layer owns a behavior before changing code: backend libraries own raw array execution, while SpaceCore owns spaces, geometry, validation, typed maps, and solver contracts.

## Backend layer

`spacecore/backend/` defines the backend abstraction. `BackendOps` is the contract implemented by concrete operation providers such as `NumpyOps` and optional JAX, Torch, and CuPy implementations. Core code should prefer `ops.method(...)` over direct backend calls when behavior must be portable.

The backend layer assumes Array API-style dense operations where possible, but it does not pretend that sparse arrays, dtype promotion, devices, tracing, mutation, or control flow are identical across NumPy, JAX, Torch, and CuPy. Backend detection is conservative: contexts and arrays are associated with a registered backend family, and optional backend classes are available only when their dependencies import successfully.

SpaceCore owns context normalization, validation hooks, conversion entry points, sparse/dense predicates, vectorization wrappers, and backend-independent control-flow helpers. NumPy, SciPy, JAX, Torch, and CuPy own their native array semantics, device placement, autograd/tracing behavior, kernel execution, sparse storage details, and low-level dtype behavior.

## Space layer

`spacecore/space/` represents mathematical structure, not merely array shapes. `VectorSpace` is the abstract linear-space contract for zero, add, scale, and axpy operations. Coordinate spaces add shape, size, flattening, unflattening, and batching hooks. Dense coordinate and dense vector spaces provide concrete dense array representations; product spaces represent structured elements composed from component spaces.

Spaces define valid elements through membership checks. Checks cover backend ownership, shape, dtype, product structure, Hermitian structure, square-matrix structure, and component membership. Spaces also own scalar-field-relevant dtype expectations, zero/add/scale behavior, and batching helpers such as batch flattening and unflattening.

Inner-product geometry is part of the space contract. Euclidean and weighted geometries define inner products, norms, and Riesz maps where available. Code that changes geometry must preserve the declared mathematical structure, not just pass array-shape tests.

## LinOp and Functional layers

`spacecore/linop/` contains typed maps between domain and codomain spaces. `apply(x)` evaluates the forward map `A x`; `rapply(y)` evaluates the metric adjoint `A^sharp y`; `.H` returns an adjoint view that swaps those actions. Algebraic operators implement composition, sums, scalar multiples, products, stacking, and block structure without eagerly materializing matrices.

Matrix-backed operators own a coordinate matrix or tensor, so they can compute metric adjoints with Riesz maps. Matrix-free operators do not own a matrix. `MatrixFreeLinOp.rapply` is a user-supplied assertion that the callable is already the correct metric adjoint for the declared spaces. SpaceCore validates membership when checks are enabled, but it does not derive or correct matrix-free adjoints.

`spacecore/functional/` contains scalar-valued maps. Functionals expose values, gradients where available, and pull-backs through LinOp composition. Gradients are space elements and must respect the domain geometry. Batched application paths should agree row-by-row with scalar `apply` or `rapply` behavior.

## Linalg layer

`spacecore/linalg/` contains iterative algorithms such as CG, LSQR, Lanczos, power iteration, and exponential action. These routines use space operations for vector arithmetic and norms, use LinOp contracts for forward and adjoint products, and avoid assuming a concrete matrix unless documented.

Solver assumptions are mathematical assumptions over declared spaces: Hermitian, positive-definite, residual, adjoint, and norm statements are with respect to the relevant space geometry. Backend independence comes from using `BackendOps` control-flow and array helpers rather than backend-specific logic in solver bodies.

## Cross-cutting infrastructure

`spacecore/_contextual/` owns context-bound objects, context normalization, default-context state, backend registration, and conversion dispatch. `spacecore/_batching.py` provides shared batching helpers. `spacecore/_checks.py` provides method-level validation decorators used by spaces, operators, and functionals. `spacecore/_tree.py` supports structured product/pytree handling. `spacecore/_version.py` is the package version source used by packaging and docs.

## Public class map

This map lists public concrete class-like exports a contributor is likely to encounter. Abstract contracts such as `BackendOps`, `LinOp`, `Functional`, `VectorSpace`, `CoordinateSpace`, `InnerProduct`, and `SpaceCheck` are discussed above and in the ADRs.

- `ArrayLike` — runtime-checkable protocol for dense or sparse array-like values accepted by public APIs.
- `BackendCheck` — membership check that validates an element belongs to the expected backend family.
- `BackendFamily` — backend-family identifier used by contexts and backend detection.
- `BlockDiagonalLinOp` — product-space operator that applies component operators independently on each block.
- `CGResult` — result tuple returned by conjugate-gradient solves.
- `ComposedFunctional` — functional representing lazy composition of a functional with a linear operator.
- `ComposedLinOp` — lazy composition `A @ B` of compatible linear operators.
- `Context` — execution context containing backend operations, default dtype, and check policy.
- `ContextBound` — base for objects that carry a `Context` and support explicit conversion.
- `DenseArray` — runtime-checkable protocol for dense array-like values.
- `DenseCoordinateSpace` — dense coordinate representation of arrays with a fixed shape and context.
- `DenseLinOp` — dense matrix- or tensor-backed linear operator.
- `DenseVectorSpace` — one-dimensional dense coordinate vector space.
- `DTypeCheck` — membership check enforcing exact dtype membership.
- `DiagonalLinOp` — coordinatewise diagonal matrix-backed operator on one space.
- `ElementwiseJordanSpace` — dense elementwise Jordan algebra space.
- `EuclideanElementwiseJordanSpace` — real Euclidean elementwise Jordan algebra space.
- `EuclideanInnerProduct` — Euclidean inner-product geometry with identity Riesz maps.
- `ExpmMultiplyResult` — result tuple returned by exponential-action routines.
- `HermitianCheck` — membership check for Hermitian matrix structure.
- `HermitianSpace` — dense Hermitian matrix space with spectral Jordan operations.
- `IdentityLinOp` — identity linear operator on one space.
- `InnerProductFunctional` — linear functional represented by an inner product against a representer.
- `JaxOps` — optional JAX backend operations class, exported when JAX is installed.
- `LanczosResult` — result tuple returned by Lanczos routines.
- `LinOpQuadraticForm` — quadratic functional represented by a linear operator.
- `LSQRResult` — result tuple returned by LSQR solves.
- `MatrixFreeLinearFunctional` — linear functional defined by a user-supplied callable and representer behavior.
- `MatrixFreeLinOp` — linear operator defined by user-supplied forward and metric-adjoint callables.
- `NumpyOps` — NumPy/SciPy backend operations class and the baseline backend implementation.
- `PowerIterationResult` — result tuple returned by power iteration.
- `TreeElement` — optional explicit binding of ordered leaves to a tree space.
- `TreeLinOp` — base class for operators with tree-structured domains or codomains.
- `TreeSpace` — finite direct-product coordinate space represented by an optree.
- `TreeSpectralDecomposition` — leafwise spectral data in deterministic tree order.
- `ScaledLinOp` — lazy scalar multiple of a linear operator.
- `ShapeCheck` — membership check enforcing element shape.
- `SparseArray` — runtime-checkable protocol for sparse array-like values.
- `SparseLinOp` — sparse matrix-backed linear operator over coordinate spaces.
- `Space` — context-bound mathematical space with membership validation.
- `SpaceValidationError` — exception raised when an element fails space validation.
- `SquareMatrixCheck` — membership check enforcing square matrix shape.
- `StackedLinOp` — operator mapping one domain into a tree codomain.
- `StackedSpace` — space representing a fixed leading-axis stack of one base space.
- `SumLinOp` — lazy finite sum of operators with common domain and codomain.
- `SumToSingleLinOp` — operator mapping a tree domain into one codomain by summing leaf outputs.
- `TorchOps` — optional Torch backend operations class, exported when Torch is installed.
- `WeightedInnerProduct` — diagonal weighted inner-product geometry with validated positive finite weights.
- `ZeroLinOp` — zero linear operator between two spaces.
- `CuPyOps` — optional CuPy backend operations class, exported when CuPy is installed.
