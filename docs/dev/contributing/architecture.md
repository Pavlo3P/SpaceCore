# Contributor architecture guide

SpaceCore is organized around mathematical contracts first and array containers second. A contributor should decide which layer owns a behavior before changing code: backend libraries own raw array execution, while SpaceCore owns spaces, geometry, validation, typed maps, and solver contracts.

## Backend layer

`spacecore/backend/` defines the backend abstraction. `BackendOps` is the contract implemented by concrete operation providers such as `NumpyOps` and optional JAX, Torch, and CuPy implementations. Core code should prefer `ops.method(...)` over direct backend calls when behavior must be portable.

The backend layer assumes Array API-style dense operations where possible, but it does not pretend that sparse arrays, dtype promotion, devices, tracing, mutation, or control flow are identical across NumPy, JAX, Torch, and CuPy. Backend detection is conservative: contexts and arrays are associated with a registered backend family, and optional backend classes are available only when their dependencies import successfully.

SpaceCore owns context normalization, validation hooks, conversion entry points, sparse/dense predicates, vectorization wrappers, and backend-independent control-flow helpers. NumPy, SciPy, JAX, Torch, and CuPy own their native array semantics, device placement, autograd/tracing behavior, kernel execution, sparse storage details, and low-level dtype behavior.

## Space layer

`spacecore/space/` represents mathematical structure, not merely array shapes. `VectorSpace` is the abstract linear-space contract for zero, add, scale, and axpy operations. Coordinate spaces add shape, size, flattening, unflattening, and batching hooks. Dense coordinate and dense vector spaces provide concrete dense array representations; `TreeSpace` and `TreeElement` represent finite direct products organized by an `optree` definition, and `StackedSpace` represents a fixed leading-axis stack of one base space.

Spaces define valid elements through membership checks. Checks cover backend ownership, shape, dtype, product structure, Hermitian structure, square-matrix structure, and component membership. Spaces also own scalar-field-relevant dtype expectations, zero/add/scale behavior, and batching helpers such as batch flattening and unflattening.

Inner-product geometry is part of the space contract. Euclidean and weighted geometries define inner products, norms, and Riesz maps where available. Code that changes geometry must preserve the declared mathematical structure, not just pass array-shape tests.

## LinOp and Functional layers

`spacecore/mapping/` holds `Mapping`, the common parent of `LinOp` and `Functional`, with the lazy algebra both families share: the operator-overload rule table, the scaled/sum/zero nodes and the scalar helpers. It also derives equality from a class's declared `_fields` (see `docs/source/design/extending.rst`).

`spacecore/linop/` contains typed maps between domain and codomain spaces. `apply(x)` evaluates the forward map `A x`; `rapply(y)` evaluates the metric adjoint `A^sharp y`; `.H` returns an adjoint view that swaps those actions. `LinOp` owns these checked public methods and subclasses implement the check-free cores `_apply_core`, `_rapply_core` (and optionally the batched ones), which composites and solvers call directly. Algebraic operators implement composition, sums, scalar multiples, and block structure without eagerly materializing matrices; `BlockMatrixLinOp` also covers stacking one space into a tree (one column) and summing a tree into one space (one row).

Matrix-backed operators own a coordinate matrix or tensor, so they can compute metric adjoints with Riesz maps. Matrix-free operators do not own a matrix. `MatrixFreeLinOp.rapply` is a user-supplied assertion that the callable is already the correct metric adjoint for the declared spaces. SpaceCore validates membership when checks are enabled, but it does not derive or correct matrix-free adjoints.

`spacecore/functional/` contains scalar-valued maps. Functionals expose values, gradients where available, and pull-backs through LinOp composition. Gradients are space elements and must respect the domain geometry. Batched application paths should agree row-by-row with scalar `apply` or `rapply` behavior.

## Linalg layer

`spacecore/linalg/` contains iterative algorithms such as CG, LSQR, Lanczos, power iteration, and exponential action. These routines use space operations for vector arithmetic and norms, use LinOp contracts for forward and adjoint products, and avoid assuming a concrete matrix unless documented.

Solver assumptions are mathematical assumptions over declared spaces: Hermitian, positive-definite, residual, adjoint, and norm statements are with respect to the relevant space geometry. Backend independence comes from using `BackendOps` control-flow and array helpers rather than backend-specific logic in solver bodies.

## Cross-cutting infrastructure

`spacecore/contextual/` owns context-bound objects, context normalization, default-context state, backend registration, and conversion, which `ContextBound` derives from a class's declared `_fields`; `PyTreeNode` in `spacecore/backend/_container.py` derives the pytree hooks from the same declaration. `spacecore/_batching.py` provides shared batching helpers. `spacecore/_checks.py` provides the method-level validation decorators used by spaces, operators, and functionals, and defines the `CheckLevel` literal (`"none"`, `"cheap"`, `"standard"`, `"strict"`) with its `CHECK_LEVELS` ordering. Tree-structured handling lives with the space implementations in `spacecore/space/concrete/_tree_space.py` (using `optree`) rather than in a top-level helper module. `spacecore/_version.py` is the package version source used by packaging and docs.

## Public names

The public API is `spacecore.__all__`; every exported name is documented, from its
docstring, on the API pages under `docs/source/api/`. To see the current list:

```bash
python -c "import spacecore; print(sorted(spacecore.__all__))"
```

Optional backends (`JaxOps`, `TorchOps`, `CuPyOps`) appear only when their package is
installed. `BackendFamily` and the other backend internals live in `spacecore.backend`.
