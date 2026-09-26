# Task: Fix `MatrixFreeLinOp` adjoint contract in non-Euclidean spaces

## Goal

Fix `MatrixFreeLinOp` so that user-supplied matrix-free callables are used exactly as provided.

For a linear operator (A : X \to Y), where (X) and (Y) are inner-product spaces, the contract is:

* `apply(x)` implements (Ax \in Y);
* `rapply(y)` implements the metric adjoint (A^\sharp y \in X);
* (A^\sharp) is characterized by
  [
  \langle Ax, y\rangle_Y = \langle x, A^\sharp y\rangle_X.
  ]

For matrix-free operators, SpaceCore must trust that the user-provided `rapply` already implements (A^\sharp). It must not insert Riesz maps, Euclidean transposes, or matrix-backed adjoint corrections.

The formula
[
A^\sharp = R_X^{-1} A^\dagger R_Y
]
belongs only to matrix-backed operators where SpaceCore owns the coordinate matrix.

## Required audit

Inspect all relevant code paths for `MatrixFreeLinOp`:

* `apply`;
* `rapply`;
* `.H`;
* adjoint-view helpers;
* `vapply`;
* `rvapply`;
* conversion paths if they wrap application;
* validation/checking paths if they call application internally.

Record findings in comments, tests, or a short developer note if useful.

## Required behavior

After the fix:

* `MatrixFreeLinOp.apply(x)` calls only the user forward callable, aside from validation/checking.
* `MatrixFreeLinOp.rapply(y)` calls only the user reverse callable, aside from validation/checking.
* `op.H.apply(y)` delegates to the original `op.rapply(y)`.
* `op.H.rapply(x)` delegates to the original `op.apply(x)`.
* `op.H.H` behaves like the original operator.
* `vapply(xs)` agrees row-by-row with repeated `apply`.
* `rvapply(ys)` agrees row-by-row with repeated `rapply`.
* matrix-backed operators are unchanged and still use the metric-adjoint formula.

## Test fixture

Add a non-Euclidean fixture where the Euclidean transpose differs from the metric adjoint.

For example, use finite-dimensional real coordinate spaces:

* (X = \mathbb{R}^n) with inner product
  [
  \langle x_1, x_2\rangle_X = x_1^\top G_X x_2,
  ]
* (Y = \mathbb{R}^m) with inner product
  [
  \langle y_1, y_2\rangle_Y = y_1^\top G_Y y_2,
  ]
  where (G_X) and (G_Y) are symmetric positive definite and not both identity.

Choose a coordinate matrix (A \in \mathbb{R}^{m \times n}). Define:

[
A^\sharp = G_X^{-1} A^\top G_Y.
]

Construct `MatrixFreeLinOp` with:

* `apply(x) = A @ x`;
* `rapply(y) = A_sharp @ y`.

The test must be able to detect if SpaceCore accidentally applies another Riesz correction.

## Required tests

Add tests covering:

1. **Forward callable is untouched**

   Construct a matrix-free operator whose forward callable is known. Verify:

   ```python
   op.apply(x) == A @ x
   ```

   It must not equal any geometry-wrapped variant unless those happen to coincide accidentally. Use a fixture where they do not coincide.

2. **Reverse callable is untouched**

   Verify:

   ```python
   op.rapply(y) == A_sharp @ y
   ```

   The test should fail against the pre-fix behavior if the old path inserted Riesz maps.

3. **Adjoint view delegates correctly**

   Verify:

   ```python
   op.H.apply(y) == op.rapply(y)
   op.H.rapply(x) == op.apply(x)
   ```

   Also verify that `op.H.H` behaves like `op`.

4. **Adjoint identity**

   For representative (x \in X), (y \in Y), verify:

   [
   \langle Ax, y\rangle_Y = \langle x, A^\sharp y\rangle_X.
   ]

   Use the spaces’ own `inner` methods, not raw Euclidean dot products.

5. **Batched consistency**

   For batched inputs `xs` and `ys`, verify:

   ```python
   op.vapply(xs)[i] == op.apply(xs[i])
   op.rvapply(ys)[i] == op.rapply(ys[i])
   ```

   This should hold for the same non-Euclidean fixture.

6. **Matrix-backed behavior is preserved**

   Existing matrix-backed adjoint tests must still pass. Add a focused regression if needed: matrix-backed operators should still compute metric adjoints using

   [
   A^\sharp = R_X^{-1} A^\dagger R_Y.
   ]

## Documentation updates

Update the relevant docstrings and developer docs:

* `MatrixFreeLinOp.rapply` must be described as a user-supplied implementation of the metric adjoint.
* State explicitly that SpaceCore does not derive or correct the matrix-free adjoint.
* State that automatic Riesz-map correction belongs to matrix-backed operators.
* Mention that incorrect `rapply` can pass construction but fail adjoint dot-tests.

Update the changelog under `[Unreleased]` with a correctness-fix entry.

## Acceptance criteria

The task is complete when:

* matrix-free `apply` and `rapply` use the supplied callables directly;
* `.H` behaves by swapping supplied `apply`/`rapply`;
* non-Euclidean matrix-free regression tests pass;
* batched matrix-free application is consistent with scalar application;
* matrix-backed adjoint behavior remains unchanged;
* docstrings explain the matrix-free adjoint contract;
* changelog records the fix;
* the following pass:

```bash
pytest tests/ -x -q
ruff check .
```
