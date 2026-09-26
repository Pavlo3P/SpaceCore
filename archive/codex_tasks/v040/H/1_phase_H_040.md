# Phase H plan — LinOp coverage

## Goal

Build generated test coverage for the full LinOp algebra:

* dense matrix-backed LinOps;
* sparse LinOps;
* diagonal LinOps;
* identity LinOps;
* zero LinOps;
* matrix-free LinOps;
* composed LinOps;
* summed LinOps;
* scaled LinOps;
* TreeSpace block-diagonal LinOps;
* TreeSpace block-matrix LinOps.

The main invariant is that every generated operator (A : X \to Y) must be testable against reference behavior for `apply`, `rapply`, `.H`, algebra, conversion, and batching.

## Dependency assumptions

Before Phase H starts, these should already exist:

1. Phase F generator infrastructure.
2. Phase G space generators, especially DenseCoordinateSpace, DenseVectorSpace, inner-product spaces, and TreeSpace.
3. Phase E block LinOps, or else H10–H11 should be recorded as blocked.
4. ProductSpace must not be used. Structured domains and codomains must use TreeSpace only.

## Navigation rule

Use `GRAPH_REPORT.md` first to locate relevant LinOp families and cross-cutting validation paths. Use `graph.json` only for exact symbol/file lookup. Verify graph-inferred relations against source before editing.

Focus especially on:

* `spacecore/linop/`;
* `spacecore/linop/tree/`;
* `spacecore/_checks.py`;
* `spacecore/_batching.py`;
* `spacecore/space/checks/`;
* existing LinOp tests;
* generator package from Phase F.

## Step 1 — Inventory public LinOp families

Create a short inventory document or test module comment listing every public LinOp family:

* DenseLinOp;
* SparseLinOp if present;
* DiagonalLinOp;
* IdentityLinOp;
* ZeroLinOp;
* MatrixFreeLinOp;
* ComposedLinOp;
* SumLinOp;
* ScaledLinOp;
* TreeLinOp;
* BlockDiagonalLinOp;
* BlockMatrixLinOp or equivalent.

For each, record:

* constructor path;
* domain space (X);
* codomain space (Y);
* whether `.H` is supported;
* whether batching is supported;
* reference implementation strategy.

## Step 2 — Add generated reference cases

Add generators under `tests/generators/linops.py` or the established Phase F location.

Each generated case should include:

```python
@dataclass(frozen=True)
class LinOpCase:
    op: LinOp
    domain: Space
    codomain: Space
    x: Any
    y: Any
    expected_apply: Any
    expected_rapply: Any
    reference_matrix: Any | None
    supports_batching: bool
    supports_conversion: bool
```

Adjust names to the existing generator protocol.

## Step 3 — Implement basic LinOp generators

Implement generators in this order:

1. Dense matrix-backed LinOp.
2. DiagonalLinOp.
3. IdentityLinOp.
4. ZeroLinOp.
5. Sparse LinOp where backend supports sparse arrays.
6. MatrixFreeLinOp using explicit reference functions.
7. ScaledLinOp from a base generated operator.
8. SumLinOp from compatible generated operators.
9. ComposedLinOp from composable generated operators.

Each generator must provide reference `apply` and `rapply`.

## Step 4 — Implement TreeSpace block generators

Implement only with TreeSpace.

Block-diagonal generator:

* generate leaf operators (A_i : X_i \to Y_i);
* domain is `TreeSpace` over leaves (X_i);
* codomain is `TreeSpace` over leaves (Y_i);
* reference apply is leafwise application.

Block-matrix generator:

* generate nested blocks (A_{ij}: X_j \to Y_i);
* validate row/column compatibility;
* reference apply is row-wise sum:

[
(Ax)*i = \sum_j A*{ij}x_j.
]

If block-matrix LinOp is not implemented yet, mark H11 blocked and create a narrow follow-up.

## Step 5 — Add law tests

Add generated tests for:

1. `apply(x)` equals reference.
2. `rapply(y)` equals reference metric adjoint.
3. Adjoint identity:

[
\langle Ax, y\rangle_Y = \langle x, A^\sharp y\rangle_X.
]

4. `.H.H` agrees with the original operator where valid.
5. Composition agrees with reference composition.
6. Sum agrees with reference sum.
7. Scaling agrees with reference scaling.
8. Zero and identity laws hold.
9. Conversion preserves behavior.
10. Batched `vapply` and `rvapply` agree with reference vectorization.

## Step 6 — Non-Euclidean coverage

Include at least one non-Euclidean inner-product domain/codomain case.

This is essential: `rapply` must test the metric adjoint, not merely coordinate transpose.

For a LinOp (A : X \to Y), the tested adjoint must satisfy:

[
\langle Ax, y\rangle_Y = \langle x, A^\sharp y\rangle_X.
]

Do not accept tests that only compare against Euclidean matrix transpose unless the spaces are Euclidean.

## Step 7 — Batching coverage

For every LinOp family that claims batching support:

* generate a leading-axis batch of inputs;
* compare `vapply(xs)` with stacked individual `apply(x_i)`;
* compare `rvapply(ys)` with stacked individual `rapply(y_i)`;
* test both unbatched and batched paths under check levels `none`, `cheap`, `standard`, and `strict` where feasible.

If batching is unsupported for a family, document that explicitly and test that the error is clear.

## Step 8 — Backend/context coverage

Run cases across supported backend/dtype/context combinations from Phase F.

At minimum:

* NumPy real;
* NumPy complex;
* JAX if installed;
* Torch if installed;
* CuPy optional skip.

Sparse tests should skip clearly when backend sparse support is unavailable.

## Step 9 — Documentation

Add contributor-facing documentation explaining how to add a new LinOp generator.

Include:

* required reference data;
* how to handle metric adjoints;
* how to handle TreeSpace block operators;
* how to mark unsupported batching;
* how to avoid ProductSpace terminology.

## Step 10 — Acceptance checklist

Phase H is complete when:

* H1–H16 are implemented or narrowly marked blocked.
* Every public LinOp family has a generator or documented exclusion.
* Dense, sparse, diagonal, identity, zero, matrix-free, composed, summed, scaled, block-diagonal, and block-matrix LinOps are covered where implemented.
* Metric-adjoint identity is tested in Euclidean and non-Euclidean spaces.
* `.H.H` identity is tested where valid.
* Algebraic identities are tested.
* Conversion is tested.
* Batching is tested or explicitly unsupported.
* TreeSpace is used for all structured operators.
* No ProductSpace support or terminology is reintroduced.
* Tests run through `.venv` only:

```bash
.venv/bin/python -m pytest tests/ -x -q
.venv/bin/python -m ruff check .
```
