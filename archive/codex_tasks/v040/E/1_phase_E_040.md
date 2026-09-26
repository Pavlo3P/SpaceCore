# Phase E plan — Block-structured LinOps

## Goal

Implement block-structured linear operators over `TreeSpace`.

Mathematically, support operators between finite direct products

[
X = \prod_i X_i, \qquad Y = \prod_j Y_j.
]

Do not treat these as tensor products.

## Preflight

1. Run only through `.venv`.
2. Check graph freshness:

   * `git rev-parse HEAD`
   * compare with `GRAPH_REPORT.md`
   * if stale, use it only for rough navigation.
3. Inspect:

   * `spacecore/linop/_base.py`
   * `spacecore/linop/tree/*`
   * `spacecore/space/concrete/_tree_space.py`
   * LinOp algebra modules: sum, scale, compose, adjoint
   * batching helpers
   * non-Euclidean inner-product tests

## E1–E6: `BlockDiagonalLinOp`

Implement first. It is simpler and should stabilize the TreeSpace/LinOp interface.

Required API:

```python
BlockDiagonalLinOp(blocks)
```

where each block is a LinOp

```python
A_i : X_i -> Y_i
```

and the full operator is

```python
A : TreeSpace(X_i) -> TreeSpace(Y_i).
```

Implementation requirements:

1. Infer domain as the TreeSpace of block domains.
2. Infer codomain as the TreeSpace of block codomains.
3. Validate all blocks are LinOps.
4. Validate compatible context/check policy.
5. `apply(x)` applies each block leafwise.
6. `rapply(y)` applies each block metric adjoint leafwise.
7. `.H` returns a block-diagonal operator with blocks `A_i.H`.
8. Add deterministic tests for:

   * Euclidean apply/rapply;
   * non-Euclidean metric adjoint;
   * `.H.H`;
   * TreeSpace structure preservation;
   * batching if currently supported.

## E7–E13: `BlockMatrixLinOp`

Implement after block diagonal is stable.

Required API:

```python
BlockMatrixLinOp(block_rows)
```

where `block_rows` is a nested tuple/list/tree such as

```python
((A, B),
 (C, D))
```

with

```python
A : X_0 -> Y_0
B : X_1 -> Y_0
C : X_0 -> Y_1
D : X_1 -> Y_1
```

and the full operator is

```python
A_block : X_0 × X_1 -> Y_0 × Y_1.
```

Implementation requirements:

1. Validate rectangular block structure.
2. Validate each row has compatible codomain leaf.
3. Validate each column has compatible domain leaf.
4. Infer domain TreeSpace from block columns.
5. Infer codomain TreeSpace from block rows.
6. `apply(x)` computes row sums:

```python
y_i = sum_j A_ij.apply(x_j)
```

7. `rapply(y)` computes column sums of metric adjoints:

```python
x_j = sum_i A_ij.rapply(y_i)
```

8. `.H` transposes the block structure and replaces each block by `.H`.

## Algebra integration

Ensure block operators work with existing LinOp algebra:

1. addition;
2. scaling;
3. composition;
4. zero/identity simplifications where already supported;
5. `.H`;
6. `.H.H`;
7. equality or structural comparison if the project already supports it.

Do not add broad simplification machinery unless needed for correctness.

## Tests

Add focused tests before broad generator tests.

Required test groups:

1. `tests/linops/test_block_diagonal_linop.py`
2. `tests/linops/test_block_matrix_linop.py`
3. non-Euclidean metric-adjoint tests;
4. batched apply/rapply tests if supported;
5. algebra tests for composition/addition/scaling;
6. public API import tests.

Use tiny deterministic spaces and matrices. Avoid random tests in Phase E unless seeded and simple.

## Docs

Add or update docs explaining:

1. block-diagonal operators;
2. block-matrix operators;
3. TreeSpace domain/codomain inference;
4. direct-product meaning;
5. metric adjoint vs coordinate transpose;
6. examples with `((A, B), (C, D))`.

Explicitly state: these are direct-product block operators, not tensor-product or Kronecker-product operators.

## Acceptance criteria

* `BlockDiagonalLinOp` works over TreeSpace.
* `BlockMatrixLinOp` works over TreeSpace.
* `apply`, `rapply`, and `.H` are correct.
* Euclidean and non-Euclidean tests pass.
* Batching is tested or explicitly documented as unsupported.
* Existing LinOp algebra works with block operators.
* Docs include direct-product limitation.
* No `ProductSpace` support is reintroduced.
* `.venv/bin/python -m pytest tests/ -x -q` passes.
* `.venv/bin/python -m ruff check .` passes.
