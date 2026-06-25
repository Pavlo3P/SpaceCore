# LinOp generator coverage

Generated LinOp tests live in `tests/generators/linops.py`. Each `LinOpCase`
stores the operator in `obj` and the following reference data: domain,
codomain, scalar inputs, batched inputs, direct `apply` and `rapply` results,
an optional coordinate matrix, and explicit batching/conversion capability
flags.

## Public family inventory

| Family | Constructor | Domain to codomain | `.H` | Batch | Reference |
| --- | --- | --- | --- | --- | --- |
| `DenseLinOp` | `sc.DenseLinOp(A, X, Y)` | coordinate `X -> Y` | yes | yes | dense matrix action and metric adjoint |
| `SparseLinOp` | `sc.SparseLinOp(A, X, Y)` | coordinate `X -> Y` | yes | yes | dense equivalent; skip unsupported sparse backends |
| `DiagonalLinOp` | `sc.DiagonalLinOp(d, X)` | `X -> X` | yes | yes | elementwise diagonal and conjugate diagonal |
| `IdentityLinOp` | `sc.IdentityLinOp(X)` | `X -> X` | yes | yes | unchanged input |
| `ZeroLinOp` | `sc.ZeroLinOp(X, Y)` | `X -> Y` | yes | yes | space-owned zero values |
| `MatrixFreeLinOp` | `sc.MatrixFreeLinOp(apply, rapply, X, Y)` | any compatible `X -> Y` | yes | yes | explicit forward and true metric-adjoint callables |
| `ScaledLinOp` | `sc.ScaledLinOp(alpha, A)` | same as `A` | yes | yes | `alpha A` and `conj(alpha) A^sharp` |
| `SumLinOp` | `sc.SumLinOp((A, B))` | common `X -> Y` | yes | yes | sum of direct references |
| `ComposedLinOp` | `sc.ComposedLinOp(B, A)` | `X -> Z` through `Y` | yes | yes | composed matrices/functions |
| `StackedLinOp` | `sc.StackedLinOp.from_operators(parts)` | `X -> TreeSpace[Y_i]` | yes | yes | leafwise forward, summed adjoints |
| `SumToSingleLinOp` | `sc.SumToSingleLinOp.from_operators(parts)` | `TreeSpace[X_i] -> Y` | yes | yes | summed forwards, leafwise adjoints |
| `BlockDiagonalLinOp` | `sc.BlockDiagonalLinOp(block_tree)` | `TreeSpace[X_i] -> TreeSpace[Y_i]` | yes | yes | leafwise application |
| `BlockMatrixLinOp` | `sc.BlockMatrixLinOp(rows)` | `TreeSpace[X_j] -> TreeSpace[Y_i]` | yes | yes | row sums and adjoint column sums |
| `TreeLinOp` | abstract base | structured endpoints | inherited | inherited | no direct case; cover every concrete subclass |

## Adding a case

1. Build backend values from deterministic NumPy data and convert them through
   the case `Context`.
2. Store both forward and reverse references. For coordinate matrix `A` and
   metric matrices `G_X`, `G_Y`, the reverse matrix is
   `solve(G_X, A.conj().T @ G_Y)`, not merely `A.conj().T`.
3. Provide leading-axis batch inputs and direct batched references. A
   `TreeSpace` batch is a tree whose leaves each carry the same leading axis.
4. For block operators, use `TreeSpace` exclusively. Block-matrix forward
   references are row sums; reverse references are sums down the adjoint
   columns.
5. Set `supports_batching` or `supports_conversion` to false only when the
   public family genuinely does not support the operation, and add a focused
   test for the documented error.
6. Add the concrete class to the inventory assertion in
   `tests/linops/test_generated_linop_laws.py`.

Describe structured operator inputs and outputs as `TreeSpace` values and do
not add legacy structured-space compatibility paths to new generator code.
