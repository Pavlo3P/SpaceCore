from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any

import numpy as np

import spacecore as sc

from ._protocol import GeneratedCase


LinOpCase = GeneratedCase[sc.LinOp]
NUMPY_LINOP_DTYPES = (np.float64, np.complex128)


def _context(dtype: Any, check_level: sc.CheckLevel | str) -> sc.Context:
    return sc.Context(sc.NumpyOps(), dtype=dtype, check_level=check_level)


def _numpy_dtype(ctx: sc.Context) -> np.dtype[Any]:
    text = str(ctx.dtype)
    if ctx.ops.is_complex_dtype(ctx.dtype):
        return np.dtype(np.complex64 if "64" in text and "128" not in text else np.complex128)
    return np.dtype(np.float32 if "32" in text else np.float64)


def _target_context(ctx: sc.Context) -> sc.Context | None:
    if ctx.ops.family != "numpy":
        return None
    dtype = np.dtype(ctx.dtype)
    target = np.complex64 if np.issubdtype(dtype, np.complexfloating) else np.float32
    return sc.Context(sc.NumpyOps(), dtype=target, check_level=ctx.check_level)


def _array(ctx: sc.Context, values: Any) -> Any:
    return ctx.asarray(np.asarray(values, dtype=_numpy_dtype(ctx)))


def _complex(ctx: sc.Context) -> bool:
    return ctx.ops.is_complex_dtype(ctx.dtype)


def _matrix_data(ctx: sc.Context) -> dict[str, np.ndarray]:
    imaginary = 1j if _complex(ctx) else 0.0
    dtype = _numpy_dtype(ctx)
    return {
        "A": np.asarray(
            [[1.0 + 0.5 * imaginary, -2.0 * imaginary], [0.5, 3.0], [4.0, -1.0]],
            dtype=dtype,
        ),
        "C": np.asarray(
            [[-0.25, 2.0], [1.0 - 0.5 * imaginary, 0.5], [2.5, -1.0]],
            dtype=dtype,
        ),
        "B": np.asarray(
            [[2.0, -0.5, 1.0], [-1.5, 0.25 + 0.5 * imaginary, 3.0]],
            dtype=dtype,
        ),
        "x": np.asarray([0.25 + 0.5 * imaginary, -1.5], dtype=dtype),
        "y": np.asarray(
            [2.0 - 0.25 * imaginary, -0.5, 1.25 + 0.5 * imaginary],
            dtype=dtype,
        ),
        "z": np.asarray([1.5, -0.75 + 0.25 * imaginary], dtype=dtype),
        "batch_x": np.asarray(
            [[0.25 + 0.5 * imaginary, -1.5], [-1.0, 0.75 - 0.25 * imaginary]],
            dtype=dtype,
        ),
        "batch_y": np.asarray(
            [
                [2.0 - 0.25 * imaginary, -0.5, 1.25 + 0.5 * imaginary],
                [-0.75, 1.5 + 0.25 * imaginary, 0.5],
            ],
            dtype=dtype,
        ),
    }


def _metric_adjoint(
    matrix: np.ndarray,
    domain_metric: np.ndarray | None = None,
    codomain_metric: np.ndarray | None = None,
) -> np.ndarray:
    gx = np.eye(matrix.shape[1], dtype=matrix.dtype) if domain_metric is None else domain_metric
    gy = np.eye(matrix.shape[0], dtype=matrix.dtype) if codomain_metric is None else codomain_metric
    return np.linalg.solve(gx, matrix.conj().T @ gy)


def _reference(
    op: sc.LinOp,
    *,
    family: str,
    x: Any,
    y: Any,
    expected_apply: Any,
    expected_rapply: Any,
    batch_x: Any,
    batch_y: Any,
    expected_vapply: Any,
    expected_rvapply: Any,
    reference_matrix: Any | None,
    target_ctx: sc.Context | None,
    supports_conversion: bool = True,
) -> LinOpCase:
    capabilities = {"adjoint", "batching", family}
    if supports_conversion:
        capabilities.add("conversion")
    if isinstance(op.domain, sc.TreeSpace) or isinstance(op.codomain, sc.TreeSpace):
        capabilities.add("tree")
    if not op.domain.is_euclidean or not op.codomain.is_euclidean:
        capabilities.add("non_euclidean")
    return LinOpCase(
        obj=op,
        reference={
            "family": family,
            "domain": op.domain,
            "codomain": op.codomain,
            "x": x,
            "y": y,
            "expected_apply": expected_apply,
            "expected_rapply": expected_rapply,
            "batch_x": batch_x,
            "batch_y": batch_y,
            "expected_vapply": expected_vapply,
            "expected_rvapply": expected_rvapply,
            "reference_matrix": reference_matrix,
            "supports_batching": True,
            "supports_conversion": supports_conversion,
            "target_ctx": target_ctx,
        },
        capabilities=frozenset(capabilities),
        id=f"{family}-{op.ctx.ops.family}-{_numpy_dtype(op.ctx).name}-checks-{op.check_level}",
    )


def dense_linop_case(
    ctx: sc.Context,
    *,
    weighted: bool = False,
) -> LinOpCase:
    data = _matrix_data(ctx)
    if weighted:
        gx = np.diag(np.asarray([2.0, 5.0], dtype=_numpy_dtype(ctx)))
        gy = np.diag(np.asarray([3.0, 7.0, 11.0], dtype=_numpy_dtype(ctx)))
        domain = sc.DenseCoordinateSpace(
            (2,), ctx, geometry=sc.WeightedInnerProduct(_array(ctx, np.diag(gx)))
        )
        codomain = sc.DenseCoordinateSpace(
            (3,), ctx, geometry=sc.WeightedInnerProduct(_array(ctx, np.diag(gy)))
        )
        family = "dense-weighted"
    else:
        gx = gy = None
        domain = sc.DenseCoordinateSpace((2,), ctx)
        codomain = sc.DenseCoordinateSpace((3,), ctx)
        family = "dense"
    adjoint = _metric_adjoint(data["A"], gx, gy)
    op = sc.DenseLinOp(_array(ctx, data["A"]), domain, codomain, ctx)
    return _reference(
        op,
        family=family,
        x=_array(ctx, data["x"]),
        y=_array(ctx, data["y"]),
        expected_apply=data["A"] @ data["x"],
        expected_rapply=adjoint @ data["y"],
        batch_x=_array(ctx, data["batch_x"]),
        batch_y=_array(ctx, data["batch_y"]),
        expected_vapply=data["batch_x"] @ data["A"].T,
        expected_rvapply=data["batch_y"] @ adjoint.T,
        reference_matrix=data["A"],
        target_ctx=_target_context(ctx),
    )


def sparse_linop_case(ctx: sc.Context) -> LinOpCase:
    data = _matrix_data(ctx)
    domain = sc.DenseCoordinateSpace((2,), ctx)
    codomain = sc.DenseCoordinateSpace((3,), ctx)
    matrix = _array(ctx, data["A"])
    op = sc.SparseLinOp(ctx.assparse(matrix), domain, codomain, ctx)
    adjoint = data["A"].conj().T
    return _reference(
        op,
        family="sparse",
        x=_array(ctx, data["x"]),
        y=_array(ctx, data["y"]),
        expected_apply=data["A"] @ data["x"],
        expected_rapply=adjoint @ data["y"],
        batch_x=_array(ctx, data["batch_x"]),
        batch_y=_array(ctx, data["batch_y"]),
        expected_vapply=data["batch_x"] @ data["A"].T,
        expected_rvapply=data["batch_y"] @ adjoint.T,
        reference_matrix=data["A"],
        target_ctx=_target_context(ctx),
    )


def diagonal_linop_case(ctx: sc.Context) -> LinOpCase:
    imaginary = 1j if _complex(ctx) else 0.0
    dtype = _numpy_dtype(ctx)
    diagonal = np.asarray([2.0 + 0.5 * imaginary, -1.0, 0.5 - imaginary], dtype=dtype)
    x = np.asarray([1.0 + imaginary, -2.0, 0.75], dtype=dtype)
    y = np.asarray([-0.5 * imaginary, 3.0, 1.25], dtype=dtype)
    batch_x = np.stack((x, 0.5 * x))
    batch_y = np.stack((y, -1.5 * y))
    space = sc.DenseCoordinateSpace((3,), ctx)
    op = sc.DiagonalLinOp(_array(ctx, diagonal), space, ctx)
    return _reference(
        op,
        family="diagonal",
        x=_array(ctx, x),
        y=_array(ctx, y),
        expected_apply=diagonal * x,
        expected_rapply=diagonal.conj() * y,
        batch_x=_array(ctx, batch_x),
        batch_y=_array(ctx, batch_y),
        expected_vapply=batch_x * diagonal,
        expected_rvapply=batch_y * diagonal.conj(),
        reference_matrix=np.diag(diagonal),
        target_ctx=_target_context(ctx),
    )


def matrix_free_linop_case(ctx: sc.Context) -> LinOpCase:
    data = _matrix_data(ctx)
    space = sc.DenseCoordinateSpace((2,), ctx)
    factor = 2.0 - (0.5j if _complex(ctx) else 0.0)
    adjoint_factor = factor.conjugate()
    op = sc.MatrixFreeLinOp(
        lambda value: factor * value,
        lambda value: adjoint_factor * value,
        space,
        space,
        ctx,
    )
    x = data["x"]
    y = data["z"]
    batch_x = data["batch_x"]
    batch_y = np.stack((y, -0.5 * y))
    matrix = factor * np.eye(2, dtype=_numpy_dtype(ctx))
    return _reference(
        op,
        family="matrix-free",
        x=_array(ctx, x),
        y=_array(ctx, y),
        expected_apply=factor * x,
        expected_rapply=adjoint_factor * y,
        batch_x=_array(ctx, batch_x),
        batch_y=_array(ctx, batch_y),
        expected_vapply=factor * batch_x,
        expected_rvapply=adjoint_factor * batch_y,
        reference_matrix=matrix,
        target_ctx=_target_context(ctx),
    )


def _coordinate_algebra_cases(ctx: sc.Context) -> tuple[LinOpCase, ...]:
    data = _matrix_data(ctx)
    x_space = sc.DenseCoordinateSpace((2,), ctx)
    y_space = sc.DenseCoordinateSpace((3,), ctx)
    z_space = sc.DenseCoordinateSpace((2,), ctx)
    a = sc.DenseLinOp(_array(ctx, data["A"]), x_space, y_space, ctx)
    c = sc.DenseLinOp(_array(ctx, data["C"]), x_space, y_space, ctx)
    b = sc.DenseLinOp(_array(ctx, data["B"]), y_space, z_space, ctx)
    x = _array(ctx, data["x"])
    y = _array(ctx, data["y"])
    batch_x = _array(ctx, data["batch_x"])
    batch_y = _array(ctx, data["batch_y"])
    alpha = 1.25 - (0.5j if _complex(ctx) else 0.0)

    cases = []
    for family, op, matrix in (
        ("scaled", sc.ScaledLinOp(alpha, a), alpha * data["A"]),
        ("sum", sc.SumLinOp((a, c)), data["A"] + data["C"]),
    ):
        adjoint = matrix.conj().T
        cases.append(
            _reference(
                op,
                family=family,
                x=x,
                y=y,
                expected_apply=matrix @ data["x"],
                expected_rapply=adjoint @ data["y"],
                batch_x=batch_x,
                batch_y=batch_y,
                expected_vapply=data["batch_x"] @ matrix.T,
                expected_rvapply=data["batch_y"] @ adjoint.T,
                reference_matrix=matrix,
                target_ctx=_target_context(ctx),
            )
        )

    composed_matrix = data["B"] @ data["A"]
    composed = sc.ComposedLinOp(b, a)
    composed_adjoint = composed_matrix.conj().T
    batch_z = np.stack((data["z"], -0.5 * data["z"]))
    cases.append(
        _reference(
            composed,
            family="composed",
            x=x,
            y=_array(ctx, data["z"]),
            expected_apply=composed_matrix @ data["x"],
            expected_rapply=composed_adjoint @ data["z"],
            batch_x=batch_x,
            batch_y=_array(ctx, batch_z),
            expected_vapply=data["batch_x"] @ composed_matrix.T,
            expected_rvapply=batch_z @ composed_adjoint.T,
            reference_matrix=composed_matrix,
            target_ctx=_target_context(ctx),
        )
    )

    identity = sc.IdentityLinOp(x_space)
    cases.append(
        _reference(
            identity,
            family="identity",
            x=x,
            y=x,
            expected_apply=data["x"],
            expected_rapply=data["x"],
            batch_x=batch_x,
            batch_y=batch_x,
            expected_vapply=data["batch_x"],
            expected_rvapply=data["batch_x"],
            reference_matrix=np.eye(2, dtype=_numpy_dtype(ctx)),
            target_ctx=_target_context(ctx),
        )
    )

    zero = sc.ZeroLinOp(x_space, y_space, ctx)
    zeros_y = np.zeros(3, dtype=_numpy_dtype(ctx))
    zeros_x = np.zeros(2, dtype=_numpy_dtype(ctx))
    cases.append(
        _reference(
            zero,
            family="zero",
            x=x,
            y=y,
            expected_apply=zeros_y,
            expected_rapply=zeros_x,
            batch_x=batch_x,
            batch_y=batch_y,
            expected_vapply=np.zeros((2, 3), dtype=_numpy_dtype(ctx)),
            expected_rvapply=np.zeros((2, 2), dtype=_numpy_dtype(ctx)),
            reference_matrix=np.zeros((3, 2), dtype=_numpy_dtype(ctx)),
            target_ctx=_target_context(ctx),
        )
    )
    return tuple(cases)


def _tree_data(ctx: sc.Context) -> dict[str, Any]:
    imaginary = 1j if _complex(ctx) else 0.0
    dtype = _numpy_dtype(ctx)
    matrices = (
        (
            np.asarray([[1.0 + 0.25 * imaginary, 2.0]], dtype=dtype),
            np.asarray([[4.0]], dtype=dtype),
        ),
        (
            np.asarray([[1.0, 0.0], [0.0, 2.0 - 0.5 * imaginary]], dtype=dtype),
            np.asarray([[3.0], [-1.0 + 0.25 * imaginary]], dtype=dtype),
        ),
    )
    x = (
        np.asarray([1.0 + imaginary, 2.0], dtype=dtype),
        np.asarray([3.0 - 0.5 * imaginary], dtype=dtype),
    )
    y = (
        np.asarray([2.0 + 0.25 * imaginary], dtype=dtype),
        np.asarray([1.0, -2.0 + imaginary], dtype=dtype),
    )
    return {"matrices": matrices, "x": x, "y": y}


def _tree_spaces_and_blocks(ctx: sc.Context):
    x_spaces = (sc.DenseCoordinateSpace((2,), ctx), sc.DenseCoordinateSpace((1,), ctx))
    y_spaces = (sc.DenseCoordinateSpace((1,), ctx), sc.DenseCoordinateSpace((2,), ctx))
    matrices = _tree_data(ctx)["matrices"]
    rows = tuple(
        tuple(
            sc.DenseLinOp(_array(ctx, matrices[i][j]), x_spaces[j], y_spaces[i], ctx)
            for j in range(2)
        )
        for i in range(2)
    )
    return x_spaces, y_spaces, rows


def _tree_expected_apply(matrices: Any, x: Any) -> tuple[np.ndarray, ...]:
    return tuple(sum(matrix @ value for matrix, value in zip(row, x)) for row in matrices)


def _tree_expected_rapply(matrices: Any, y: Any) -> tuple[np.ndarray, ...]:
    return tuple(
        sum((matrices[row][column].conj().T @ y[row] for row in range(len(matrices))))
        for column in range(len(matrices[0]))
    )


def _stack_tree(values: Sequence[tuple[np.ndarray, ...]]) -> tuple[np.ndarray, ...]:
    return tuple(np.stack(leaves) for leaves in zip(*values))


def _stack_values(first: Any, second: Any) -> Any:
    if isinstance(first, tuple):
        return tuple(_stack_values(left, right) for left, right in zip(first, second))
    return np.stack((first, second))


def _tree_case(
    op: sc.LinOp,
    *,
    family: str,
    x: Any,
    y: Any,
    apply_fn: Any,
    rapply_fn: Any,
) -> LinOpCase:
    second_x = tuple(-0.5 * leaf for leaf in x) if isinstance(x, tuple) else -0.5 * x
    second_y = tuple(1.5 * leaf for leaf in y) if isinstance(y, tuple) else 1.5 * y
    batch_x_np = _stack_values(x, second_x)
    batch_y_np = _stack_values(y, second_y)
    expected_apply = apply_fn(x)
    expected_rapply = rapply_fn(y)
    expected_vapply = _stack_values(expected_apply, apply_fn(second_x))
    expected_rvapply = _stack_values(expected_rapply, rapply_fn(second_y))

    def convert(value: Any) -> Any:
        if isinstance(value, tuple):
            return tuple(convert(leaf) for leaf in value)
        return _array(op.ctx, value)

    return _reference(
        op,
        family=family,
        x=convert(x),
        y=convert(y),
        expected_apply=expected_apply,
        expected_rapply=expected_rapply,
        batch_x=convert(batch_x_np),
        batch_y=convert(batch_y_np),
        expected_vapply=expected_vapply,
        expected_rvapply=expected_rvapply,
        reference_matrix=None,
        target_ctx=_target_context(op.ctx),
    )


def tree_linop_cases(ctx: sc.Context) -> tuple[LinOpCase, ...]:
    data = _tree_data(ctx)
    matrices = data["matrices"]
    x_spaces, y_spaces, rows = _tree_spaces_and_blocks(ctx)
    x, y = data["x"], data["y"]
    cases = []

    diagonal_matrices = (matrices[0][0], matrices[1][1])
    diagonal = sc.BlockDiagonalLinOp((rows[0][0], rows[1][1]))
    cases.append(
        _tree_case(
            diagonal,
            family="block-diagonal",
            x=x,
            y=y,
            apply_fn=lambda value: tuple(
                matrix @ leaf for matrix, leaf in zip(diagonal_matrices, value)
            ),
            rapply_fn=lambda value: tuple(
                matrix.conj().T @ leaf for matrix, leaf in zip(diagonal_matrices, value)
            ),
        )
    )

    block_matrix = sc.BlockMatrixLinOp(rows)
    cases.append(
        _tree_case(
            block_matrix,
            family="block-matrix",
            x=x,
            y=y,
            apply_fn=lambda value: _tree_expected_apply(matrices, value),
            rapply_fn=lambda value: _tree_expected_rapply(matrices, value),
        )
    )

    stacked_rows = (rows[0][0], rows[1][0])
    stacked = sc.StackedLinOp.from_operators(stacked_rows)
    cases.append(
        _tree_case(
            stacked,
            family="stacked",
            x=x[0],
            y=y,
            apply_fn=lambda value: tuple(matrix @ value for matrix in (matrices[0][0], matrices[1][0])),
            rapply_fn=lambda value: sum(
                matrix.conj().T @ leaf
                for matrix, leaf in zip((matrices[0][0], matrices[1][0]), value)
            ),
        )
    )

    sum_parts = (rows[1][0], rows[1][1])
    sum_to_single = sc.SumToSingleLinOp.from_operators(sum_parts)
    cases.append(
        _tree_case(
            sum_to_single,
            family="sum-to-single",
            x=x,
            y=y[1],
            apply_fn=lambda value: matrices[1][0] @ value[0] + matrices[1][1] @ value[1],
            rapply_fn=lambda value: (
                matrices[1][0].conj().T @ value,
                matrices[1][1].conj().T @ value,
            ),
        )
    )
    return tuple(cases)


def linop_cases(
    *,
    dtypes: Iterable[Any] = NUMPY_LINOP_DTYPES,
    check_levels: Iterable[sc.CheckLevel | str] = ("standard",),
    include_weighted: bool = True,
) -> tuple[LinOpCase, ...]:
    """Generate all concrete public LinOp families with direct references."""
    cases = []
    for check_level in check_levels:
        for dtype in dtypes:
            ctx = _context(dtype, check_level)
            cases.extend(
                (
                    dense_linop_case(ctx),
                    sparse_linop_case(ctx),
                    diagonal_linop_case(ctx),
                    matrix_free_linop_case(ctx),
                )
            )
            cases.extend(_coordinate_algebra_cases(ctx))
            cases.extend(tree_linop_cases(ctx))
            if include_weighted:
                cases.append(dense_linop_case(ctx, weighted=True))
    return tuple(cases)


def backend_linop_cases(ctx: sc.Context) -> tuple[LinOpCase, ...]:
    """Generate backend-portable dense, diagonal, and matrix-free operators."""
    return dense_linop_case(ctx), diagonal_linop_case(ctx), matrix_free_linop_case(ctx)
