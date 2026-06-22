from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any, Literal

import numpy as np

import spacecore as sc
from spacecore.space.checks import BackendCheck, DTypeCheck, FieldCheck, ShapeCheck

from ._arrays import DEFAULT_DENSE_SHAPES, dense_array_case
from ._hermitian import hermitian_case
from ._metrics import spd_metric_case
from ._protocol import GeneratedCase
from ._seed import DEFAULT_SEED, resolve_rng


SpaceCase = GeneratedCase[sc.Space]
NUMPY_SPACE_DTYPES = (np.float32, np.float64, np.complex64, np.complex128)


def _dtype_name(dtype: Any) -> str:
    return np.dtype(dtype).name


def _target_dtype(dtype: Any) -> np.dtype[Any]:
    dtype = np.dtype(dtype)
    mapping = {
        np.dtype(np.float32): np.dtype(np.float64),
        np.dtype(np.float64): np.dtype(np.float32),
        np.dtype(np.complex64): np.dtype(np.complex128),
        np.dtype(np.complex128): np.dtype(np.complex64),
    }
    return mapping[dtype]


def _context(dtype: Any, check_level: sc.CheckLevel | str) -> sc.Context:
    return sc.Context(sc.NumpyOps(), dtype=dtype, check_level=check_level)


def _scalar(ctx: sc.Context, real: float, imag: float = 0.0) -> Any:
    value = real + 1j * imag if ctx.ops.is_complex_dtype(ctx.dtype) else real
    return ctx.asarray(np.asarray(value, dtype=np.dtype(ctx.dtype)))


def _wrong_shape(shape: tuple[int, ...]) -> tuple[int, ...]:
    if not shape:
        return (1,)
    return (shape[0] + 1,) + shape[1:]


def _alternate_dtype(dtype: Any) -> np.dtype[Any]:
    dtype = np.dtype(dtype)
    if dtype == np.dtype(np.float32):
        return np.dtype(np.float64)
    if dtype == np.dtype(np.float64):
        return np.dtype(np.float32)
    if dtype == np.dtype(np.complex64):
        return np.dtype(np.complex128)
    return np.dtype(np.complex64)


def _dense_reference(
    space: sc.CoordinateSpace,
    *,
    rng: np.random.Generator,
) -> dict[str, Any]:
    ctx = space.ctx
    generated = tuple(
        dense_array_case(ctx, space.shape, seed=None, rng=rng) for _ in range(3)
    )
    batch_x = dense_array_case(ctx, space.shape, batch_shape=(2,), seed=None, rng=rng)
    batch_y = dense_array_case(ctx, space.shape, batch_shape=(2,), seed=None, rng=rng)
    wrong_shape = _wrong_shape(tuple(space.shape))
    wrong_dtype = _alternate_dtype(ctx.dtype)
    invalid_shape = np.zeros(wrong_shape, dtype=np.dtype(ctx.dtype))
    invalid_dtype = np.zeros(space.shape, dtype=wrong_dtype)
    invalid_batch = np.zeros((2,) + wrong_shape, dtype=np.dtype(ctx.dtype))
    if space.field == "real":
        invalid_field = np.ones(space.shape, dtype=np.complex64)
    else:
        invalid_field = np.ones(space.shape, dtype=ctx.ops.real_dtype(ctx.dtype))
    target_ctx = _context(_target_dtype(ctx.dtype), space.check_level)
    return {
        "ctx": ctx,
        "dtype": ctx.dtype,
        "field": space.field,
        "check_level": space.check_level,
        "shape": tuple(space.shape),
        "x": generated[0].obj,
        "y": generated[1].obj,
        "z": generated[2].obj,
        "x_numpy": generated[0].reference["array"],
        "y_numpy": generated[1].reference["array"],
        "z_numpy": generated[2].reference["array"],
        "batch_x": batch_x.obj,
        "batch_y": batch_y.obj,
        "batch_x_numpy": batch_x.reference["array"],
        "batch_y_numpy": batch_y.reference["array"],
        "batch_shape": (2,),
        "a": _scalar(ctx, 0.5, 0.25),
        "b": _scalar(ctx, -1.25, 0.5),
        "invalid_shape": invalid_shape,
        "invalid_dtype": invalid_dtype,
        "invalid_field": invalid_field,
        "invalid_backend": [0.0] * max(1, space.size),
        "invalid_batch": invalid_batch,
        "target_ctx": target_ctx,
    }


def dense_coordinate_space_cases(
    *,
    shapes: Sequence[Sequence[int]] = DEFAULT_DENSE_SHAPES,
    dtypes: Sequence[Any] = NUMPY_SPACE_DTYPES,
    check_levels: Iterable[sc.CheckLevel | str] = sc.CHECK_LEVELS,
    seed: int | None = DEFAULT_SEED,
) -> tuple[SpaceCase, ...]:
    """Generate dense coordinate spaces with values and policy metadata."""
    rng = resolve_rng(seed=seed)
    cases: list[SpaceCase] = []
    for check_level in check_levels:
        for dtype in dtypes:
            ctx = _context(dtype, check_level)
            for shape_like in shapes:
                shape = tuple(int(dimension) for dimension in shape_like)
                space = sc.DenseCoordinateSpace(shape, ctx)
                reference = _dense_reference(space, rng=rng)
                cases.append(
                    GeneratedCase(
                        obj=space,
                        reference=reference,
                        capabilities=frozenset(
                            {"vector", "coordinate", "inner_product", "batching", space.field}
                        ),
                        id=(
                            f"dense-coordinate-{_dtype_name(dtype)}-"
                            f"{'scalar' if not shape else 'x'.join(map(str, shape))}-"
                            f"checks-{check_level}"
                        ),
                    )
                )
    return tuple(cases)


def dense_vector_space_cases(
    *,
    sizes: Sequence[int] = (1, 3),
    dtypes: Sequence[Any] = NUMPY_SPACE_DTYPES,
    check_levels: Iterable[sc.CheckLevel | str] = sc.CHECK_LEVELS,
    seed: int | None = DEFAULT_SEED,
) -> tuple[SpaceCase, ...]:
    """Generate one-dimensional dense vector spaces."""
    rng = resolve_rng(seed=seed)
    cases: list[SpaceCase] = []
    for check_level in check_levels:
        for dtype in dtypes:
            ctx = _context(dtype, check_level)
            for size in sizes:
                space = sc.DenseVectorSpace((int(size),), ctx)
                cases.append(
                    GeneratedCase(
                        obj=space,
                        reference=_dense_reference(space, rng=rng),
                        capabilities=frozenset(
                            {
                                "vector",
                                "coordinate",
                                "inner_product",
                                "star",
                                "batching",
                                space.field,
                            }
                        ),
                        id=(
                            f"dense-vector-{_dtype_name(dtype)}-{size}-"
                            f"checks-{check_level}"
                        ),
                    )
                )
    return tuple(cases)


class MatrixInnerProduct(sc.InnerProduct):
    """Dense SPD metric geometry used by generated vector-space tests."""

    def __init__(self, matrix: Any, inverse: Any) -> None:
        self.matrix = matrix
        self.inverse = inverse

    def _apply(self, ops: Any, matrix: Any, x: Any) -> Any:
        return ops.einsum("ij,...j->...i", matrix, x)

    def inner(self, ops: Any, x: Any, y: Any) -> Any:
        return ops.vdot(x, self.riesz(ops, y))

    def riesz(self, ops: Any, x: Any) -> Any:
        return self._apply(ops, self.matrix, x)

    def riesz_inverse(self, ops: Any, x: Any) -> Any:
        return self._apply(ops, self.inverse, x)

    def convert(self, ctx: sc.Context) -> MatrixInnerProduct:
        return type(self)(ctx.asarray(self.matrix), ctx.asarray(self.inverse))

    def validate_for(self, space: sc.InnerProductSpace) -> None:
        if not isinstance(space, sc.CoordinateSpace) or len(space.shape) != 1:
            raise TypeError("MatrixInnerProduct requires a one-dimensional CoordinateSpace.")
        expected = (space.size, space.size)
        if tuple(self.matrix.shape) != expected or tuple(self.inverse.shape) != expected:
            raise ValueError(f"MatrixInnerProduct matrices must have shape {expected}.")
        if space.ops.get_dtype(self.matrix) != space.dtype:
            raise TypeError("MatrixInnerProduct matrix dtype must match the space dtype.")

    @property
    def is_euclidean(self) -> bool:
        return False

    def __eq__(self, other: Any) -> bool:
        return isinstance(other, MatrixInnerProduct) and bool(
            np.allclose(np.asarray(self.matrix), np.asarray(other.matrix))
        )


def inner_product_space_cases(
    *,
    size: int = 3,
    dtypes: Sequence[Any] = NUMPY_SPACE_DTYPES,
    check_level: sc.CheckLevel | str = "standard",
    seed: int | None = DEFAULT_SEED,
) -> tuple[SpaceCase, ...]:
    """Generate Euclidean and full-SPD inner-product vector spaces."""
    rng = resolve_rng(seed=seed)
    cases: list[SpaceCase] = []
    for dtype in dtypes:
        ctx = _context(dtype, check_level)
        euclidean = sc.DenseVectorSpace((size,), ctx)
        euclidean_ref = _dense_reference(euclidean, rng=rng)
        euclidean_ref["geometry"] = "euclidean"
        cases.append(
            GeneratedCase(
                obj=euclidean,
                reference=euclidean_ref,
                capabilities=frozenset(
                    {"vector", "inner_product", "euclidean", "batching", euclidean.field}
                ),
                id=f"inner-euclidean-{_dtype_name(dtype)}",
            )
        )

        metric = spd_metric_case(ctx, size, condition_number=8.0, seed=None, rng=rng)
        geometry = MatrixInnerProduct(metric.obj, ctx.asarray(metric.reference["inverse"]))
        spd = sc.DenseVectorSpace((size,), ctx, geometry=geometry)
        spd_ref = _dense_reference(spd, rng=rng)
        spd_ref.update(metric=metric.reference, geometry="spd")
        cases.append(
            GeneratedCase(
                obj=spd,
                reference=spd_ref,
                capabilities=frozenset(
                    {"vector", "inner_product", "spd_metric", "batching", spd.field}
                ),
                id=f"inner-spd-{_dtype_name(dtype)}",
            )
        )
    return tuple(cases)


class _CoordinateOnlySpace(sc.CoordinateSpace):
    """Minimal coordinate space used to test truthful tree capability intersections."""

    def _local_checks(self) -> tuple[Any, ...]:
        return BackendCheck(), ShapeCheck(), FieldCheck(), DTypeCheck()

    def zeros(self) -> Any:
        return self.ops.zeros(self.shape, dtype=self.dtype)

    def add(self, x: Any, y: Any) -> Any:
        return x + y

    def add_batch(self, x: Any, y: Any) -> Any:
        return x + y

    def scale(self, a: Any, x: Any) -> Any:
        return a * x

    def scale_batch(self, a: Any, x: Any) -> Any:
        return a * x

    def flatten(self, x: Any) -> Any:
        return x.reshape((-1,))

    def unflatten(self, vector: Any) -> Any:
        return vector.reshape(self.shape)

    def flatten_batch(self, values: Any) -> Any:
        return values.reshape((values.shape[0], -1))

    def unflatten_batch(self, vectors: Any) -> Any:
        return vectors.reshape((vectors.shape[0],) + self.shape)

    def _convert(self, new_ctx: sc.Context) -> _CoordinateOnlySpace:
        return type(self)(self.shape, new_ctx)


TreeProfile = Literal["inner", "mixed"]


def _tree_layout(kind: Literal["tuple", "nested", "dict"]) -> tuple[Any, tuple[tuple[int, ...], ...]]:
    if kind == "tuple":
        return (0, 0), ((2,), (3,))
    if kind == "nested":
        return (0, [0, (0,)]), ((2,), (1,), (2, 2))
    if kind == "dict":
        return {"point": 0, "weight": 0}, ((2,), (1,))
    raise ValueError(f"Unknown tree kind {kind!r}.")


def _tree_value(space: sc.TreeSpace, leaves: Sequence[Any]) -> Any:
    return space.unflatten_tree(tuple(leaves))


def _tree_reference(
    space: sc.TreeSpace,
    *,
    rng: np.random.Generator,
    profile: TreeProfile,
) -> dict[str, Any]:
    leaf_sets = []
    for _ in range(3):
        leaf_sets.append(
            tuple(
                dense_array_case(space.ctx, leaf.shape, seed=None, rng=rng).obj
                for leaf in space.leaf_spaces
            )
        )
    batch_sets = []
    for _ in range(2):
        batch_sets.append(
            tuple(
                dense_array_case(
                    space.ctx,
                    leaf.shape,
                    batch_shape=(2,),
                    seed=None,
                    rng=rng,
                ).obj
                for leaf in space.leaf_spaces
            )
        )
    invalid_leaves = list(leaf_sets[0])
    first_shape = tuple(space.leaf_spaces[0].shape)
    invalid_leaves[0] = space.ctx.asarray(
        np.zeros(_wrong_shape(first_shape), dtype=np.dtype(space.ctx.dtype))
    )
    mismatch = list(leaf_sets[0]) if isinstance(space.unflatten_tree(leaf_sets[0]), tuple) else tuple(leaf_sets[0])
    return {
        "ctx": space.ctx,
        "dtype": space.dtype,
        "field": space.field,
        "check_level": space.check_level,
        "profile": profile,
        "x": _tree_value(space, leaf_sets[0]),
        "y": _tree_value(space, leaf_sets[1]),
        "z": _tree_value(space, leaf_sets[2]),
        "batch_x": _tree_value(space, batch_sets[0]),
        "batch_y": _tree_value(space, batch_sets[1]),
        "batch_shape": (2,),
        "a": _scalar(space.ctx, 0.5, 0.25),
        "b": _scalar(space.ctx, -1.25, 0.5),
        "leaf_paths": space.leaf_paths,
        "invalid_leaf": _tree_value(space, invalid_leaves),
        "mismatch": mismatch,
        "target_ctx": _context(_target_dtype(space.dtype), space.check_level),
    }


def tree_space_generated_cases(
    *,
    dtype: Any = np.float64,
    check_level: sc.CheckLevel | str = "standard",
    include_mixed: bool = True,
    seed: int | None = DEFAULT_SEED,
) -> tuple[SpaceCase, ...]:
    """Generate tuple, nested, dictionary, and mixed-capability tree spaces."""
    rng = resolve_rng(seed=seed)
    ctx = _context(dtype, check_level)
    cases: list[SpaceCase] = []
    for kind in ("tuple", "nested", "dict"):
        template, shapes = _tree_layout(kind)
        leaves = tuple(sc.DenseCoordinateSpace(shape, ctx) for shape in shapes)
        space = sc.TreeSpace.from_template(template, leaves, ctx=ctx)
        cases.append(
            GeneratedCase(
                obj=space,
                reference=_tree_reference(space, rng=rng, profile="inner"),
                capabilities=frozenset(
                    {"vector", "coordinate", "tree", "inner_product", "batching", space.field}
                ),
                id=f"tree-{kind}-inner-{_dtype_name(dtype)}-checks-{check_level}",
            )
        )
    if include_mixed:
        template, shapes = _tree_layout("tuple")
        leaves = (
            sc.DenseCoordinateSpace(shapes[0], ctx),
            _CoordinateOnlySpace(shapes[1], ctx),
        )
        space = sc.TreeSpace.from_template(template, leaves, ctx=ctx)
        cases.append(
            GeneratedCase(
                obj=space,
                reference=_tree_reference(space, rng=rng, profile="mixed"),
                capabilities=frozenset(
                    {"vector", "coordinate", "tree", "mixed_capabilities", "batching", space.field}
                ),
                id=f"tree-tuple-mixed-{_dtype_name(dtype)}-checks-{check_level}",
            )
        )
    return tuple(cases)


def _jordan_dense_reference(
    space: sc.CoordinateSpace,
    *,
    rng: np.random.Generator,
) -> dict[str, Any]:
    reference = _dense_reference(space, rng=rng)
    reference["kind"] = "elementwise"
    return reference


def _hermitian_reference(
    space: sc.HermitianSpace,
    *,
    rng: np.random.Generator,
) -> dict[str, Any]:
    values = tuple(
        hermitian_case(
            space.ctx,
            space.n,
            complex=space.ctx.ops.is_complex_dtype(space.dtype),
            seed=None,
            rng=rng,
        ).obj
        for _ in range(3)
    )
    bad = space.ctx.asarray(
        np.asarray([[1.0, 2.0], [0.0, 1.0]], dtype=np.dtype(space.dtype))
    )
    return {
        "ctx": space.ctx,
        "dtype": space.dtype,
        "field": space.field,
        "check_level": space.check_level,
        "kind": "hermitian",
        "x": values[0],
        "y": values[1],
        "z": values[2],
        "a": _scalar(space.ctx, 0.5),
        "b": _scalar(space.ctx, -1.25),
        "invalid_spectral": bad,
        "target_ctx": _context(_target_dtype(space.dtype), space.check_level),
    }


def jordan_space_cases(
    *,
    check_level: sc.CheckLevel | str = "strict",
    seed: int | None = DEFAULT_SEED,
) -> tuple[SpaceCase, ...]:
    """Generate elementwise, Hermitian, stacked, and tree Jordan spaces."""
    rng = resolve_rng(seed=seed)
    cases: list[SpaceCase] = []
    for dtype in (np.float64, np.complex128):
        ctx = _context(dtype, check_level)
        elementwise = sc.ElementwiseJordanSpace((3,), ctx)
        cases.append(
            GeneratedCase(
                obj=elementwise,
                reference=_jordan_dense_reference(elementwise, rng=rng),
                capabilities=frozenset(
                    {"vector", "inner_product", "star", "jordan", "spectral", elementwise.field}
                ),
                id=f"jordan-elementwise-{_dtype_name(dtype)}",
            )
        )
        hermitian = sc.HermitianSpace(2, atol=1e-10, rtol=1e-10, ctx=ctx)
        cases.append(
            GeneratedCase(
                obj=hermitian,
                reference=_hermitian_reference(hermitian, rng=rng),
                capabilities=frozenset(
                    {"vector", "inner_product", "star", "jordan", "spectral", hermitian.field}
                ),
                id=f"jordan-hermitian-{_dtype_name(dtype)}",
            )
        )

    real_ctx = _context(np.float64, check_level)
    euclidean_elementwise = sc.EuclideanElementwiseJordanSpace((3,), real_ctx)
    cases.append(
        GeneratedCase(
            obj=euclidean_elementwise,
            reference=_jordan_dense_reference(euclidean_elementwise, rng=rng),
            capabilities=frozenset(
                {
                    "vector",
                    "inner_product",
                    "star",
                    "jordan",
                    "spectral",
                    "euclidean",
                    euclidean_elementwise.field,
                }
            ),
            id="jordan-euclidean-elementwise-float64",
        )
    )

    base = sc.ElementwiseJordanSpace((3,), real_ctx)
    stacked = base.stacked(2)
    stacked_ref = _jordan_dense_reference(stacked, rng=rng)
    stacked_ref["kind"] = "stacked"
    cases.append(
        GeneratedCase(
            obj=stacked,
            reference=stacked_ref,
            capabilities=frozenset(
                {"vector", "inner_product", "star", "jordan", "spectral", "stacked", "real"}
            ),
            id="jordan-stacked-real",
        )
    )

    leaves = (
        sc.ElementwiseJordanSpace((2,), real_ctx),
        sc.ElementwiseJordanSpace((1,), real_ctx),
    )
    tree = sc.TreeSpace.from_leaf_spaces(leaves, real_ctx)
    tree_ref = _tree_reference(tree, rng=rng, profile="inner")
    tree_ref["kind"] = "tree"
    cases.append(
        GeneratedCase(
            obj=tree,
            reference=tree_ref,
            capabilities=frozenset(
                {"vector", "inner_product", "star", "jordan", "spectral", "tree", "real"}
            ),
            id="jordan-tree-real",
        )
    )
    return tuple(cases)


def mixed_jordan_tree_case(
    *,
    check_level: sc.CheckLevel | str = "standard",
    seed: int | None = DEFAULT_SEED,
) -> SpaceCase:
    """Generate a tree whose mixed leaves must not advertise Jordan or star."""
    rng = resolve_rng(seed=seed)
    ctx = _context(np.float64, check_level)
    leaves = (
        sc.ElementwiseJordanSpace((2,), ctx),
        sc.DenseCoordinateSpace((1,), ctx),
    )
    space = sc.TreeSpace.from_leaf_spaces(leaves, ctx)
    return GeneratedCase(
        obj=space,
        reference=_tree_reference(space, rng=rng, profile="inner"),
        capabilities=frozenset({"vector", "inner_product", "tree", "mixed_capabilities"}),
        id="tree-mixed-no-jordan",
    )


def vector_space_law_cases(*, seed: int | None = DEFAULT_SEED) -> tuple[SpaceCase, ...]:
    """Return a representative deterministic set for generic vector-space laws."""
    dense = dense_coordinate_space_cases(check_levels=("standard",), seed=seed)
    vectors = dense_vector_space_cases(sizes=(3,), check_levels=("standard",), seed=seed)
    trees = (
        *tree_space_generated_cases(dtype=np.float64, include_mixed=False, seed=seed),
        *tree_space_generated_cases(dtype=np.complex128, include_mixed=False, seed=seed),
    )
    jordan = tuple(
        case
        for case in jordan_space_cases(check_level="standard", seed=seed)
        if case.reference["kind"] in {"elementwise", "stacked", "tree"}
    )
    return tuple(dense) + tuple(vectors) + tuple(trees) + jordan
