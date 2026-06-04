from __future__ import annotations

from typing import Any

import numpy as np
import pytest

import spacecore as sc


class FiniteSetSpace(sc.Space):
    def __init__(self, values: set[Any], ctx=None):
        super().__init__(ctx)
        self.values = values

    def _check_member(self, x: Any) -> None:
        if x not in self.values:
            raise ValueError("not a member")

    def _convert(self, new_ctx):
        return FiniteSetSpace(self.values, new_ctx)


class PairVectorSpace(sc.VectorSpace):
    def __init__(self, ctx=None):
        super().__init__(ctx)

    def zeros(self):
        return (0.0, 0.0)

    def add(self, x, y):
        return (x[0] + y[0], x[1] + y[1])

    def scale(self, a, x):
        return (a * x[0], a * x[1])

    def _convert(self, new_ctx):
        return PairVectorSpace(new_ctx)


class NonCoordinateInnerProductSpace(sc.InnerProductSpace):
    def __init__(self, ctx=None):
        super().__init__(ctx)
        self.geometry = sc.EuclideanInnerProduct()

    def zeros(self):
        return (0.0, 0.0)

    def add(self, x, y):
        return (x[0] + y[0], x[1] + y[1])

    def scale(self, a, x):
        return (a * x[0], a * x[1])

    def inner(self, x, y):
        return x[0] * y[0] + x[1] * y[1]

    def _convert(self, new_ctx):
        return NonCoordinateInnerProductSpace(new_ctx)


def test_minimal_space_membership_only():
    space = FiniteSetSpace({"a", "b"}, sc.Context(sc.NumpyOps(), enable_checks=True))

    space.check_member("a")
    with pytest.raises(ValueError, match="not a member"):
        space.check_member("c")

    assert not isinstance(space, sc.VectorSpace)
    assert not hasattr(space, "shape")


def test_vector_space_is_abstract_but_linear_capability_is_non_coordinate():
    with pytest.raises(TypeError):
        sc.VectorSpace()

    space = PairVectorSpace()

    assert isinstance(space, sc.VectorSpace)
    assert not isinstance(space, sc.CoordinateSpace)
    assert space.axpy(2.0, (1.0, 3.0), (-1.0, 4.0)) == (1.0, 10.0)


def test_coordinate_space_dense_vectors_matrices_tensors_and_stacking():
    ctx = sc.Context(sc.NumpyOps(), dtype=np.float64)
    for shape in ((3,), (2, 3), (2, 1, 3)):
        space = sc.DenseCoordinateSpace(shape, ctx)
        x = ctx.asarray(np.arange(space.size, dtype=float).reshape(shape))
        flat = space.flatten(x)
        batch = ctx.asarray(np.stack([np.asarray(x), np.asarray(x) + 1.0]))

        assert isinstance(space, sc.CoordinateSpace)
        assert tuple(flat.shape) == (space.size,)
        np.testing.assert_allclose(space.unflatten(flat), x)
        np.testing.assert_allclose(space.flatten_batch(batch), np.asarray(batch).reshape((2, -1)))
        assert space.stacked(2).shape == (2,) + shape


def test_dense_vector_space_is_only_one_dimensional():
    ctx = sc.Context(sc.NumpyOps(), dtype=np.float64)
    space = sc.DenseVectorSpace((4,), ctx)

    assert isinstance(space, sc.DenseVectorSpace)
    assert isinstance(space, sc.CoordinateSpace)
    assert isinstance(space, sc.InnerProductSpace)
    assert isinstance(space, sc.StarSpace)
    assert isinstance(space, sc.EuclideanJordanAlgebraSpace)

    with pytest.raises(ValueError, match="one-dimensional"):
        sc.DenseVectorSpace((2, 2), ctx)


def test_non_coordinate_inner_product_space_norm():
    space = NonCoordinateInnerProductSpace()

    assert isinstance(space, sc.InnerProductSpace)
    assert not isinstance(space, sc.CoordinateSpace)
    assert space.inner((1.0, 2.0), (3.0, 4.0)) == 11.0
    assert float(space.norm((3.0, 4.0))) == 5.0


def test_star_involution_and_conjugate_linearity():
    ctx = sc.Context(sc.NumpyOps(), dtype=np.complex128)
    space = sc.DenseVectorSpace((2,), ctx)
    x = ctx.asarray([1.0 + 2.0j, -3.0 + 0.5j])
    alpha = 2.0 - 3.0j

    np.testing.assert_allclose(space.star(space.star(x)), x)
    np.testing.assert_allclose(space.star(alpha * x), np.conj(alpha) * np.asarray(space.star(x)))

    herm = sc.HermitianSpace(2, ctx=ctx)
    h = ctx.asarray([[1.0 + 0j, 2.0 - 1.0j], [2.0 + 1.0j, 3.0 + 0j]])
    np.testing.assert_allclose(herm.star(herm.star(h)), h)


def test_jordan_identity_for_dense_vectors_and_hermitian_space():
    ctx = sc.Context(sc.NumpyOps(), dtype=np.float64)
    vector = sc.DenseVectorSpace((3,), ctx)
    x = ctx.asarray([1.0, 2.0, -1.0])
    y = ctx.asarray([0.5, -3.0, 4.0])
    z = ctx.asarray([2.0, 1.0, 0.25])

    lhs = vector.inner(vector.jordan(x, y), z)
    rhs = vector.inner(y, vector.jordan(x, z))
    np.testing.assert_allclose(lhs, rhs)
    np.testing.assert_allclose(vector.spectral_apply(x, lambda t: t * t), x * x)

    herm = sc.HermitianSpace(2, ctx=ctx)
    a = ctx.asarray([[1.0, 0.25], [0.25, 2.0]])
    b = ctx.asarray([[0.5, -0.75], [-0.75, 3.0]])
    c = ctx.asarray([[2.0, 1.0], [1.0, -1.0]])
    np.testing.assert_allclose(herm.inner(herm.jordan(a, b), c), herm.inner(b, herm.jordan(a, c)))


class PairCoordinateSpace(sc.CoordinateSpace):
    def __init__(self, ctx=None):
        super().__init__((2,), ctx)

    def zeros(self):
        return (0.0, 0.0)

    def add(self, x, y):
        return (x[0] + y[0], x[1] + y[1])

    def scale(self, a, x):
        return (a * x[0], a * x[1])

    def flatten(self, x):
        return self.ctx.asarray([x[0], x[1]])

    def unflatten(self, v):
        return (float(v[0]), float(v[1]))

    def _convert(self, new_ctx):
        return PairCoordinateSpace(new_ctx)


def test_product_space_baseline_and_specialized_capabilities():
    ctx = sc.Context(sc.NumpyOps(), dtype=np.float64)
    base_product = sc.ProductSpace((PairCoordinateSpace(ctx), PairCoordinateSpace(ctx)), ctx)

    assert type(base_product) is sc.ProductSpace
    assert isinstance(base_product, sc.CoordinateSpace)
    assert not isinstance(base_product, sc.InnerProductSpace)
    assert not isinstance(base_product, sc.StarSpace)
    assert not isinstance(base_product, sc.JordanAlgebraSpace)

    inner_product = sc.ProductSpace((sc.DenseCoordinateSpace((2,), ctx), sc.DenseCoordinateSpace((1,), ctx)), ctx)
    assert isinstance(inner_product, sc.ProductInnerProductSpace)
    assert isinstance(inner_product, sc.InnerProductSpace)
    assert not isinstance(inner_product, sc.JordanAlgebraSpace)

    jordan_product = sc.ProductSpace((sc.ElementwiseJordanSpace((2,), ctx), sc.HermitianSpace(2, ctx=ctx)), ctx)
    assert isinstance(jordan_product, sc.StarSpace)
    assert isinstance(jordan_product, sc.EuclideanJordanAlgebraSpace)

    with pytest.raises(TypeError, match="StarSpace"):
        sc.ProductStarSpace((sc.DenseCoordinateSpace((1,), ctx),), ctx)
    with pytest.raises(TypeError, match="EuclideanJordanAlgebraSpace"):
        sc.ProductEuclideanJordanAlgebraSpace((sc.DenseCoordinateSpace((1,), ctx),), ctx)


def test_stacked_space_capability_dispatch_matches_base():
    ctx = sc.Context(sc.NumpyOps(), dtype=np.float64)

    coordinate = PairCoordinateSpace(ctx).stacked(2)
    assert type(coordinate) is sc.StackedSpace
    assert isinstance(coordinate, sc.CoordinateSpace)
    assert not isinstance(coordinate, sc.InnerProductSpace)

    inner = sc.DenseCoordinateSpace((2,), ctx).stacked(2)
    assert isinstance(inner, sc.StackedInnerProductSpace)
    assert isinstance(inner, sc.InnerProductSpace)
    assert not isinstance(inner, sc.JordanAlgebraSpace)
    np.testing.assert_allclose(inner.inner(ctx.asarray([[1.0, 2.0], [3.0, 4.0]]), ctx.asarray([[5.0, 6.0], [7.0, 8.0]])), 70.0)

    jordan = sc.ElementwiseJordanSpace((2,), ctx).stacked(2)
    assert isinstance(jordan, sc.StarSpace)
    assert isinstance(jordan, sc.EuclideanJordanAlgebraSpace)
    x = ctx.asarray([[1.0, 2.0], [3.0, 4.0]])
    np.testing.assert_allclose(jordan.jordan(x, x), x * x)


def test_dense_coordinate_space_is_generic_not_jordan():
    ctx = sc.Context(sc.NumpyOps(), dtype=np.float64)
    space = sc.DenseCoordinateSpace((2, 2), ctx)

    assert isinstance(space, sc.CoordinateSpace)
    assert isinstance(space, sc.InnerProductSpace)
    assert not isinstance(space, sc.StarSpace)
    assert not isinstance(space, sc.JordanAlgebraSpace)
    assert not hasattr(space, "spectrum")


def test_elementwise_jordan_geometry_compatibility_real_complex_and_weighted():
    real_ctx = sc.Context(sc.NumpyOps(), dtype=np.float64)
    complex_ctx = sc.Context(sc.NumpyOps(), dtype=np.complex128)

    for ctx in (real_ctx, complex_ctx):
        space = sc.ElementwiseJordanSpace((2,), ctx)
        x = ctx.asarray([1.0, 2.0])
        y = ctx.asarray([3.0, -1.0])
        z = ctx.asarray([0.5, 4.0])
        np.testing.assert_allclose(space.inner(space.jordan(x, y), z), space.inner(y, space.jordan(x, z)))

    weights = real_ctx.asarray([2.0, 3.0])
    with pytest.raises(TypeError, match="EuclideanInnerProduct"):
        sc.ElementwiseJordanSpace((2,), real_ctx, geometry=sc.WeightedInnerProduct(weights))
    with pytest.raises(TypeError, match="EuclideanInnerProduct"):
        sc.DenseVectorSpace((2,), real_ctx, geometry=sc.WeightedInnerProduct(weights))

    weighted = sc.DenseCoordinateSpace((2,), real_ctx, geometry=sc.WeightedInnerProduct(weights))
    assert not isinstance(weighted, sc.JordanAlgebraSpace)


def test_no_repository_examples_instantiate_abstract_vector_space():
    from pathlib import Path
    import re

    root = Path(__file__).resolve().parents[2]
    pattern = re.compile(r"(?<!class )\b(?:sc\.)?VectorSpace\s*\(")
    checked_roots = ["spacecore", "tests", "docs", "test_generators", "README.md"]
    offenders = []
    for rel in checked_roots:
        path = root / rel
        files = [path] if path.is_file() else list(path.rglob("*.py")) + list(path.rglob("*.rst")) + list(path.rglob("*.md"))
        for file in files:
            if file == Path(__file__).resolve():
                continue
            text = file.read_text(encoding="utf-8")
            if pattern.search(text):
                offenders.append(str(file.relative_to(root)))
    assert offenders == []


class InnerCoordinateSpace(PairCoordinateSpace, sc.InnerProductSpace):
    def __init__(self, ctx=None):
        PairCoordinateSpace.__init__(self, ctx)
        self.geometry = sc.EuclideanInnerProduct()

    def inner(self, x, y):
        return x[0] * y[0] + x[1] * y[1]

    def riesz(self, x):
        return x

    def riesz_inverse(self, x):
        return x

    @property
    def is_euclidean(self):
        return True

    def _convert(self, new_ctx):
        return type(self)(new_ctx)


class StarCoordinateSpace(PairCoordinateSpace, sc.StarSpace):
    def star(self, x):
        return self.ops.conj(x)

    def _convert(self, new_ctx):
        return type(self)(new_ctx)


class JordanCoordinateSpace(PairCoordinateSpace, sc.JordanAlgebraSpace):
    def jordan(self, x, y):
        return x * y

    def spectrum(self, x):
        return x

    def spectral_decompose(self, x):
        return x, None

    def from_spectrum(self, eigvals, frame):
        if frame is not None:
            raise ValueError("frame must be None")
        return eigvals

    def _convert(self, new_ctx):
        return type(self)(new_ctx)


class InnerStarCoordinateSpace(InnerCoordinateSpace, sc.StarSpace):
    def star(self, x):
        return self.ops.conj(x)


class InnerJordanCoordinateSpace(InnerCoordinateSpace, sc.JordanAlgebraSpace):
    def jordan(self, x, y):
        return x * y

    def spectrum(self, x):
        return x

    def spectral_decompose(self, x):
        return x, None

    def from_spectrum(self, eigvals, frame):
        if frame is not None:
            raise ValueError("frame must be None")
        return eigvals


class StarJordanCoordinateSpace(StarCoordinateSpace, sc.JordanAlgebraSpace):
    def jordan(self, x, y):
        return x * y

    def spectrum(self, x):
        return x

    def spectral_decompose(self, x):
        return x, None

    def from_spectrum(self, eigvals, frame):
        if frame is not None:
            raise ValueError("frame must be None")
        return eigvals


class InnerStarJordanCoordinateSpace(InnerStarCoordinateSpace, sc.JordanAlgebraSpace):
    def jordan(self, x, y):
        return x * y

    def spectrum(self, x):
        return x

    def spectral_decompose(self, x):
        return x, None

    def from_spectrum(self, eigvals, frame):
        if frame is not None:
            raise ValueError("frame must be None")
        return eigvals


class EuclideanJordanCoordinateSpace(PairCoordinateSpace, sc.EuclideanJordanAlgebraSpace):
    def __init__(self, ctx=None):
        PairCoordinateSpace.__init__(self, ctx)
        self.geometry = sc.EuclideanInnerProduct()

    def inner(self, x, y):
        return x[0] * y[0] + x[1] * y[1]

    def riesz(self, x):
        return x

    def riesz_inverse(self, x):
        return x

    @property
    def is_euclidean(self):
        return True

    def jordan(self, x, y):
        return x * y

    def spectrum(self, x):
        return x

    def spectral_decompose(self, x):
        return x, None

    def from_spectrum(self, eigvals, frame):
        if frame is not None:
            raise ValueError("frame must be None")
        return eigvals

    def _convert(self, new_ctx):
        return type(self)(new_ctx)


def _capability_flags(space):
    return {
        sc.InnerProductSpace: isinstance(space, sc.InnerProductSpace),
        sc.StarSpace: isinstance(space, sc.StarSpace),
        sc.JordanAlgebraSpace: isinstance(space, sc.JordanAlgebraSpace),
        sc.EuclideanJordanAlgebraSpace: isinstance(space, sc.EuclideanJordanAlgebraSpace),
    }


@pytest.mark.parametrize(
    "factory, expected",
    [
        (PairCoordinateSpace, set()),
        (InnerCoordinateSpace, {sc.InnerProductSpace}),
        (StarCoordinateSpace, {sc.StarSpace}),
        (JordanCoordinateSpace, {sc.JordanAlgebraSpace}),
        (InnerStarCoordinateSpace, {sc.InnerProductSpace, sc.StarSpace}),
        (InnerJordanCoordinateSpace, {sc.InnerProductSpace, sc.JordanAlgebraSpace}),
        (StarJordanCoordinateSpace, {sc.StarSpace, sc.JordanAlgebraSpace}),
        (
            InnerStarJordanCoordinateSpace,
            {sc.InnerProductSpace, sc.StarSpace, sc.JordanAlgebraSpace},
        ),
        (
            EuclideanJordanCoordinateSpace,
            {sc.InnerProductSpace, sc.JordanAlgebraSpace, sc.EuclideanJordanAlgebraSpace},
        ),
        (
            sc.ElementwiseJordanSpace,
            {
                sc.InnerProductSpace,
                sc.StarSpace,
                sc.JordanAlgebraSpace,
                sc.EuclideanJordanAlgebraSpace,
            },
        ),
    ],
)
def test_product_and_stacked_preserve_exact_capability_combinations(factory, expected):
    ctx = sc.Context(sc.NumpyOps(), dtype=np.float64)
    if factory is sc.ElementwiseJordanSpace:
        base = factory((2,), ctx)
    else:
        base = factory(ctx)

    product = sc.ProductSpace((base, base), ctx)
    stacked = base.stacked(2)

    for result in (product, stacked):
        assert isinstance(result, sc.CoordinateSpace)
        flags = _capability_flags(result)
        for capability, present in flags.items():
            assert present is (capability in expected)


def test_baseline_product_and_stacked_do_not_expose_capability_methods():
    ctx = sc.Context(sc.NumpyOps(), dtype=np.float64)
    product = sc.ProductSpace((PairCoordinateSpace(ctx), PairCoordinateSpace(ctx)), ctx)
    stacked = PairCoordinateSpace(ctx).stacked(2)

    for space in (product, stacked):
        for name in (
            "inner",
            "riesz",
            "riesz_inverse",
            "norm",
            "is_euclidean",
            "star",
            "jordan",
            "spectrum",
            "spectral_decompose",
            "from_spectrum",
            "spectral_apply",
            "apply",
        ):
            assert not hasattr(space, name), (type(space).__name__, name)


def test_product_capability_is_component_intersection():
    ctx = sc.Context(sc.NumpyOps(), dtype=np.float64)
    product = sc.ProductSpace((InnerStarCoordinateSpace(ctx), InnerCoordinateSpace(ctx)), ctx)

    assert isinstance(product, sc.InnerProductSpace)
    assert not isinstance(product, sc.StarSpace)
    assert not isinstance(product, sc.JordanAlgebraSpace)
    assert hasattr(product, "inner")
    assert not hasattr(product, "star")


def test_constructor_validation_is_early_and_clear():
    ctx = sc.Context(sc.NumpyOps(), dtype=np.float64)

    with pytest.raises(TypeError, match="component 0 is FiniteSetSpace"):
        sc.ProductSpace((FiniteSetSpace({"a"}, ctx),), ctx)
    with pytest.raises(TypeError, match="component 0 is DenseCoordinateSpace"):
        sc.ProductStarSpace((sc.DenseCoordinateSpace((1,), ctx),), ctx)
    with pytest.raises(TypeError, match="base is FiniteSetSpace"):
        sc.StackedSpace(FiniteSetSpace({"a"}, ctx), 1, ctx)
    with pytest.raises(TypeError, match="base is DenseCoordinateSpace"):
        sc.StackedStarSpace(sc.DenseCoordinateSpace((1,), ctx), 1, ctx)
    with pytest.raises(ValueError, match="nonnegative"):
        sc.StackedSpace(PairCoordinateSpace(ctx), -1, ctx)


def test_no_space_apply_method_or_old_module_paths_remain():
    import importlib
    from pathlib import Path

    for cls in (sc.DenseCoordinateSpace, sc.ElementwiseJordanSpace, sc.HermitianSpace):
        assert "apply" not in cls.__dict__

    for module_name in (
        "spacecore.space._base",
        "spacecore.space._checks",
        "spacecore.space._herm",
        "spacecore.space._inner",
        "spacecore.space._product",
        "spacecore.space._stacked",
        "spacecore.space._vector",
    ):
        with pytest.raises(ModuleNotFoundError):
            importlib.import_module(module_name)

    root = Path(__file__).resolve().parents[2]
    offenders = []
    for file in (root / "spacecore" / "space").rglob("*.py"):
        text = file.read_text(encoding="utf-8")
        if "def apply(" in text:
            offenders.append(str(file.relative_to(root)))
    assert offenders == []
