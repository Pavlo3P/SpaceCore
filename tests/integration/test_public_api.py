import importlib
import tomllib
from pathlib import Path

from tests._helpers import has_cupy, has_jax, has_torch


ROOT = Path(__file__).resolve().parents[2]


def test___all___contains_importable_names():
    sc = importlib.import_module("spacecore")
    assert isinstance(sc.__all__, list)
    for name in sc.__all__:
        assert hasattr(sc, name)


def test_expected_names_are_exported():
    sc = importlib.import_module("spacecore")
    expected = {
        "Context",
        "BackendOps",
        "NumpyOps",
        "DenseLinOp",
        "SparseLinOp",
        "ScaledLinOp",
        "SumLinOp",
        "ComposedLinOp",
        "ZeroLinOp",
        "IdentityLinOp",
        "MatrixFreeLinOp",
        "make_sum",
        "make_scaled",
        "make_composed",
        "ProductLinOp",
        "BlockDiagonalLinOp",
        "StackedLinOp",
        "SumToSingleLinOp",
        "Functional",
        "LinearFunctional",
        "InnerProductFunctional",
        "ComposedFunctional",
        "MatrixFreeLinearFunctional",
        "QuadraticForm",
        "LinOpQuadraticForm",
        "make_functional_composed",
        "DenseCoordinateSpace",
        "HermitianSpace",
        "ProductSpace",
        "ProductStructure",
        "TupleStructure",
        "PytreeStructure",
        "StackedSpace",
        "Space",
        "InnerProduct",
        "EuclideanInnerProduct",
        "WeightedInnerProduct",
        "DenseArray",
        "SparseArray",
        "ArrayLike",
        "set_context",
        "get_context",
        "resolve_context_priority",
        "register_ops",
        "LanczosResult",
        "lanczos_smallest",
        "ExpmMultiplyResult",
        "expm_multiply",
    }
    if has_jax():
        expected |= {"JaxOps", "jax_pytree_class"}
    if has_cupy():
        expected |= {"CuPyOps"}
    if has_torch():
        expected |= {"TorchOps"}
    assert expected.issubset(set(sc.__all__))


def test_top_level_objects_match_source_modules():
    sc = importlib.import_module("spacecore")
    backend = importlib.import_module("spacecore.backend")
    space = importlib.import_module("spacecore.space")
    linop = importlib.import_module("spacecore.linop")
    functional = importlib.import_module("spacecore.functional")
    linalg = importlib.import_module("spacecore.linalg")
    contextual = importlib.import_module("spacecore._contextual")

    assert sc.Context is backend.Context
    assert sc.NumpyOps is backend.NumpyOps
    if has_cupy():
        assert sc.CuPyOps is backend.CuPyOps
    if has_torch():
        assert sc.TorchOps is backend.TorchOps
    assert sc.Space is space.Space
    assert sc.InnerProduct is space.InnerProduct
    assert sc.EuclideanInnerProduct is space.EuclideanInnerProduct
    assert sc.WeightedInnerProduct is space.WeightedInnerProduct
    assert sc.DenseCoordinateSpace is space.DenseCoordinateSpace
    assert sc.ProductStructure is space.ProductStructure
    assert sc.TupleStructure is space.TupleStructure
    assert sc.PytreeStructure is space.PytreeStructure
    assert sc.StackedSpace is space.StackedSpace
    assert sc.ProductLinOp is linop.ProductLinOp
    assert sc.DenseLinOp is linop.DenseLinOp
    assert sc.Functional is functional.Functional
    assert sc.ComposedFunctional is functional.ComposedFunctional
    assert sc.InnerProductFunctional is functional.InnerProductFunctional
    assert sc.LanczosResult is linalg.LanczosResult
    assert sc.ExpmMultiplyResult is linalg.ExpmMultiplyResult
    assert sc.expm_multiply is linalg.expm_multiply
    assert sc.get_context is contextual.get_context
    assert sc.resolve_context_priority is contextual.resolve_context_priority


def test_package_version_matches_project_metadata():
    sc = importlib.import_module("spacecore")
    metadata = tomllib.loads((ROOT / "pyproject.toml").read_text())
    assert metadata["project"]["dynamic"] == ["version"]
    assert metadata["tool"]["setuptools"]["dynamic"]["version"]["attr"] == (
        "spacecore._version.__version__"
    )
    assert sc.__version__ == "0.3.1"
