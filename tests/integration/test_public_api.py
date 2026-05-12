import importlib
from tests._helpers import has_jax


def test___all___contains_importable_names():
    sc = importlib.import_module("spacecore")
    assert isinstance(sc.__all__, list)
    for name in sc.__all__:
        assert hasattr(sc, name)


def test_expected_names_are_exported():
    sc = importlib.import_module("spacecore")
    expected = {
        "Context", "BackendOps", "NumpyOps", "DenseLinOp", "SparseLinOp",
        "BlockDiagonalLinOp", "StackedLinOp", "SumToSingleLinOp",
        "VectorSpace", "HermitianSpace", "ProductSpace", "Space",
        "DenseArray", "SparseArray", "ArrayLike",
        "set_context", "get_context", "register_ops",
        "set_resolution_policy", "set_dtype_resolution_policy",
        "get_resolution_policy", "get_dtype_resolution_policy",
    }
    if has_jax():
        expected |= {"JaxOps", "jax_pytree_class"}
    assert expected.issubset(set(sc.__all__))


def test_top_level_objects_match_source_modules():
    sc = importlib.import_module("spacecore")
    backend = importlib.import_module("spacecore.backend")
    space = importlib.import_module("spacecore.space")
    linop = importlib.import_module("spacecore.linop")
    manager = importlib.import_module("spacecore._contextual.manager")

    assert sc.Context is backend.Context
    assert sc.NumpyOps is backend.NumpyOps
    assert sc.Space is space.Space
    assert sc.VectorSpace is space.VectorSpace
    assert sc.DenseLinOp is linop.DenseLinOp
    assert sc.get_context is manager.get_context
