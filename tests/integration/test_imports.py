import importlib


def test_import_spacecore():
    mod = importlib.import_module("spacecore")
    assert mod is not None


def test_top_level_exports_exist():
    sc = importlib.import_module("spacecore")
    for name in [
        "Context", "BackendOps", "NumpyOps", "DenseLinOp",
        "VectorSpace", "HermitianSpace", "ProductSpace", "StackedSpace",
        "set_context", "get_context", "register_ops",
    ]:
        assert hasattr(sc, name)


def test_subpackages_import():
    for name in [
        "spacecore.backend",
        "spacecore.space",
        "spacecore.linop",
        "spacecore._contextual",
    ]:
        assert importlib.import_module(name) is not None
