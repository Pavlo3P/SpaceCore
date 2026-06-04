import importlib
import numpy as np
import pytest


def test_invalid_backend_raises():
    sc = importlib.import_module("spacecore")
    with pytest.raises(Exception):
        sc.Context("definitely_not_a_backend", dtype=np.float64)


def test_invalid_dtype_raises():
    sc = importlib.import_module("spacecore")
    with pytest.raises(Exception):
        sc.Context(sc.NumpyOps(), dtype="not_a_dtype")


def test_vector_space_check_member_wrong_shape_raises():
    sc = importlib.import_module("spacecore")
    X = sc.DenseCoordinateSpace((2,), sc.Context(sc.NumpyOps(), dtype=np.float32, enable_checks=True))
    with pytest.raises(Exception):
        X.check_member(np.asarray([1,2,3], dtype=np.float32))


def test_dense_linop_bad_shape_raises():
    sc = importlib.import_module("spacecore")
    ctx = sc.Context(sc.NumpyOps(), dtype=np.float64)
    with pytest.raises(Exception):
        sc.DenseLinOp(ctx.asarray([[1.,2.],[3.,4.]]), sc.DenseCoordinateSpace((2,),ctx), sc.DenseCoordinateSpace((3,),ctx), ctx)


def test_empty_product_linops_raise():
    sc = importlib.import_module("spacecore")
    with pytest.raises(Exception):
        sc.BlockDiagonalLinOp.from_operators(())
    with pytest.raises(Exception):
        sc.StackedLinOp.from_operators(())
    with pytest.raises(Exception):
        sc.SumToSingleLinOp.from_operators(())
