from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pytest

import spacecore as sc


def _ctx(dtype=np.float64, enable_checks=True):
    return sc.Context(sc.NumpyOps(), dtype=dtype, enable_checks=enable_checks)


def test_backend_check_rejects_non_backend_dense_array():
    space = sc.DenseCoordinateSpace((2,), _ctx())

    with pytest.raises((ValueError, TypeError), match="Expected dense array for numpy"):
        sc.BackendCheck()(space, [1.0, 2.0])


def test_shape_check_rejects_wrong_shape():
    ctx = _ctx()
    space = sc.DenseCoordinateSpace((2,), ctx)

    with pytest.raises(ValueError, match=r"Expected shape \(2,\), got \(3,\)"):
        sc.ShapeCheck()(space, ctx.asarray([1.0, 2.0, 3.0]))


def test_dtype_check_rejects_wrong_dtype():
    ctx = _ctx(np.float32)
    space = sc.DenseCoordinateSpace((2,), ctx)

    with pytest.raises(ValueError, match="Expected dtype float32, got float64"):
        sc.DTypeCheck()(space, np.asarray([1.0, 2.0], dtype=np.float64))


def test_square_matrix_check_rejects_non_square_matrix():
    ctx = _ctx()
    space = sc.HermitianSpace(2, ctx=ctx)

    with pytest.raises(ValueError, match=r"Expected square matrix, got shape \(2, 3\)"):
        sc.SquareMatrixCheck()(space, ctx.asarray([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]))


def test_hermitian_check_rejects_non_hermitian_matrix():
    ctx = _ctx()
    space = sc.HermitianSpace(2, ctx=ctx)

    with pytest.raises(ValueError, match="not Hermitian"):
        sc.HermitianCheck()(space, ctx.asarray([[1.0, 2.0], [0.0, 1.0]]))


def test_hermitian_check_uses_space_tolerances_and_enforce_flag():
    ctx = _ctx()
    almost = ctx.asarray([[1.0, 1.0], [1.0 + 1e-5, 1.0]])

    space = sc.HermitianSpace(2, ctx=ctx)

    assert not sc.HermitianCheck(atol=0.0, rtol=0.0).is_valid(space, almost)
    assert sc.HermitianCheck(atol=1e-4, rtol=0.0).is_valid(space, almost)
    assert sc.HermitianCheck(enforce=False).is_valid(space, ctx.asarray([[1.0, 2.0], [0.0, 1.0]]))


def test_hermitian_space_uses_configured_check_parameters():
    ctx = _ctx()
    almost = ctx.asarray([[1.0, 1.0], [1.0 + 1e-5, 1.0]])
    loose = sc.HermitianSpace(2, atol=1e-4, rtol=0.0, ctx=ctx)
    disabled = sc.HermitianSpace(2, enforce_herm=False, ctx=ctx)
    loose_check = next(
        check for check in loose.member_checks() if isinstance(check, sc.HermitianCheck)
    )
    disabled_check = next(
        check for check in disabled.member_checks() if isinstance(check, sc.HermitianCheck)
    )

    assert loose_check.atol == 1e-4
    assert loose_check.rtol == 0.0
    assert loose_check.enforce is True
    assert disabled_check.enforce is False
    loose.check_member(almost)
    disabled.check_member(ctx.asarray([[1.0, 2.0], [0.0, 1.0]]))


def test_product_structure_check_rejects_non_tuple_and_wrong_arity():
    ctx = _ctx()
    product = sc.ProductSpace(
        (sc.DenseCoordinateSpace((2,), ctx), sc.DenseCoordinateSpace((3,), ctx)), ctx
    )

    with pytest.raises(ValueError, match="ProductSpace element must be a tuple"):
        sc.ProductStructureCheck()(product, [ctx.asarray([1.0, 2.0]), ctx.asarray([3.0, 4.0, 5.0])])

    with pytest.raises(ValueError, match="Expected tuple of length 2, got 1"):
        sc.ProductStructureCheck()(product, (ctx.asarray([1.0, 2.0]),))


def test_product_component_check_rejects_invalid_component():
    ctx = _ctx()
    product = sc.ProductSpace(
        (sc.DenseCoordinateSpace((2,), ctx), sc.DenseCoordinateSpace((3,), ctx)), ctx
    )

    with pytest.raises(ValueError, match=r"Invalid component 1.*Expected shape \(3,\)"):
        sc.ProductComponentCheck()(product, (ctx.asarray([1.0, 2.0]), ctx.asarray([3.0, 4.0])))


@dataclass(frozen=True)
class RejectFirstEntryCheck(sc.SpaceCheck):
    value: float = 0.0

    def is_valid(self, space: Any, x: Any) -> bool:
        return bool(x[0] != self.value)

    def error_message(self, space: Any, x: Any) -> str:
        return f"First entry must not be {self.value}."


def test_subclass_checks_extend_parent_checks():
    class ParentVectorSpace(sc.DenseCoordinateSpace):
        checks = (RejectFirstEntryCheck("parent_reject", 1.0),)

    class ChildVectorSpace(ParentVectorSpace):
        checks = (RejectFirstEntryCheck("child_reject", 2.0),)

    ctx = _ctx()
    space = ChildVectorSpace((1,), ctx)
    names = [check.name for check in space.member_checks()]
    assert names == ["backend", "shape", "dtype", "parent_reject", "child_reject"]

    with pytest.raises(ValueError, match="1.0"):
        space.check_member(ctx.asarray([1.0]))

    with pytest.raises(ValueError, match="2.0"):
        space.check_member(ctx.asarray([2.0]))


def test_disabled_context_skips_inherited_checks():
    class ChildVectorSpace(sc.DenseCoordinateSpace):
        checks = (RejectFirstEntryCheck("child_reject", 0.0),)

    ctx = _ctx(enable_checks=False)
    space = ChildVectorSpace((1,), ctx)

    space.check_member(ctx.asarray([0.0]))


def test_instance_specific_local_checks_extend_parent_checks():
    class ParameterizedVectorSpace(sc.DenseCoordinateSpace):
        def __init__(self, shape, reject_value, ctx=None):
            super().__init__(shape, ctx)
            self.reject_value = reject_value

        def _local_checks(self):
            return (RejectFirstEntryCheck("instance_reject", self.reject_value),)

    ctx = _ctx()
    space = ParameterizedVectorSpace((1,), 3.0, ctx)
    names = [check.name for check in space.member_checks()]

    assert names == ["backend", "shape", "dtype", "instance_reject"]
    with pytest.raises(ValueError, match="3.0"):
        space.check_member(ctx.asarray([3.0]))
