from __future__ import annotations

from typing import Any, Iterable

import numpy as np
import pytest

import spacecore as sc
from tests._helpers import to_numpy


def case_params(cases: Iterable[Any]) -> tuple[Any, ...]:
    return tuple(pytest.param(case, id=case.id) for case in cases)


def tolerances(dtype: Any) -> tuple[float, float]:
    real_dtype = np.empty((), dtype=np.dtype(dtype)).real.dtype
    return (2e-5, 2e-6) if real_dtype == np.dtype(np.float32) else (1e-10, 1e-11)


def assert_allclose(space: sc.CoordinateSpace, actual: Any, expected: Any) -> None:
    rtol, atol = tolerances(space.dtype)
    if not isinstance(space, sc.TreeSpace) and not space.ops.is_dense(expected):
        expected = space.ctx.asarray(expected)
    np.testing.assert_allclose(
        to_numpy(space.flatten(actual)),
        to_numpy(space.flatten(expected)),
        rtol=rtol,
        atol=atol,
    )


def assert_batch_allclose(space: sc.CoordinateSpace, actual: Any, expected: Any) -> None:
    rtol, atol = tolerances(space.dtype)
    np.testing.assert_allclose(
        to_numpy(space.flatten_batch(actual)),
        to_numpy(space.flatten_batch(expected)),
        rtol=rtol,
        atol=atol,
    )


def batch_item(space: sc.CoordinateSpace, values: Any, index: int) -> Any:
    if isinstance(space, sc.TreeSpace):
        return space.unflatten_tree(tuple(leaf[index] for leaf in space.flatten_tree(values)))
    value = values[index]
    return space.ctx.asarray(value) if space.shape == () else value


def convert_element(space: sc.CoordinateSpace, value: Any, target: sc.CoordinateSpace) -> Any:
    coordinates = to_numpy(space.flatten(value))
    return target.unflatten(target.ctx.asarray(coordinates))
