from __future__ import annotations

import pytest

from tests.generators import vector_space_law_cases
from tests.spaces._generated_helpers import assert_allclose, case_params


CASES = vector_space_law_cases()


@pytest.mark.parametrize("case", case_params(CASES))
def test_generated_vector_space_laws(case):
    space = case.obj
    ref = case.reference
    x, y, z = ref["x"], ref["y"], ref["z"]
    a, b = ref["a"], ref["b"]
    zero = space.zeros()

    assert_allclose(space, space.add(x, zero), x)
    assert_allclose(space, space.add(x, y), space.add(y, x))
    assert_allclose(space, space.add(space.add(x, y), z), space.add(x, space.add(y, z)))
    assert_allclose(space, space.scale(1, x), x)
    assert_allclose(
        space,
        space.scale(a, space.add(x, y)),
        space.add(space.scale(a, x), space.scale(a, y)),
    )
    assert_allclose(
        space,
        space.scale(a + b, x),
        space.add(space.scale(a, x), space.scale(b, x)),
    )
    assert_allclose(space, space.scale(a * b, x), space.scale(a, space.scale(b, x)))
