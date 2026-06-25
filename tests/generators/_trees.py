from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Literal

import numpy as np

from spacecore.backend import Context
from spacecore.space import DenseCoordinateSpace, TreeSpace

from ._arrays import dense_array_case
from ._protocol import GeneratedCase
from ._seed import resolve_rng


TreeKind = Literal["tuple", "nested", "dict"]


def _tree_layout(kind: TreeKind) -> tuple[Any, tuple[tuple[int, ...], ...]]:
    if kind == "tuple":
        return (0, 0), ((2,), (3,))
    if kind == "nested":
        return (0, [0, (0,)]), ((2,), (1,), (2, 2))
    if kind == "dict":
        return {"point": 0, "weight": 0}, ((2,), (1,))
    raise ValueError(f"Unknown tree kind {kind!r}. Expected 'tuple', 'nested', or 'dict'.")


def _mismatch(kind: TreeKind, leaves: tuple[Any, ...]) -> Any:
    if kind == "tuple":
        return list(leaves)
    if kind == "nested":
        return tuple(leaves)
    return tuple(leaves)


def tree_space_case(
    ctx: Context,
    kind: TreeKind = "tuple",
    *,
    leaf_shapes: Sequence[Sequence[int]] | None = None,
    batch_shape: Sequence[int] = (),
    seed: int | None = None,
    rng: np.random.Generator | None = None,
) -> GeneratedCase[TreeSpace]:
    """Generate a TreeSpace, matching values, paths, and a mismatch example."""
    template, default_shapes = _tree_layout(kind)
    shapes = default_shapes if leaf_shapes is None else tuple(tuple(shape) for shape in leaf_shapes)
    generator = resolve_rng(seed=seed, rng=rng)
    leaf_spaces = tuple(DenseCoordinateSpace(tuple(shape), ctx) for shape in shapes)
    space = TreeSpace.from_template(template, leaf_spaces, ctx=ctx)
    if len(shapes) != space.arity:
        raise ValueError(
            f"{kind} tree requires {space.arity} leaf shapes, got {len(shapes)}."
        )
    leaf_cases = tuple(
        dense_array_case(
            ctx,
            shape,
            batch_shape=batch_shape,
            seed=None,
            rng=generator,
        )
        for shape in shapes
    )
    leaves = tuple(case.obj for case in leaf_cases)
    element = space.unflatten_tree(leaves)
    prefix = tuple(int(dimension) for dimension in batch_shape)
    capabilities = {"tree", kind, "structured"}
    if prefix:
        capabilities.add("batched")
    return GeneratedCase(
        obj=space,
        reference={
            "element": element,
            "tree_element": space.element(element),
            "leaves": leaves,
            "flattened_leaves": tuple(case.reference["array"] for case in leaf_cases),
            "leaf_paths": space.leaf_paths,
            "leaf_spaces": space.leaf_spaces,
            "mismatch": _mismatch(kind, leaves),
            "batch_shape": prefix,
        },
        capabilities=frozenset(capabilities),
        id=f"tree-{kind}" + (f"-batch-{'x'.join(map(str, prefix))}" if prefix else ""),
    )


def tree_space_cases(
    ctx: Context,
    *,
    kinds: Sequence[TreeKind] = ("tuple", "nested", "dict"),
    batch_shape: Sequence[int] = (),
    seed: int | None = None,
    rng: np.random.Generator | None = None,
) -> tuple[GeneratedCase[TreeSpace], ...]:
    """Generate the standard tuple, nested, and dictionary TreeSpace cases."""
    generator = resolve_rng(seed=seed, rng=rng)
    return tuple(
        tree_space_case(
            ctx,
            kind,
            batch_shape=batch_shape,
            seed=None,
            rng=generator,
        )
        for kind in kinds
    )
