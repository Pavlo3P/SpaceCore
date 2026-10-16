from __future__ import annotations

from dataclasses import dataclass
from itertools import accumulate, pairwise
from math import prod
from typing import Any, Sequence

import optree

from ..._checks import minimum_check_level, normalize_check_level
from ..._checks import checked_method
from ...contextual import resolve_context_priority
from ...contextual import Context
from ...backend import CheckLevel
from ...backend import PyTreeNode
from ...types import DenseArray
from ..base import (
    CoordinateSpace,
    Space,
)
from ..checks import SpaceCheck, _run_checks
from ._dense_vector import (
    EuclideanElementwiseJordanSpace,
    _validate_euclidean_elementwise_jordan,
)

from .._capabilities import (
    CapabilitySet, registry_key,
    _space_capabilities,
)

_TREE_REGISTRY: dict[CapabilitySet, type[TreeSpace]] = {}


def _validate_leaf_spaces(
    leaf_spaces: Any, owner: str = "TreeSpace"
) -> tuple[CoordinateSpace, ...]:
    """Validate ordered finite-coordinate leaf spaces."""
    # A non-empty sequence of coordinate spaces.
    if not isinstance(leaf_spaces, Sequence):
        raise TypeError(
            f"{owner} requires a sequence of CoordinateSpace leaves; "
            f"got {type(leaf_spaces).__name__}."
        )
    spaces = tuple(leaf_spaces)
    if not spaces:
        raise ValueError(f"{owner} requires at least one leaf space.")
    for index, space in enumerate(spaces):
        if not isinstance(space, CoordinateSpace):
            raise TypeError(
                f"{owner} requires every leaf to be a CoordinateSpace; "
                f"leaf {index} is {type(space).__name__}."
            )
    return spaces


def _tree_capabilities(spaces: Sequence[Space]) -> CapabilitySet:
    """Return capabilities shared by every tree leaf."""
    # A capability is kept only if every leaf has it.
    for space in spaces:
        if isinstance(space, EuclideanElementwiseJordanSpace):
            _validate_euclidean_elementwise_jordan(space, space.geometry)
    return frozenset.intersection(*map(_space_capabilities, spaces))


def _format_path(path: tuple[Any, ...]) -> str:
    """Format an optree leaf path as a compact Python-style location."""
    # JSONPath-like string such as $.point[0], for error messages.
    result = "$"
    for entry in path:
        if isinstance(entry, str) and entry.isidentifier():
            result += f".{entry}"
        else:
            result += f"[{entry!r}]"
    return result


@dataclass(frozen=True, eq=False)
class TreeElement(PyTreeNode):
    r"""
    Leaves bound to a :class:`TreeSpace`; plain trees of the right structure work too.

    Leaves are not coerced; the leaf spaces check them.

    Parameters
    ----------
    space : TreeSpace
        The space fixing leaf order and structure.
    leaves : sequence
        Leaf values in ``space.leaf_paths`` order.

    Raises
    ------
    TypeError
        If ``space`` is not a TreeSpace.
    ValueError
        If the leaf count differs from ``space.arity``.

    Examples
    --------
    >>> import numpy as np
    >>> import spacecore as sc
    >>> ctx = sc.Context(sc.NumpyOps(), dtype=np.float64)
    >>> X = sc.DenseCoordinateSpace((1,), ctx)
    >>> T = sc.TreeSpace((0, 0), (X, X), ctx=ctx)
    >>> element = sc.TreeElement(T, (ctx.asarray([1.0]), ctx.asarray([2.0])))
    >>> element.value
    (array([1.]), array([2.]))
    """

    space: "TreeSpace"
    leaves: tuple[Any, ...]

    def __init__(self, space: "TreeSpace", leaves: Sequence[Any]) -> None:
        # Leaves must match the space's arity; the fields are frozen.
        if not isinstance(space, TreeSpace):
            raise TypeError(f"TreeElement space must be a TreeSpace, got {type(space).__name__}.")
        leaves_tuple = tuple(leaves)
        if len(leaves_tuple) != space.arity:
            raise ValueError(
                f"TreeElement expected {space.arity} leaves, got {len(leaves_tuple)}."
            )
        object.__setattr__(self, "space", space)
        object.__setattr__(self, "leaves", leaves_tuple)

    @property
    def value(self) -> Any:
        """Reconstruct the Python tree value represented by this element."""
        return self.space.unflatten_tree(self.leaves)

    def __repr__(self) -> str:
        """Summarize leaves rather than dumping their full array contents."""
        from ..._repr import summarize_value

        return f"TreeElement({self.space._short_repr()}, leaves={summarize_value(self.leaves)})"

    def tree_flatten(self):
        """Expose element leaves as JAX pytree children."""
        return self.leaves, self.space

    @classmethod
    def tree_unflatten(cls, space: "TreeSpace", children: Sequence[Any]) -> "TreeElement":
        """Rebuild an element from JAX pytree children."""
        return cls(space, tuple(children))


class _TreeStructureCheck(SpaceCheck):
    """Validate TreeElement ownership and leaf arity."""

    name = "tree_structure"
    minimum_level = "cheap"

    def is_valid(self, space: "TreeSpace", x: Any) -> bool:
        try:
            space.flatten_tree(x)
        except Exception:
            return False
        return True

    def error_message(self, space: "TreeSpace", x: Any) -> str:
        try:
            space.flatten_tree(x)
        except Exception as exc:
            return str(exc)
        return "Invalid TreeSpace structure."


class _TreeLeafCheck(SpaceCheck):
    """Validate each TreeElement leaf against its corresponding space."""

    name = "tree_leaves"
    minimum_level = "cheap"

    def is_valid(self, space: "TreeSpace", x: Any) -> bool:
        return self.validate(space, x, allow_leading=False)

    def validate(self, space: "TreeSpace", x: Any, *, allow_leading: bool) -> bool:
        # Every leaf must pass its own leaf space's checks.
        try:
            leaves = space._components(x)
        except Exception:
            return False
        for leaf_space, leaf in zip(space.leaf_spaces, leaves):
            try:
                _run_checks(leaf_space, leaf, allow_leading=allow_leading)
            except Exception:
                return False
        return True

    def error_message(self, space: "TreeSpace", x: Any) -> str:
        return self.validation_message(space, x, allow_leading=False)

    def validation_message(
        self, space: "TreeSpace", x: Any, *, allow_leading: bool
    ) -> str:
        # Report the first failing leaf by its path.
        try:
            leaves = space._components(x)
        except Exception as exc:
            return str(exc)
        for path, leaf_space, leaf in zip(space.leaf_paths, space.leaf_spaces, leaves):
            try:
                _run_checks(leaf_space, leaf, allow_leading=allow_leading)
            except Exception as exc:
                return (
                    f"Invalid leaf at {_format_path(path)} for "
                    f"{type(leaf_space).__name__}: {exc}"
                )
        return "Invalid TreeSpace leaf."


class TreeSpace(PyTreeNode, CoordinateSpace):
    r"""
    Finite direct product :math:`\prod_\ell X_\ell` whose elements are Python trees.

    Any tuple, list, dict, named-tuple or optree-registered structure works. Vector
    operations, conversion, checks and batching act leaf by leaf, and a capability
    (inner product, star, Jordan) is offered only when every leaf has it.

    Parameters
    ----------
    treedef : optree.PyTreeSpec or tree
        Structure, or an example tree; its leaf order pairs with ``leaf_spaces``.
    leaf_spaces : sequence of CoordinateSpace
        Leaf spaces, in leaf order.
    ctx : Context, str, or None, optional
        Backend context; resolved from ``leaf_spaces`` when omitted.
    check_level : {"none", "cheap", "standard", "strict"}, optional
        Validation level; the ambient default when omitted.

    Raises
    ------
    TypeError
        If ``leaf_spaces`` are not coordinate spaces.
    ValueError
        If there are no leaves or the leaf count does not match the tree.

    Examples
    --------
    >>> import numpy as np
    >>> import spacecore as sc
    >>> ctx = sc.Context(sc.NumpyOps(), dtype=np.float64)
    >>> X = sc.DenseCoordinateSpace((2,), ctx)
    >>> S = sc.DenseCoordinateSpace((1,), ctx)
    >>> T = sc.TreeSpace({"point": 0, "weight": 0}, (X, S), ctx=ctx)
    >>> x = {"point": ctx.asarray([1.0, 2.0]), "weight": ctx.asarray([3.0])}
    >>> T.flatten(x)
    array([1., 2., 3.])
    >>> T.scale(2.0, x)["point"]
    array([2., 4.])
    """

    _fields = ("treedef", "leaf_spaces", "ctx")

    def __new__(
        cls,
        treedef: optree.PyTreeSpec | Any,
        leaf_spaces: Sequence[CoordinateSpace],
        *,
        ctx: Context | str | None = None,
        check_level: CheckLevel | str | None = None,
    ):
        # Pick the subclass matching the capabilities all leaves share.
        if cls is TreeSpace:
            spaces = _validate_leaf_spaces(leaf_spaces)
            resolved_ctx = resolve_context_priority(ctx, *spaces)
            converted = tuple(space.convert(resolved_ctx) for space in spaces)
            cls = _TREE_REGISTRY.get(registry_key(_tree_capabilities(converted)), TreeSpace)
        return super(TreeSpace, cls).__new__(cls)

    def __init__(
        self,
        treedef: optree.PyTreeSpec | Any,
        leaf_spaces: Sequence[CoordinateSpace],
        *,
        ctx: Context | str | None = None,
        check_level: CheckLevel | str | None = None,
    ) -> None:
        # The structure must have exactly one leaf per leaf space.
        spaces = _validate_leaf_spaces(leaf_spaces, type(self).__name__)
        resolved_ctx = resolve_context_priority(ctx, *spaces)
        treespec = treedef if isinstance(treedef, optree.PyTreeSpec) else optree.tree_structure(treedef)
        if treespec.num_leaves != len(spaces):
            raise ValueError(
                "TreeSpace leaf-count mismatch: "
                f"treedef has {treespec.num_leaves} leaves but {len(spaces)} leaf spaces were given."
            )

        # Convert leaves to one context; record flat sizes and slices into the coordinate vector.
        uniform_spaces = tuple(space.convert(resolved_ctx) for space in spaces)
        self._treedef = treespec
        self._leaf_paths = tuple(optree.treespec_paths(treespec))
        self.leaf_spaces = uniform_spaces
        offsets = (0, *accumulate(prod(map(int, s.shape)) for s in uniform_spaces))
        self._slices = tuple(slice(a, b) for a, b in pairwise(offsets))
        # Coordinate shape is the total size; check level: explicit, else the least strict leaf.
        super(TreeSpace, self).__init__((offsets[-1],), resolved_ctx)
        if check_level is not None:
            self._check_level = normalize_check_level(check_level)
        else:
            leaf_levels = tuple(space.check_level for space in uniform_spaces)
            self._check_level = minimum_check_level(leaf_levels)

    @classmethod
    def from_leaf_spaces(
        cls,
        leaf_spaces: Sequence[CoordinateSpace],
        ctx: Context | str | None = None,
        *,
        check_level: CheckLevel | str | None = None,
    ) -> "TreeSpace":
        """Build a tuple-structured ``TreeSpace`` from ordered leaf spaces."""
        spaces = _validate_leaf_spaces(leaf_spaces)
        return cls(tuple(range(len(spaces))), spaces, ctx=ctx, check_level=check_level)

    @classmethod
    def from_template(
        cls,
        template: Any,
        leaf_spaces: Sequence[CoordinateSpace],
        *,
        ctx: Context | str | None = None,
        check_level: CheckLevel | str | None = None,
    ) -> "TreeSpace":
        """Build a tree space from an example Python tree value."""
        return cls(template, leaf_spaces, ctx=ctx, check_level=check_level)

    def _eq_algebra(self, other: Any) -> bool:
        # Tier 2: tree structure (treedef) + ordered leaf spaces.
        return (
            super()._eq_algebra(other)
            and self.treedef == other.treedef
            and self.leaf_spaces == other.leaf_spaces
        )

    @property
    def treedef(self) -> optree.PyTreeSpec:
        """Return the immutable optree structure definition."""
        return self._treedef

    @property
    def leaf_paths(self) -> tuple[tuple[Any, ...], ...]:
        """Return paths in the same deterministic order as ``leaf_spaces``."""
        return self._leaf_paths

    @property
    def arity(self) -> int:
        """Return the number of ordered leaves."""
        return len(self.leaf_spaces)

    def _repr_class_name(self) -> str:
        """Present the public ``TreeSpace`` label, not the private dispatch subclass."""
        return "TreeSpace"

    def _space_descriptor(self) -> str:
        """Return ``Tree(<leaf descriptors>)``, abbreviating wide trees."""
        from ..._repr import describe_space

        leaves = self.leaf_spaces
        shown = [describe_space(leaf) for leaf in leaves[:4]]
        if len(leaves) > 4:
            shown.append(f"…(+{len(leaves) - 4})")
        return f"Tree({', '.join(shown)})"

    def _local_checks(self):
        return _TreeStructureCheck("tree_structure"), _TreeLeafCheck("tree_leaves")

    def flatten_tree(self, value: Any) -> tuple[Any, ...]:
        """Flatten a matching Python tree value into ordered leaves."""
        # A bound TreeElement already holds ordered leaves; a plain tree must match the structure.
        if isinstance(value, TreeElement):
            if value.space != self:
                raise TypeError("TreeElement is bound to a different TreeSpace.")
            return value.leaves
        leaves, treespec = optree.tree_flatten(value)
        if treespec != self.treedef:
            raise TypeError(
                f"TreeSpace structure mismatch: expected {self.treedef}, got {treespec}."
            )
        if len(leaves) != self.arity:
            raise ValueError(f"TreeSpace expected {self.arity} leaves, got {len(leaves)}.")
        return tuple(leaves)

    def unflatten_tree(self, leaves: Sequence[Any]) -> Any:
        """Rebuild the configured Python tree from ordered leaves."""
        leaves_tuple = tuple(leaves)
        if len(leaves_tuple) != self.arity:
            raise ValueError(f"TreeSpace expected {self.arity} leaves, got {len(leaves_tuple)}.")
        return optree.tree_unflatten(self.treedef, leaves_tuple)

    def element(self, value: Any) -> TreeElement:
        """Bind a matching Python tree value to this space."""
        return TreeElement(self, self.flatten_tree(value))

    def _components(self, x: TreeElement) -> tuple[Any, ...]:
        return self.flatten_tree(x)

    def _from_components(self, parts: tuple[Any, ...]) -> Any:
        return self.unflatten_tree(parts)

    def zero(self) -> Any:
        """Return the additive identity."""
        return self.zeros()

    def zeros(self) -> Any:
        """Return the leafwise additive identity."""
        return self._from_components(tuple(space.zeros() for space in self.leaf_spaces))

    def ones(self) -> Any:
        """Return a leafwise all-ones element when leaf spaces support it."""
        # Each leaf's own ones if it has them, else an array of ones.
        parts = []
        for space in self.leaf_spaces:
            ones = getattr(space, "ones", None)
            parts.append(ones() if callable(ones) else self.ops.ones(space.shape, dtype=self.dtype))
        return self._from_components(tuple(parts))

    def _add_core(self, x: Any, y: Any) -> Any:
        """Return the componentwise sum."""
        # Leafwise: each leaf space does its own operation.
        return self._from_components(
            tuple(
                space._add_core(xi, yi)
                for space, xi, yi in zip(
                    self.leaf_spaces, self._components(x), self._components(y)
                )
            )
        )

    def add_batch(self, x: Any, y: Any) -> Any:
        """Return the componentwise leading-axis batch sum."""
        return self._from_components(
            tuple(
                space.add_batch(xi, yi)
                for space, xi, yi in zip(
                    self.leaf_spaces, self._components(x), self._components(y)
                )
            )
        )

    def _scale_core(self, a: Any, x: Any) -> Any:
        """Return the componentwise scalar product."""
        return self._from_components(
            tuple(
                space._scale_core(a, leaf)
                for space, leaf in zip(self.leaf_spaces, self._components(x))
            )
        )

    def scale_batch(self, a: Any, x: Any) -> Any:
        """Return the componentwise batched scalar product."""
        return self._from_components(
            tuple(
                space.scale_batch(a, leaf)
                for space, leaf in zip(self.leaf_spaces, self._components(x))
            )
        )

    def stacked(self, count: int) -> "TreeSpace":
        """Return a tree whose leaves are stacked leaf spaces."""
        return TreeSpace(
            self.treedef,
            tuple(space.stacked(count) for space in self.leaf_spaces),
            ctx=self.ctx,
        )

    @checked_method(in_space="self")
    def flatten(self, x: Any) -> DenseArray:
        """Concatenate leaf coordinate vectors into one dense vector."""
        # Concatenate the leaves' flat coordinates in leaf order.
        parts = []
        for space, leaf in zip(self.leaf_spaces, self._components(x)):
            coordinates = space.flatten(leaf)
            if self._checks_at_least("cheap"):
                coordinates = self.ctx.assert_dense(coordinates)
            parts.append(coordinates)
        if len(parts) == 1:
            return parts[0]
        return self.ops.concatenate(parts, axis=0)

    def unflatten(self, v: DenseArray) -> Any:
        """Split dense coordinates into a tree element."""
        # Slice the flat vector back into leaves and rebuild the tree.
        if self._checks_at_least("cheap"):
            v = self.ctx.assert_dense(v)
            v = (
                v
                if tuple(getattr(v, "shape", ())) == self.shape
                else v.reshape((-1,))
            )
        leaves = tuple(
            space.unflatten(v[leaf_slice])
            for space, leaf_slice in zip(self.leaf_spaces, self._slices)
        )
        return self._from_components(leaves)

    def flatten_batch(self, xs: Any) -> DenseArray:
        """Concatenate batched leaf coordinates into shape ``(N, size)``."""
        parts = tuple(
            space.flatten_batch(leaf)
            for space, leaf in zip(self.leaf_spaces, self._components(xs))
        )
        if len(parts) == 1:
            return parts[0]
        return self.ops.concatenate(parts, axis=1)

    def unflatten_batch(self, vs: DenseArray) -> Any:
        """Split batched dense coordinates into batched leaves."""
        if self._checks_at_least("cheap"):
            vs = self.ctx.assert_dense(vs)
        return self._from_components(
            tuple(
                space.unflatten_batch(vs[:, leaf_slice])
                for space, leaf_slice in zip(self.leaf_spaces, self._slices)
            )
        )

    def check(self, x: Any) -> None:
        """Validate an element according to this space's check level."""
        self.check_member(x)

    def convert_element(
        self, x: Any, new_ctx: Context | str | None = None
    ) -> Any:
        """Convert each leaf and preserve the configured tree structure."""
        # Move each leaf through its flat coordinates onto the target context.
        target = self.convert(new_ctx)
        converted = []
        for source_space, target_space, leaf in zip(
            self.leaf_spaces, target.leaf_spaces, self._components(x)
        ):
            coordinates = source_space.flatten(leaf)
            converted.append(target_space.unflatten(target.ctx.asarray(coordinates)))
        return target._from_components(tuple(converted))


# Imported last: the capability subclasses derive from TreeSpace and fill _TREE_REGISTRY.
from ._tree_capabilities import TreeSpectralDecomposition  # noqa: E402


__all__ = [
    "TreeElement",
    "TreeSpace",
    "TreeSpectralDecomposition",
    "_space_capabilities",
    "_tree_capabilities",
]
