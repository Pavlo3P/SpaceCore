from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Sequence, cast

import optree

from ..._check_policy import normalize_check_level
from ..._checks import checked_method
from ..._contextual import resolve_context_priority
from ...backend import BackendOps, CheckLevel, Context, jax_pytree_class
from ...types import DenseArray
from ..base import (
    CoordinateSpace,
    EuclideanJordanAlgebraSpace,
    InnerProductSpace,
    JordanAlgebraSpace,
    Space,
    StarSpace,
)
from ..checks import SpaceCheck, _run_checks
from ._dense_vector import (
    EuclideanElementwiseJordanSpace,
    _validate_euclidean_elementwise_jordan,
)

CapabilitySet = frozenset[type]
_CAP_INNER = InnerProductSpace
_CAP_STAR = StarSpace
_CAP_JORDAN = JordanAlgebraSpace
_CAP_EUCLIDEAN_JORDAN = EuclideanJordanAlgebraSpace
_TREE_REGISTRY: dict[CapabilitySet, type[TreeSpace]] = {}


def _prod_int(shape: tuple[int, ...]) -> int:
    """Return the integer product of a shape tuple."""
    result = 1
    for dimension in shape:
        result *= int(dimension)
    return int(result)


def _validate_leaf_spaces(
    leaf_spaces: Any, owner: str = "TreeSpace"
) -> tuple[CoordinateSpace, ...]:
    """Validate ordered finite-coordinate leaf spaces."""
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


def _space_capabilities(space: Space) -> CapabilitySet:
    """Return structural capabilities advertised by one leaf space."""
    capabilities: set[type] = set()
    if isinstance(space, InnerProductSpace):
        capabilities.add(_CAP_INNER)
    if isinstance(space, StarSpace):
        capabilities.add(_CAP_STAR)
    if isinstance(space, JordanAlgebraSpace):
        capabilities.add(_CAP_JORDAN)
    if isinstance(space, EuclideanJordanAlgebraSpace):
        if isinstance(space, EuclideanElementwiseJordanSpace):
            _validate_euclidean_elementwise_jordan(space, space.geometry)
        capabilities.add(_CAP_EUCLIDEAN_JORDAN)
    return frozenset(capabilities)


def _tree_capabilities(spaces: Sequence[Space]) -> CapabilitySet:
    """Return capabilities shared by every tree leaf."""
    if not spaces:
        return frozenset()
    shared = set(_space_capabilities(spaces[0]))
    for space in spaces[1:]:
        shared.intersection_update(_space_capabilities(space))
    return frozenset(shared)


def _format_path(path: tuple[Any, ...]) -> str:
    """Format an optree leaf path as a compact Python-style location."""
    result = "$"
    for entry in path:
        if isinstance(entry, str) and entry.isidentifier():
            result += f".{entry}"
        else:
            result += f"[{entry!r}]"
    return result


def _context_with_check_level(ctx: Context, check_level: CheckLevel | str | None) -> Context:
    """Return ``ctx`` with an explicit validation policy when requested."""
    if check_level is None or ctx.check_level == check_level:
        return ctx
    return Context(ctx.ops, dtype=ctx.dtype, check_level=normalize_check_level(check_level))


@jax_pytree_class
@dataclass(frozen=True, eq=False)
class TreeElement:
    r"""
    Bind ordered leaves to a :class:`TreeSpace`.

    ``TreeElement`` is an explicit binding helper. Ordinary tuple, list,
    dictionary, named-tuple, and registered optree values with the configured
    structure are also valid TreeSpace elements.

    Parameters
    ----------
    space : TreeSpace
        Finite direct-product space that defines leaf order and structure.
    leaves : sequence
        Leaf values in ``space.leaf_paths`` order.

    Attributes
    ----------
    space : TreeSpace
        Bound tree space.
    leaves : tuple
        Ordered leaf values.

    Raises
    ------
    TypeError
        If ``space`` is not a TreeSpace.
    ValueError
        If the leaf count does not match ``space.arity``.

    Notes
    -----
    The wrapper does not coerce leaf backend, dtype, field, or shape. Those
    contracts are inherited from the leaf spaces and enforced by
    :meth:`TreeSpace.check` according to ``space.check_level``.

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


@jax_pytree_class
class TreeSpace(CoordinateSpace):
    r"""
    Represent a finite direct product as a Python tree.

    ``TreeSpace`` represents
    :math:`X = \prod_{\ell \in L} X_\ell`, where each leaf space
    :math:`X_\ell` is a finite-coordinate SpaceCore space. The optree
    definition records the Python organization of an element; it does not
    define a tensor product. Tuple, list, dictionary, named-tuple, and
    registered optree structures are supported.

    Parameters
    ----------
    treedef : optree.PyTreeSpec or tree
        Immutable structure definition or example tree. Its deterministic leaf
        order is paired with ``leaf_spaces``.
    leaf_spaces : sequence of CoordinateSpace
        Nonempty ordered spaces for the tree leaves.
    ctx : Context, str, or None, optional
        Backend context. If omitted, it is resolved from ``leaf_spaces``.
    check_level : {"none", "cheap", "standard", "strict"}, optional
        Validation policy override. Leaf backend, dtype, field, and shape checks
        retain their normal minimum levels.

    Attributes
    ----------
    treedef : optree.PyTreeSpec
        Tree structure independent of element values.
    leaf_spaces : tuple of CoordinateSpace
        Ordered leaf spaces converted to ``ctx``.
    leaf_paths : tuple of tuple
        Deterministic paths corresponding to ``leaf_spaces``.
    shape : tuple of int
        Dense coordinate shape ``(sum(leaf.size),)``.

    Raises
    ------
    TypeError
        If ``leaf_spaces`` is not a sequence of coordinate spaces.
    ValueError
        If there are no leaf spaces or the tree leaf count does not match.

    See Also
    --------
    TreeElement
        Optional explicit binding of leaves to a TreeSpace.

    Notes
    -----
    Vector operations, conversion, validation, and batching are leafwise.
    Batches are trees of leaves with leading batch dimensions. The TreeSpace
    advertises inner-product, star, Jordan, and Euclidean-Jordan capabilities
    only when every leaf advertises the same mathematically valid capability.
    Leaf spaces are converted to one resolved context, so the TreeSpace field
    and representation dtype are inherited uniformly from those leaves. Each
    leaf performs the actual field and exact-dtype membership checks.

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

    def __new__(
        cls,
        treedef: optree.PyTreeSpec | Any,
        leaf_spaces: Sequence[CoordinateSpace],
        *,
        ctx: Context | str | None = None,
        check_level: CheckLevel | str | None = None,
    ):
        if cls is TreeSpace:
            spaces = _validate_leaf_spaces(leaf_spaces)
            resolved_ctx = resolve_context_priority(ctx, *spaces)
            resolved_ctx = _context_with_check_level(resolved_ctx, check_level)
            converted = tuple(space.convert(resolved_ctx) for space in spaces)
            cls = _TREE_REGISTRY.get(_tree_capabilities(converted), TreeSpace)
        return super(TreeSpace, cls).__new__(cls)

    def __init__(
        self,
        treedef: optree.PyTreeSpec | Any,
        leaf_spaces: Sequence[CoordinateSpace],
        *,
        ctx: Context | str | None = None,
        check_level: CheckLevel | str | None = None,
    ) -> None:
        spaces = _validate_leaf_spaces(leaf_spaces, type(self).__name__)
        resolved_ctx = resolve_context_priority(ctx, *spaces)
        resolved_ctx = _context_with_check_level(resolved_ctx, check_level)
        treespec = treedef if isinstance(treedef, optree.PyTreeSpec) else optree.tree_structure(treedef)
        if treespec.num_leaves != len(spaces):
            raise ValueError(
                "TreeSpace leaf-count mismatch: "
                f"treedef has {treespec.num_leaves} leaves but {len(spaces)} leaf spaces were given."
            )

        uniform_spaces = tuple(space.convert(resolved_ctx) for space in spaces)
        self._treedef = treespec
        self._leaf_paths = tuple(optree.treespec_paths(treespec))
        self.leaf_spaces = uniform_spaces
        self._dims = tuple(_prod_int(space.shape) for space in uniform_spaces)
        offsets = [0]
        for dimension in self._dims:
            offsets.append(offsets[-1] + dimension)
        self._offsets = tuple(offsets)
        self._slices = tuple(
            slice(offsets[index], offsets[index + 1]) for index in range(len(self._dims))
        )
        super(TreeSpace, self).__init__((offsets[-1],), resolved_ctx)

    @classmethod
    def from_leaf_spaces(
        cls,
        leaf_spaces: Sequence[CoordinateSpace],
        ctx: Context | str | None = None,
        *,
        check_level: CheckLevel | str | None = None,
    ) -> "TreeSpace":
        """Build a tuple-structured tree from ordered leaf spaces.

        Parameters
        ----------
        leaf_spaces : sequence of CoordinateSpace
            Nonempty ordered spaces for tuple leaves.
        ctx : Context, str, or None, optional
            Backend context resolved from the leaves when omitted.
        check_level : {"none", "cheap", "standard", "strict"}, optional
            Validation policy override.

        Returns
        -------
        TreeSpace
            Tuple-structured finite direct product.
        """
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

    def _ones_for_space(self, space: CoordinateSpace) -> Any:
        ones = getattr(space, "ones", None)
        if callable(ones):
            return ones()
        return self.ops.ones(space.shape, dtype=self.dtype)

    def zero(self) -> Any:
        """Return the additive identity."""
        return self.zeros()

    def zeros(self) -> Any:
        """Return the leafwise additive identity."""
        return self._from_components(tuple(space.zeros() for space in self.leaf_spaces))

    def ones(self) -> Any:
        """Return a leafwise all-ones element when leaf spaces support it."""
        return self._from_components(
            tuple(self._ones_for_space(space) for space in self.leaf_spaces)
        )

    @checked_method(in_space="self", arg_positions=(0, 1))
    def add(self, x: Any, y: Any) -> Any:
        """Return the componentwise sum."""
        return self._from_components(
            tuple(
                space.add(xi, yi)
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

    @checked_method(in_space="self", arg_positions=(1,))
    def scale(self, a: Any, x: Any) -> Any:
        """Return the componentwise scalar product."""
        return self._from_components(
            tuple(
                space.scale(a, leaf)
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

    def _convert(self, new_ctx: Context) -> "TreeSpace":
        """Convert all leaf spaces while preserving structure and paths."""
        return TreeSpace(
            self.treedef,
            tuple(space.convert(new_ctx) for space in self.leaf_spaces),
            ctx=new_ctx,
        )

    def convert_element(
        self, x: Any, new_ctx: Context | str | None = None
    ) -> Any:
        """Convert each leaf and preserve the configured tree structure."""
        target = self.convert(new_ctx)
        converted = []
        for source_space, target_space, leaf in zip(
            self.leaf_spaces, target.leaf_spaces, self._components(x)
        ):
            coordinates = source_space.flatten(leaf)
            converted.append(target_space.unflatten(target.ctx.asarray(coordinates)))
        return target._from_components(tuple(converted))

    def tree_flatten(self):
        """Flatten this space as static JAX pytree data."""
        return (), (self.treedef, self.leaf_spaces, self.ctx)

    @classmethod
    def tree_unflatten(cls, aux: Any, children: Sequence[Any]) -> "TreeSpace":
        """Rebuild this space from JAX pytree auxiliary data."""
        treedef, leaf_spaces, ctx = aux
        return cls(treedef, leaf_spaces, ctx=ctx)


class _LeafwiseHostMixin:
    """Type-only declarations of the TreeSpace host surface used by leaf mixins."""

    if TYPE_CHECKING:
        # Provided by the TreeSpace host these mixins are combined with; leaves
        # are narrowed to the relevant capability per method (see ``cast`` calls).
        @property
        def leaf_spaces(self) -> tuple[CoordinateSpace, ...]: ...
        @property
        def ops(self) -> BackendOps: ...
        @property
        def arity(self) -> int: ...
        def _components(self, x: Any) -> tuple[Any, ...]: ...
        def _from_components(self, parts: tuple[Any, ...]) -> Any: ...


class _LeafwiseInnerProductMixin(_LeafwiseHostMixin):
    """Inner-product operations for trees whose leaves all support them."""

    @checked_method(in_space="self", arg_positions=(0, 1))
    def inner(self, x: Any, y: Any) -> Any:
        """Return the sum of leaf inner products."""
        leaf_spaces = cast("Sequence[InnerProductSpace]", self.leaf_spaces)
        accumulator = None
        for space, xi, yi in zip(
            leaf_spaces, self._components(x), self._components(y)
        ):
            value = space.inner(xi, yi)
            accumulator = value if accumulator is None else accumulator + value
        return accumulator

    def riesz(self, x: Any) -> Any:
        """Apply each leaf space's Riesz map."""
        leaf_spaces = cast("Sequence[InnerProductSpace]", self.leaf_spaces)
        return self._from_components(
            tuple(
                space.riesz(leaf)
                for space, leaf in zip(leaf_spaces, self._components(x))
            )
        )

    def riesz_inverse(self, x: Any) -> Any:
        """Apply each leaf space's inverse Riesz map."""
        leaf_spaces = cast("Sequence[InnerProductSpace]", self.leaf_spaces)
        return self._from_components(
            tuple(
                space.riesz_inverse(leaf)
                for space, leaf in zip(leaf_spaces, self._components(x))
            )
        )

    @property
    def is_euclidean(self) -> bool:
        """Return whether every leaf geometry is Euclidean."""
        leaf_spaces = cast("Sequence[InnerProductSpace]", self.leaf_spaces)
        return all(space.is_euclidean for space in leaf_spaces)


class _LeafwiseStarMixin(_LeafwiseHostMixin):
    """Star operation for trees whose leaves all support it."""

    def star(self, x: Any) -> Any:
        """Return the leafwise star operation in the same tree structure."""
        leaf_spaces = cast("Sequence[StarSpace]", self.leaf_spaces)
        return self._from_components(
            tuple(
                space.star(leaf)
                for space, leaf in zip(leaf_spaces, self._components(x))
            )
        )


@jax_pytree_class
@dataclass(frozen=True)
class TreeSpectralDecomposition:
    """
    Store leafwise Jordan spectral data in deterministic leaf order.

    Parameters
    ----------
    eigvals : tuple
        Eigenvalue data for each TreeSpace leaf.
    frames : tuple
        Spectral frame data for each TreeSpace leaf.
    """

    eigvals: tuple[Any, ...]
    frames: tuple[Any, ...]

    def __repr__(self) -> str:
        """Summarize spectral arrays rather than dumping their full contents."""
        from ..._repr import summarize_value

        return (
            f"TreeSpectralDecomposition(eigvals={summarize_value(self.eigvals)}, "
            f"frames={summarize_value(self.frames)})"
        )

    def tree_flatten(self):
        """Flatten spectral data for JAX pytree registration."""
        return (self.eigvals, self.frames), None

    @classmethod
    def tree_unflatten(cls, aux: Any, children: Sequence[Any]) -> "TreeSpectralDecomposition":
        """Rebuild spectral data from JAX pytree children."""
        eigvals, frames = children
        return cls(tuple(eigvals), tuple(frames))


class _LeafwiseJordanMixin(_LeafwiseHostMixin):
    """Jordan operations for trees whose leaves all support them."""

    @checked_method(in_space="self", arg_positions=(0, 1))
    def jordan(self, x: Any, y: Any) -> Any:
        """Return the leafwise Jordan product."""
        leaf_spaces = cast("Sequence[JordanAlgebraSpace]", self.leaf_spaces)
        return self._from_components(
            tuple(
                space.jordan(xi, yi)
                for space, xi, yi in zip(
                    leaf_spaces, self._components(x), self._components(y)
                )
            )
        )

    def spectrum(self, x: Any) -> DenseArray:
        """Concatenate leaf Jordan spectra in deterministic leaf order."""
        leaf_spaces = cast("Sequence[JordanAlgebraSpace]", self.leaf_spaces)
        parts = tuple(
            space.spectrum(leaf)
            for space, leaf in zip(leaf_spaces, self._components(x))
        )
        if len(parts) == 1:
            return parts[0]
        return self.ops.concatenate(parts, axis=-1)

    def spectral_decompose(self, x: Any) -> TreeSpectralDecomposition:
        """Return leafwise spectral data independent of tree structure."""
        leaf_spaces = cast("Sequence[JordanAlgebraSpace]", self.leaf_spaces)
        decompositions = tuple(
            space.spectral_decompose(leaf)
            for space, leaf in zip(leaf_spaces, self._components(x))
        )
        return TreeSpectralDecomposition(
            eigvals=tuple(eigvals for eigvals, _frame in decompositions),
            frames=tuple(frame for _eigvals, frame in decompositions),
        )

    def from_spectrum(
        self,
        eigvals: TreeSpectralDecomposition,
        frame: Any = None,
    ) -> Any:
        """Reconstruct a tree element from leafwise spectral data."""
        decomposition = eigvals
        if frame is not None:
            raise TypeError("TreeSpace.from_spectrum expects TreeSpectralDecomposition only.")
        if not isinstance(decomposition, TreeSpectralDecomposition):
            raise TypeError(
                "TreeSpace.from_spectrum expects TreeSpectralDecomposition; "
                f"got {type(decomposition).__name__}."
            )
        if len(decomposition.eigvals) != self.arity or len(decomposition.frames) != self.arity:
            raise ValueError("TreeSpace.from_spectrum decomposition arity mismatch.")
        leaf_spaces = cast("Sequence[JordanAlgebraSpace]", self.leaf_spaces)
        return self._from_components(
            tuple(
                space.from_spectrum(eigvals, spectral_frame)
                for space, eigvals, spectral_frame in zip(
                    leaf_spaces,
                    decomposition.eigvals,
                    decomposition.frames,
                )
            )
        )

    @checked_method(in_space="self", out_space="self")
    def spectral_apply(self, x: Any, f: Callable[[Any], Any]) -> Any:
        """Apply each leaf space's spectral calculus independently."""
        leaf_spaces = cast("Sequence[JordanAlgebraSpace]", self.leaf_spaces)
        return self._from_components(
            tuple(
                space.spectral_apply(leaf, f)
                for space, leaf in zip(leaf_spaces, self._components(x))
            )
        )

    def trace(self, x: Any) -> Any:
        """Return the direct-sum trace: the sum of the leaf traces."""
        leaf_spaces = cast("Sequence[JordanAlgebraSpace]", self.leaf_spaces)
        total = None
        for space, leaf in zip(leaf_spaces, self._components(x)):
            value = space.trace(leaf)
            total = value if total is None else total + value
        return total

    def determinant(self, x: Any) -> Any:
        """Return the direct-sum determinant: the product of the leaf determinants."""
        leaf_spaces = cast("Sequence[JordanAlgebraSpace]", self.leaf_spaces)
        total = None
        for space, leaf in zip(leaf_spaces, self._components(x)):
            value = space.determinant(leaf)
            total = value if total is None else total * value
        return total

    def unit(self) -> Any:
        """Return the leafwise Jordan identity assembled into a tree element."""
        leaf_spaces = cast("Sequence[JordanAlgebraSpace]", self.leaf_spaces)
        return self._from_components(tuple(space.unit() for space in leaf_spaces))


class TreeInnerProductSpace(_LeafwiseInnerProductMixin, TreeSpace, InnerProductSpace):
    """TreeSpace specialization whose leaves all have inner products."""


class _TreeStarSpace(_LeafwiseStarMixin, TreeSpace, StarSpace):
    """TreeSpace specialization whose leaves all have star operations."""


class _TreeJordanAlgebraSpace(_LeafwiseJordanMixin, TreeSpace, JordanAlgebraSpace):
    """TreeSpace specialization whose leaves all have Jordan operations."""


class _TreeEuclideanJordanAlgebraSpace(
    _LeafwiseInnerProductMixin,
    _LeafwiseJordanMixin,
    TreeSpace,
    EuclideanJordanAlgebraSpace,
):
    """TreeSpace specialization whose leaves are Euclidean Jordan algebras."""


class _TreeInnerProductStarSpace(
    _LeafwiseInnerProductMixin,
    _LeafwiseStarMixin,
    TreeSpace,
    InnerProductSpace,
    StarSpace,
):
    """TreeSpace specialization with inner-product and star capabilities."""


class _TreeInnerProductJordanSpace(
    _LeafwiseInnerProductMixin,
    _LeafwiseJordanMixin,
    TreeSpace,
    InnerProductSpace,
    JordanAlgebraSpace,
):
    """TreeSpace specialization with inner-product and Jordan capabilities."""


class _TreeStarJordanSpace(
    _LeafwiseStarMixin,
    _LeafwiseJordanMixin,
    TreeSpace,
    StarSpace,
    JordanAlgebraSpace,
):
    """TreeSpace specialization with star and Jordan capabilities."""


class _TreeInnerProductStarJordanSpace(
    _LeafwiseInnerProductMixin,
    _LeafwiseStarMixin,
    _LeafwiseJordanMixin,
    TreeSpace,
    InnerProductSpace,
    StarSpace,
    JordanAlgebraSpace,
):
    """TreeSpace specialization with inner-product, star, and Jordan capabilities."""


class _TreeEuclideanJordanStarSpace(
    _LeafwiseStarMixin,
    _TreeEuclideanJordanAlgebraSpace,
    StarSpace,
):
    """TreeSpace specialization with Euclidean-Jordan and star capabilities."""


_TREE_REGISTRY.update(
    {
        frozenset(): TreeSpace,
        frozenset({_CAP_INNER}): TreeInnerProductSpace,
        frozenset({_CAP_STAR}): _TreeStarSpace,
        frozenset({_CAP_JORDAN}): _TreeJordanAlgebraSpace,
        frozenset({_CAP_INNER, _CAP_STAR}): _TreeInnerProductStarSpace,
        frozenset({_CAP_INNER, _CAP_JORDAN}): _TreeInnerProductJordanSpace,
        frozenset({_CAP_STAR, _CAP_JORDAN}): _TreeStarJordanSpace,
        frozenset({_CAP_INNER, _CAP_STAR, _CAP_JORDAN}): _TreeInnerProductStarJordanSpace,
        frozenset(
            {_CAP_INNER, _CAP_JORDAN, _CAP_EUCLIDEAN_JORDAN}
        ): _TreeEuclideanJordanAlgebraSpace,
        frozenset(
            {_CAP_INNER, _CAP_STAR, _CAP_JORDAN, _CAP_EUCLIDEAN_JORDAN}
        ): _TreeEuclideanJordanStarSpace,
    }
)

for _tree_type in set(_TREE_REGISTRY.values()):
    jax_pytree_class(_tree_type)


__all__ = [
    "TreeElement",
    "TreeSpace",
    "TreeSpectralDecomposition",
    "_space_capabilities",
    "_tree_capabilities",
]
