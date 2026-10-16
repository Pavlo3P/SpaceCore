"""Capability mixins and the dispatch subclasses TreeSpace picks from."""

from __future__ import annotations

import operator
from dataclasses import dataclass
from functools import reduce
from typing import TYPE_CHECKING, Any, Callable, Sequence, cast

import optree

from ..._checks import checked_method
from ...backend import BackendOps
from ...backend import PyTreeNode
from ..base import (
    CoordinateSpace,
    InnerProductSpace,
    JordanAlgebraSpace,
)

from .._capabilities import (
    _CAP_INNER, _CAP_STAR, _CAP_JORDAN,
)

from ._dispatch import build_dispatch_classes
from ._tree_space import _TREE_REGISTRY, TreeSpace


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
        def _leafwise(self, fn: Callable[..., Any], *xs: Any) -> tuple[Any, ...]: ...


class _LeafwiseInnerProductMixin(_LeafwiseHostMixin):
    """Inner-product operations for trees whose leaves all support them."""

    def _inner_core(self, x: Any, y: Any) -> Any:
        """Return the sum of leaf inner products."""
        # Sum of the leaves' inner products.
        return reduce(operator.add, self._leafwise(lambda s, xi, yi: s._inner_core(xi, yi), x, y))

    def _riesz_core(self, x: Any) -> Any:
        """Apply each leaf space's Riesz map."""
        return self._from_components(self._leafwise(lambda s, leaf: s._riesz_core(leaf), x))

    def _riesz_inverse_core(self, x: Any) -> Any:
        """Apply each leaf space's inverse Riesz map."""
        return self._from_components(self._leafwise(lambda s, leaf: s._riesz_inverse_core(leaf), x))

    @property
    def is_euclidean(self) -> bool:
        """Return whether every leaf geometry is Euclidean."""
        leaf_spaces = cast("Sequence[InnerProductSpace]", self.leaf_spaces)
        return all(space.is_euclidean for space in leaf_spaces)


class _LeafwiseStarMixin(_LeafwiseHostMixin):
    """Star operation for trees whose leaves all support it."""

    def _star_core(self, x: Any) -> Any:
        """The leafwise star operation in the same tree structure."""
        return self._from_components(self._leafwise(lambda s, leaf: s._star_core(leaf), x))


@dataclass(frozen=True)
class TreeSpectralDecomposition(PyTreeNode):
    """
    Leafwise Jordan spectral data, in leaf order.

    Parameters
    ----------
    eigvals : tuple
        Eigenvalues of each leaf.
    frames : tuple
        Spectral frames of each leaf.
    treedef : optree.PyTreeSpec or None, optional
        The producing space's structure, used by :meth:`to_tree`; ``None`` for hand-built
        data. It is static pytree data, so decompositions merged in one ``lax.cond`` or
        ``while_loop`` must agree on it.
    """

    eigvals: tuple[Any, ...]
    frames: tuple[Any, ...]
    treedef: Any = None

    def __repr__(self) -> str:
        """Summarize spectral arrays rather than dumping their full contents."""
        from ..._repr import summarize_value

        return (
            f"TreeSpectralDecomposition(eigvals={summarize_value(self.eigvals)}, "
            f"frames={summarize_value(self.frames)})"
        )

    def to_tree(self) -> Any:
        """Return the eigenvalues as a pytree matching the space ``treedef``.

        The result mirrors the tree structure (nested for a nested tree), each leaf
        holding that leaf's eigenvalue vector. It is a structure-only view: the
        leaves are eigenvalue-shaped, so it is **not** a member of the space and
        must not be fed back into space operations. Requires ``treedef`` (set on
        decompositions produced by :meth:`TreeSpace.spectral_decompose`).
        """
        # Rebuild the eigenvalues in the producing tree's structure.
        if self.treedef is None:
            raise ValueError(
                "TreeSpectralDecomposition.to_tree() requires 'treedef'; it is set "
                "only on decompositions produced by TreeSpace.spectral_decompose."
            )
        return optree.tree_unflatten(self.treedef, self.eigvals)

    def tree_flatten(self):
        """Flatten spectral data for JAX pytree registration (``treedef`` is static aux)."""
        return (self.eigvals, self.frames), self.treedef

    @classmethod
    def tree_unflatten(cls, aux: Any, children: Sequence[Any]) -> "TreeSpectralDecomposition":
        """Rebuild spectral data from JAX pytree children."""
        eigvals, frames = children
        return cls(tuple(eigvals), tuple(frames), treedef=aux)


class _LeafwiseJordanMixin(_LeafwiseHostMixin):
    """Jordan operations for trees whose leaves all support them."""

    @checked_method(in_space="self", arg_positions=(0, 1))
    def jordan(self, x: Any, y: Any) -> Any:
        """Return the leafwise Jordan product."""
        return self._from_components(self._leafwise(lambda s, xi, yi: s.jordan(xi, yi), x, y))

    def spectrum(self, x: Any, *, structured: bool = False) -> Any:
        """Return the leaf Jordan spectra.

        By default (``structured=False``) the leaf spectra are concatenated in
        deterministic leaf order into one flat array — the direct-sum spectrum, as
        spectral norms and ``trace``/``determinant`` consume it. With
        ``structured=True`` the result mirrors the space ``treedef`` instead: a
        (possibly nested) pytree whose leaves hold each leaf's eigenvalue vector.
        The structured result is a structure-only view — its leaves are
        eigenvalue-shaped, so it is **not** a member of the space.
        """
        # Per-leaf spectra: as a tree, or concatenated into one vector.
        parts = self._leafwise(lambda s, leaf: s.spectrum(leaf), x)
        if structured:
            return self._from_components(parts)
        if len(parts) == 1:
            return parts[0]
        return self.ops.concatenate(parts, axis=-1)

    def spectral_decompose(self, x: Any) -> TreeSpectralDecomposition:
        """Return leafwise spectral data tagged with the space ``treedef``.

        Eigenvalues and frames are stored as flat leaf-order tuples (so
        :meth:`from_spectrum` round-trips, including on nested trees); the
        ``treedef`` lets :meth:`TreeSpectralDecomposition.to_tree` expose the
        eigenvalues in the tree's own structure.
        """
        # Decompose each leaf and keep the tree structure for to_tree.
        decompositions = self._leafwise(lambda s, leaf: s.spectral_decompose(leaf), x)
        return TreeSpectralDecomposition(
            eigvals=tuple(eigvals for eigvals, _frame in decompositions),
            frames=tuple(frame for _eigvals, frame in decompositions),
            treedef=self.treedef,
        )

    def from_spectrum(
        self,
        eigvals: TreeSpectralDecomposition,
        frame: Any = None,
    ) -> Any:
        """Reconstruct a tree element from leafwise spectral data."""
        # Accept only a TreeSpectralDecomposition of matching arity, then rebuild leafwise.
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
        return self._from_components(self._leafwise(lambda s, leaf: s.spectral_apply(leaf, f), x))

    def trace(self, x: Any) -> Any:
        """Return the direct-sum trace: the sum of the leaf traces."""
        # The trace of a direct product is the sum of the leaves' traces.
        return reduce(operator.add, self._leafwise(lambda s, leaf: s.trace(leaf), x))

    def determinant(self, x: Any) -> Any:
        """Return the direct-sum determinant: the product of the leaf determinants."""
        # The determinant of a direct product is the product of the leaves' determinants.
        return reduce(operator.mul, self._leafwise(lambda s, leaf: s.determinant(leaf), x))

    def unit(self) -> Any:
        """Return the leafwise Jordan identity assembled into a tree element."""
        leaf_spaces = cast("Sequence[JordanAlgebraSpace]", self.leaf_spaces)
        return self._from_components(tuple(space.unit() for space in leaf_spaces))


_TREE_REGISTRY.update(build_dispatch_classes(
    TreeSpace,
    {_CAP_INNER: _LeafwiseInnerProductMixin, _CAP_STAR: _LeafwiseStarMixin, _CAP_JORDAN: _LeafwiseJordanMixin},
    prefix="_Tree",
    module_globals=globals(),
))
# The capability subclasses register as pytrees through ``PyTreeNode.__init_subclass__``.
TreeInnerProductSpace = _TreeInnerProductSpace  # noqa: F821 - bound by build_dispatch_classes; historical public name
