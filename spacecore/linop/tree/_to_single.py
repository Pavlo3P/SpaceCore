from __future__ import annotations

from typing import Any, Sequence, Tuple, cast

from ._base import TreeLinOp
from .._base import LinOp, Codomain
from ..._checks import checked_method
from ...space import DenseCoordinateSpace, DenseVectorSpace, ElementwiseJordanSpace, TreeSpace
from ...backend import jax_pytree_class, Context


@jax_pytree_class
class SumToSingleLinOp(TreeLinOp[TreeSpace, Codomain]):
    r"""
    Represent a sum of leaf operators from a tree domain.

    If ``dom = X1 x ... x Xk`` and ``cod = Y``, component ``parts[i]`` maps
    ``Xi`` to ``Y``. Forward application sums component outputs in ``Y``;
    adjoint application returns a value with ``dom.treedef``.

    Parameters
    ----------
    dom : TreeSpace
        Tree-structured domain.
    cod : Space
        Shared codomain.
    parts : sequence of LinOp
        Operators from each product component to ``cod``.
    ctx : Context, str, or None, optional
        Backend context specification.
    """

    def __init__(
        self,
        dom: TreeSpace,
        cod: Codomain,
        parts: Sequence[LinOp],
        ctx: Context | str | None = None,
    ) -> None:
        super().__init__(dom, cod, parts, ctx)
        self._flat_dense_apply_mats = self._make_flat_dense_apply_mats()

    def _make_flat_dense_apply_mats(self):
        """Return dense matrices for the exact flat-vector fast path."""
        if (
            type(self.cod) not in (DenseCoordinateSpace, DenseVectorSpace, ElementwiseJordanSpace)
            or not self.cod.is_euclidean
        ):
            return None
        if tuple(self.cod.shape) != (self.cod._size,):
            return None
        mats = []
        for op in self.parts:
            if (
                type(op.dom) not in (DenseCoordinateSpace, DenseVectorSpace, ElementwiseJordanSpace)
                or not op.dom.is_euclidean
                or tuple(op.dom.shape) != (op.dom._size,)
                or not hasattr(op, "_A2")
            ):
                return None
            mats.append(cast(Any, op)._A2)
        return tuple(mats)

    def _check_layout(self) -> None:
        """Check that every component maps one product part to the shared codomain."""
        if not isinstance(self.dom, TreeSpace):
            raise TypeError("SumToSingleLinOp expects dom to be TreeSpace.")

        if len(self.parts) != self.dom.arity:
            raise ValueError("Number of ops must match domain tree arity.")

        for i, A in enumerate(self.parts):
            if A.dom == self.dom.leaf_spaces[i] and A.cod == self.cod:
                continue
            else:
                raise TypeError(f"Component op {i} must map dom.leaf_spaces[{i}] -> cod.")

    @checked_method(in_space="domain", out_space="codomain")
    def apply(self, x: Any) -> Any:
        """Apply operators to components of a domain product element and sum."""
        return self._apply_unchecked(x)

    def _apply_unchecked(self, x: Any) -> Any:
        """Apply component operators without membership checks."""
        x_parts = self.dom._components(x)
        mats = self._flat_dense_apply_mats
        if mats is not None:
            if self._num_parts == 2:
                return mats[0] @ x_parts[0] + mats[1] @ x_parts[1]
            acc = mats[0] @ x_parts[0]
            for mat, xi in zip(mats[1:], x_parts[1:]):
                acc = acc + mat @ xi
            return acc
        if self._num_parts == 2:
            y0 = self._apply_parts[0](x_parts[0])
            y1 = self._apply_parts[1](x_parts[1])
            if type(self.cod) in (DenseCoordinateSpace, DenseVectorSpace, ElementwiseJordanSpace):
                return y0 + y1
            return self.cod.add(y0, y1)
        acc = None
        for apply, xi in zip(self._apply_parts, x_parts):
            yi = apply(xi)
            if acc is None:
                acc = yi
            elif type(self.cod) in (DenseCoordinateSpace, DenseVectorSpace, ElementwiseJordanSpace):
                acc = acc + yi
            else:
                acc = self.cod.add(acc, yi)
        return acc

    @checked_method(in_space="codomain", out_space="domain")
    def rapply(self, y: Any) -> Any:
        """Apply component adjoints and return a domain product element."""
        return self._rapply_unchecked(y)

    def _rapply_unchecked(self, y: Any) -> Any:
        """Apply component adjoints without checks and rebuild domain representation."""
        if self._num_parts == 2:
            x_parts = (self._rapply_parts[0](y), self._rapply_parts[1](y))
        else:
            x_parts = tuple(rapply(y) for rapply in self._rapply_parts)
        return self.dom._from_components(x_parts)

    @checked_method(in_space="domain", out_space="codomain", in_batched=True, out_batched=True)
    def vapply(self, x: Any) -> Any:
        """Apply this sum-to-single operator over a structured product batch."""
        return self._vapply_unchecked(x)

    def _vapply_unchecked(self, x: Any) -> Any:
        """Apply over a product batch without membership checks."""
        x_parts = self.dom._components(x)
        mats = self._flat_dense_apply_mats
        if mats is not None:
            if self._num_parts == 2:
                acc = x_parts[0] @ mats[0].T + x_parts[1] @ mats[1].T
            else:
                acc = x_parts[0] @ mats[0].T
                for mat, xi in zip(mats[1:], x_parts[1:]):
                    acc = acc + xi @ mat.T
            return acc
        acc = None
        for op, xi in zip(self.parts, x_parts):
            yi = op.vapply(xi)
            if acc is None:
                acc = yi
            elif type(self.codomain) in (
                DenseCoordinateSpace,
                DenseVectorSpace,
                ElementwiseJordanSpace,
            ):
                acc = acc + yi
            else:
                acc = self.codomain.add_batch(acc, yi)
        return acc

    @checked_method(in_space="codomain", out_space="domain", in_batched=True, out_batched=True)
    def rvapply(self, y: Any) -> Any:
        """Apply the adjoint over a codomain batch and preserve domain structure."""
        return self._rvapply_unchecked(y)

    def _rvapply_unchecked(self, y: Any) -> Any:
        """Apply the adjoint over a codomain batch without checks and rebuild domain representation."""
        x_parts = tuple(op.rvapply(y) for op in self.parts)
        return self.dom._from_components(x_parts)

    @classmethod
    def from_operators(cls, parts: Tuple[LinOp, ...]) -> SumToSingleLinOp:
        """Build a sum-to-single operator from component operators."""
        if not parts:
            raise ValueError("Parts must be non-empty.")

        dom = TreeSpace(tuple(range(len(parts))), tuple(op.dom for op in parts))
        cod = parts[0].cod

        return cls(dom, cod, parts)

    def _convert(self, new_ctx: Context) -> SumToSingleLinOp:
        """Convert spaces and component operators to ``new_ctx``."""
        new_dom = self.dom.convert(new_ctx)
        new_cod = self.cod.convert(new_ctx)
        new_parts = [op.convert(new_ctx) for op in self.parts]
        return SumToSingleLinOp(new_dom, new_cod, new_parts, new_ctx)
