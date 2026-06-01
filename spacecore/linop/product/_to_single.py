from __future__ import annotations

from typing import Any, Sequence, Tuple

from ._base import ProductLinOp
from .._base import LinOp, Codomain
from ..._batching import _check_batched
from ...space import ProductSpace, VectorSpace
from ...backend import jax_pytree_class, Context


@jax_pytree_class
class SumToSingleLinOp(ProductLinOp[ProductSpace, Codomain]):
    r"""
    Sum of component operators from a product domain into a single codomain.

    If ``dom = X1 x ... x Xk`` and ``cod = Y``, component ``parts[i]`` maps
    ``Xi`` to ``Y``. Forward application sums component outputs in ``Y``;
    adjoint application returns the tuple of component adjoints.

    Parameters
    ----------
    dom : ProductSpace
        Product domain.
    cod : Space
        Shared codomain.
    parts : sequence of LinOp
        Operators from each product component to ``cod``.
    ctx : Context, str, or None, optional
        Backend context specification.
    """

    def __init__(
        self,
        dom: ProductSpace,
        cod: Codomain,
        parts: Sequence[LinOp],
        ctx: Context | str | None = None,
    ) -> None:
        super().__init__(dom, cod, parts, ctx)
        self._flat_dense_apply_mats = self._make_flat_dense_apply_mats()

    def _make_flat_dense_apply_mats(self):
        """Return dense matrices for the exact flat-vector fast path."""
        if type(self.cod) is not VectorSpace or not self.cod.is_euclidean:
            return None
        if tuple(self.cod.shape) != (self.cod._size,):
            return None
        mats = []
        for op in self.parts:
            if (
                type(op.dom) is not VectorSpace
                or not op.dom.is_euclidean
                or tuple(op.dom.shape) != (op.dom._size,)
                or not hasattr(op, "_A2")
            ):
                return None
            mats.append(op._A2)
        return tuple(mats)

    def _check_layout(self) -> None:
        """Check that every component maps one product part to the shared codomain."""
        if not isinstance(self.dom, ProductSpace):
            raise TypeError("SumToSingleLinOp expects dom to be ProductSpace.")

        if len(self.parts) != len(self.dom.spaces):
            raise ValueError("Number of ops must match product arity.")

        for i, A in enumerate(self.parts):
            if A.dom == self.dom.spaces[i] and A.cod == self.cod:
                continue
            else:
                raise TypeError(f"Component op {i} must map dom.spaces[{i}] -> cod.")

    def apply(self, x: Any) -> Any:
        """Apply component operators and sum in the codomain."""
        if self._enable_checks:
            self.dom._check_member(x)
        y = self._apply_unchecked(x)
        if self._enable_checks:
            self.cod._check_member(y)
        return y

    def _apply_unchecked(self, x: Any) -> Any:
        """Apply component operators without membership checks."""
        mats = self._flat_dense_apply_mats
        if mats is not None:
            if self._num_parts == 2:
                return mats[0] @ x[0] + mats[1] @ x[1]
            acc = mats[0] @ x[0]
            for mat, xi in zip(mats[1:], x[1:]):
                acc = acc + mat @ xi
            return acc
        if self._num_parts == 2:
            y0 = self._apply_parts[0](x[0])
            y1 = self._apply_parts[1](x[1])
            if type(self.cod) is VectorSpace:
                return y0 + y1
            return self.cod.add(y0, y1)
        acc = None
        for apply, xi in zip(self._apply_parts, x):
            yi = apply(xi)
            if acc is None:
                acc = yi
            elif type(self.cod) is VectorSpace:
                acc = acc + yi
            else:
                acc = self.cod.add(acc, yi)
        return acc

    def rapply(self, y: Any) -> Any:
        """Apply each component adjoint to the shared codomain element."""
        if self._enable_checks:
            self.cod._check_member(y)
        x = self._rapply_unchecked(y)
        if self._enable_checks:
            self.dom._check_member(x)
        return x

    def _rapply_unchecked(self, y: Any) -> Any:
        """Apply component adjoints without membership checks."""
        if self._num_parts == 2:
            return self._rapply_parts[0](y), self._rapply_parts[1](y)
        return tuple(rapply(y) for rapply in self._rapply_parts)

    def vapply(self, x: Any) -> Any:
        """Apply this sum-to-single operator over a product batch."""
        if self._enable_checks:
            _check_batched(self.domain, x)
        acc = self._vapply_unchecked(x)
        if self._enable_checks:
            _check_batched(self.codomain, acc)
        return acc

    def _vapply_unchecked(self, x: Any) -> Any:
        """Apply over a product batch without membership checks."""
        mats = self._flat_dense_apply_mats
        if mats is not None:
            if self._num_parts == 2:
                acc = x[0] @ mats[0].T + x[1] @ mats[1].T
            else:
                acc = x[0] @ mats[0].T
                for mat, xi in zip(mats[1:], x[1:]):
                    acc = acc + xi @ mat.T
            return acc
        acc = None
        for op, xi in zip(self.parts, x):
            yi = op.vapply(xi)
            if acc is None:
                acc = yi
            elif type(self.codomain) is VectorSpace:
                acc = acc + yi
            else:
                acc = self.codomain.add_batch(acc, yi)
        return acc

    def rvapply(self, y: Any) -> Any:
        """Apply the adjoint over a codomain batch."""
        if self._enable_checks:
            _check_batched(self.codomain, y)
        x = self._rvapply_unchecked(y)
        if self._enable_checks:
            _check_batched(self.domain, x)
        return x

    def _rvapply_unchecked(self, y: Any) -> Any:
        """Apply the adjoint over a codomain batch without checks."""
        x = tuple(op.rvapply(y) for op in self.parts)
        return x

    @classmethod
    def from_operators(cls, parts: Tuple[LinOp, ...]) -> SumToSingleLinOp:
        """Build a sum-to-single operator from component operators."""
        if not parts:
            raise ValueError("Parts must be non-empty.")

        dom = ProductSpace(tuple(op.dom for op in parts))
        cod = parts[0].cod

        return cls(dom, cod, parts)

    def _convert(self, new_ctx: Context) -> SumToSingleLinOp:
        """Convert spaces and component operators to ``new_ctx``."""
        new_dom = self.dom.convert(new_ctx)
        new_cod = self.cod.convert(new_ctx)
        new_parts = [op.convert(new_ctx) for op in self.parts]
        return SumToSingleLinOp(new_dom, new_cod, new_parts, new_ctx)
