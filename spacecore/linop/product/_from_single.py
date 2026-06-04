from __future__ import annotations

from typing import Any, Sequence, Tuple

from ._base import ProductLinOp
from .._base import LinOp, Domain
from ..._checks import checked_method
from ...space import ProductSpace, VectorSpace
from ...backend import jax_pytree_class, Context


@jax_pytree_class
class StackedLinOp(ProductLinOp[Domain, ProductSpace]):
    r"""
    Stack of operators from a single domain into a product codomain.

    If ``dom = X`` and ``cod = Y1 x ... x Yk``, component ``parts[i]`` maps
    ``X`` to ``Yi``. Forward application returns a tuple of component outputs;
    adjoint application sums component adjoints in ``X``.

    Parameters
    ----------
    dom : Space
        Shared component domain.
    cod : ProductSpace
        Product codomain.
    parts : sequence of LinOp
        Operators from ``dom`` to each component of ``cod``.
    ctx : Context, str, or None, optional
        Backend context specification.
    """

    def __init__(
        self,
        dom: Domain,
        cod: ProductSpace,
        parts: Sequence[LinOp],
        ctx: Context | str | None = None,
    ) -> None:
        super().__init__(dom, cod, parts, ctx)
        self._flat_dense_rapply_mats = self._make_flat_dense_rapply_mats()

    def _make_flat_dense_rapply_mats(self):
        """Return dense adjoint matrices for the exact flat-vector fast path."""
        if type(self.dom) is not VectorSpace or not self.dom.is_euclidean:
            return None
        if tuple(self.dom.shape) != (self.dom._size,):
            return None
        mats = []
        for op in self.parts:
            if (
                type(op.cod) is not VectorSpace
                or not op.cod.is_euclidean
                or tuple(op.cod.shape) != (op.cod._size,)
                or not hasattr(op, "_A2H")
            ):
                return None
            mats.append(op._A2H)
        return tuple(mats)

    def _check_layout(self) -> None:
        """Check that every component maps the shared domain to one codomain part."""
        if not isinstance(self.cod, ProductSpace):
            raise TypeError("StackedLinOp expects cod to be ProductSpace.")

        if len(self.parts) != len(self.cod.spaces):
            raise ValueError("Number of ops must match codomain product arity.")

        for i, A in enumerate(self.parts):
            if A.dom == self.dom and A.cod == self.cod.spaces[i]:
                continue
            else:
                raise TypeError(f"Component op {i} must map dom -> cod.spaces[{i}].")

    @checked_method(in_space="domain", out_space="codomain")
    def apply(self, x: Any) -> Any:
        """Apply each component operator to the same input."""
        return self._apply_unchecked(x)

    def _apply_unchecked(self, x: Any) -> Any:
        """Apply component operators without membership checks."""
        if self._num_parts == 2:
            y_parts = (self._apply_parts[0](x), self._apply_parts[1](x))
        else:
            y_parts = tuple(apply(x) for apply in self._apply_parts)
        return self.cod._from_components(y_parts)

    @checked_method(in_space="codomain", out_space="domain")
    def rapply(self, y: Any) -> Any:
        """Apply component adjoints and sum in the shared domain."""
        return self._rapply_unchecked(y)

    def _rapply_unchecked(self, y: Any) -> Any:
        """Apply component adjoints without membership checks."""
        y_parts = self.cod._components(y)
        mats = self._flat_dense_rapply_mats
        if mats is not None:
            if self._num_parts == 2:
                return mats[0] @ y_parts[0] + mats[1] @ y_parts[1]
            acc = mats[0] @ y_parts[0]
            for mat, yi in zip(mats[1:], y_parts[1:]):
                acc = acc + mat @ yi
            return acc
        if self._num_parts == 2:
            x0 = self._rapply_parts[0](y_parts[0])
            x1 = self._rapply_parts[1](y_parts[1])
            if type(self.dom) is VectorSpace:
                return x0 + x1
            return self.dom.add(x0, x1)
        acc = None
        for rapply, yi in zip(self._rapply_parts, y_parts):
            xi = rapply(yi)
            if acc is None:
                acc = xi
            elif type(self.dom) is VectorSpace:
                acc = acc + xi
            else:
                acc = self.dom.add(acc, xi)
        return acc

    @checked_method(in_space="domain", out_space="codomain", in_batched=True, out_batched=True)
    def vapply(self, x: Any) -> Any:
        """Apply this stacked operator over a batch."""
        return self._vapply_unchecked(x)

    def _vapply_unchecked(self, x: Any) -> Any:
        """Apply over a batch without membership checks."""
        y_parts = tuple(op.vapply(x) for op in self.parts)
        return self.cod._from_components(y_parts)

    @checked_method(in_space="codomain", out_space="domain", in_batched=True, out_batched=True)
    def rvapply(self, y: Any) -> Any:
        """Apply the adjoint stacked operator over a product batch."""
        return self._rvapply_unchecked(y)

    def _rvapply_unchecked(self, y: Any) -> Any:
        """Apply the adjoint over a product batch without membership checks."""
        y_parts = self.cod._components(y)
        mats = self._flat_dense_rapply_mats
        if mats is not None:
            if self._num_parts == 2:
                acc = y_parts[0] @ mats[0].T + y_parts[1] @ mats[1].T
            else:
                acc = y_parts[0] @ mats[0].T
                for mat, yi in zip(mats[1:], y_parts[1:]):
                    acc = acc + yi @ mat.T
            return acc
        acc = None
        for op, yi in zip(self.parts, y_parts):
            xi = op.rvapply(yi)
            if acc is None:
                acc = xi
            elif type(self.domain) is VectorSpace:
                acc = acc + xi
            else:
                acc = self.domain.add_batch(acc, xi)
        return acc

    @classmethod
    def from_operators(cls, parts: Tuple[LinOp, ...]) -> StackedLinOp:
        """Build a stacked operator from component operators."""
        if not parts:
            raise ValueError("Parts must be non-empty.")

        cod = ProductSpace(tuple(op.cod for op in parts))
        dom = parts[0].dom

        return cls(dom, cod, parts)

    def _convert(self, new_ctx: Context) -> StackedLinOp:
        """Convert spaces and component operators to ``new_ctx``."""
        new_dom = self.dom.convert(new_ctx)
        new_cod = self.cod.convert(new_ctx)
        new_parts = [op.convert(new_ctx) for op in self.parts]
        return StackedLinOp(new_dom, new_cod, new_parts, new_ctx)
