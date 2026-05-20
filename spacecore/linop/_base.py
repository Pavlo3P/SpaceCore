from __future__ import annotations

from abc import abstractmethod
from math import prod
from numbers import Number
from typing import Any, Generic, TypeVar

from ..space import Space
from ..backend import Context
from .._contextual import ContextBound
from .._contextual.manager import ctx_manager

Domain = TypeVar('Domain', bound=Space)
Codomain = TypeVar('Codomain', bound=Space)

class LinOp(ContextBound, Generic[Domain, Codomain]):
    """
    Minimal linear operator (morphism) between two spaces.

    This class is intentionally small. It defines no matrix semantics,
    arithmetic, or storage assumptions.

    Its sole purpose is to represent a linear map
    ``A : dom -> cod``
    with access to both forward and adjoint actions.
    """

    def __init__(self, dom: Domain, cod: Codomain, ctx: Context | str | None = None):
        ctx = ctx_manager.resolve_context_priority(ctx, dom, cod)
        super(LinOp, self).__init__(ctx)

        self.dom = dom.convert(self.ctx)
        self.cod = cod.convert(self.ctx)
        self._enable_checks = self.ctx.enable_checks

    @property
    def domain(self) -> Domain:
        """Domain space of this linear operator."""
        return self.dom

    @property
    def codomain(self) -> Codomain:
        """Codomain space of this linear operator."""
        return self.cod

    @abstractmethod
    def apply(self, x: Any) -> Any:
        """
        Forward application: y = A x

        Contract:
          - x is an element of self.dom
          - return value is an element of self.cod
        """

    @abstractmethod
    def rapply(self, y: Any) -> Any:
        """
        Adjoint application: x = A^* y

        Contract:
          - y is an element of self.cod
          - return value is an element of self.dom
        """

    def __call__(self, x: Any) -> Any:
        """Apply this linear operator to ``x``."""
        return self.apply(x)

    def adjoint_apply(self, y: Any) -> Any:
        """Apply the adjoint of this linear operator to ``y``."""
        return self.rapply(y)

    @property
    def H(self) -> LinOp:
        """Hermitian-adjoint view of this linear operator."""
        from ._algebra import _AdjointViewLinOp

        return _AdjointViewLinOp(self)

    def __add__(self, other: Any) -> LinOp:
        """Return the lazy sum ``self + other`` of two compatible operators."""
        from ._algebra import make_sum

        if not isinstance(other, LinOp):
            return NotImplemented
        return make_sum((self, other))

    def __radd__(self, other: Any) -> LinOp:
        """Return the lazy sum ``other + self`` of two compatible operators."""
        from ._algebra import make_sum

        if isinstance(other, Number) and other == 0:
            return self
        if not isinstance(other, LinOp):
            return NotImplemented
        return make_sum((other, self))

    def __neg__(self) -> LinOp:
        """Return the lazy negation ``-self``."""
        from ._algebra import make_scaled

        return make_scaled(-1, self)

    def __sub__(self, other: Any) -> LinOp:
        """Return the lazy difference ``self - other`` of two compatible operators."""
        from ._algebra import make_scaled, make_sum

        if not isinstance(other, LinOp):
            return NotImplemented
        return make_sum((self, make_scaled(-1, other)))

    def __rsub__(self, other: Any) -> LinOp:
        """Return the lazy difference ``other - self`` of two compatible operators."""
        from ._algebra import make_scaled, make_sum

        if isinstance(other, Number) and other == 0:
            return make_scaled(-1, self)
        if not isinstance(other, LinOp):
            return NotImplemented
        return make_sum((other, make_scaled(-1, self)))

    def __mul__(self, scalar: Any) -> LinOp:
        """Return the lazy right scalar multiple ``self * scalar``."""
        from ._algebra import is_scalar_like, make_scaled

        if not is_scalar_like(scalar):
            return NotImplemented
        return make_scaled(scalar, self)

    def __rmul__(self, scalar: Any) -> LinOp:
        """Return the lazy left scalar multiple ``scalar * self``."""
        from ._algebra import is_scalar_like, make_scaled

        if not is_scalar_like(scalar):
            return NotImplemented
        return make_scaled(scalar, self)

    def __matmul__(self, other: Any) -> LinOp:
        """Return the lazy composition ``self @ other`` of two compatible operators."""
        from ._algebra import make_composed

        if not isinstance(other, LinOp):
            return NotImplemented
        return make_composed(self, other)

    def adjoint(self) -> LinOp:
        """Return the Hermitian-adjoint view of this linear operator."""
        return self.H

    def to_dense(self) -> Any:
        """
        Materialize this operator as a dense backend array.

        The returned array has shape ``self.codomain.shape + self.domain.shape``.
        The default implementation applies the operator to each standard basis
        vector of the domain, stacks the flattened outputs as matrix columns,
        and reshapes the result back to tensor-operator form. Subclasses that
        already store the matrix should override this method for efficiency.
        """
        domain_size = prod(self.domain.shape)
        codomain_size = prod(self.codomain.shape)
        zero = self.ops.zeros((domain_size,), dtype=self.dtype)
        columns = []
        for i in range(domain_size):
            basis_vector = self.ops.index_set(zero, i, 1, copy=True)
            x = self.domain.unflatten(basis_vector)
            y = self.apply(x)
            columns.append(self.codomain.flatten(y))
        matrix = self.ops.stack(tuple(columns), axis=1)
        return self.ops.reshape(matrix, tuple(self.codomain.shape) + tuple(self.domain.shape))

    def assert_domain(self, x: Any) -> None:
        self.dom.check_member(x)

    def assert_codomain(self, y: Any) -> None:
        self.cod.check_member(y)

    def __eq__(self, other: Any) -> bool:
        return NotImplemented

    @abstractmethod
    def tree_flatten(self):
        ...

    @classmethod
    @abstractmethod
    def tree_unflatten(cls, aux, children):
        ...
