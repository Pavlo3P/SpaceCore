from __future__ import annotations

from abc import abstractmethod
from typing import Any, Callable

from ._inner_product import InnerProductSpace
from ._vector import VectorSpace


class JordanAlgebraSpace(VectorSpace):
    """
    Vector space capability with a Jordan product and spectral calculus.

    Parameters
    ----------
    ctx : Context, str, or None, optional
        Context specification used for elements and validation checks.
    """

    @abstractmethod
    def jordan(self, x: Any, y: Any) -> Any:
        """Return the Jordan product of ``x`` and ``y``."""

    @abstractmethod
    def spectrum(self, x: Any) -> Any:
        """Return Jordan-algebraic eigenvalues of ``x``."""

    @abstractmethod
    def spectral_decompose(self, x: Any) -> Any:
        """Return spectral data sufficient to reconstruct ``x``."""

    @abstractmethod
    def from_spectrum(self, eigvals: Any, frame: Any) -> Any:
        """Reconstruct an element from spectral data."""

    def spectral_apply(self, x: Any, f: Callable) -> Any:
        eigvals, frame = self.spectral_decompose(x)
        feigvals = f(eigvals)
        return self.from_spectrum(feigvals, frame)

    @abstractmethod
    def unit(self) -> Any:
        """Return the Jordan-algebraic identity element ``e`` (all-ones spectrum)."""

    def trace(self, x: Any) -> Any:
        """Return the Jordan trace ``sum_i lambda_i(x)`` of the spectrum."""
        return self.ops.sum(self.spectrum(x), axis=-1)

    def determinant(self, x: Any) -> Any:
        """Return the Jordan determinant ``prod_i lambda_i(x)`` of the spectrum."""
        return self.ops.prod(self.spectrum(x), axis=-1)


class EuclideanJordanAlgebraSpace(JordanAlgebraSpace, InnerProductSpace):
    """
    Jordan algebra capability with a compatible inner product.

    Parameters
    ----------
    ctx : Context, str, or None, optional
        Context specification used for elements and validation checks.
    """
