from __future__ import annotations

from abc import abstractmethod
from typing import Any, Callable

from ..._checks import checked_method
from ._inner_product import InnerProductSpace
from ._vector import VectorSpace


class JordanAlgebraSpace(VectorSpace):
    """
    Vector space capability with a Jordan product and spectral calculus.

    Parameters
    ----------
    ctx : Context, str, or None, optional
        Context specification used for elements and validation checks.
    check_level : {"none", "cheap", "standard", "strict"}, optional
        Validation level; the ambient default when omitted.
    """

    # Public Jordan operations: check, then call the core. Families implement the
    # abstract cores; spectral_apply, trace and determinant derive from them.

    @checked_method(in_space="self", out_space="self", arg_positions=(0, 1))
    def jordan(self, x: Any, y: Any) -> Any:
        """Return the Jordan product of ``x`` and ``y``."""
        return self._jordan_core(x, y)

    @checked_method(in_space="self")
    def spectrum(self, x: Any, **kwargs: Any) -> Any:
        """Return Jordan-algebraic eigenvalues of ``x``."""
        return self._spectrum_core(x, **kwargs)

    @checked_method(in_space="self")
    def spectral_decompose(self, x: Any) -> Any:
        """Return spectral data sufficient to reconstruct ``x``."""
        return self._spectral_decompose_core(x)

    @checked_method(out_space="self")
    def from_spectrum(self, eigvals: Any, frame: Any = None) -> Any:
        """Reconstruct an element from spectral data."""
        return self._from_spectrum_core(eigvals, frame)

    @checked_method(in_space="self", out_space="self")
    def spectral_apply(self, x: Any, f: Callable) -> Any:
        """Apply ``f`` to the spectrum of ``x`` and reconstruct."""
        return self._spectral_apply_core(x, f)

    @checked_method(in_space="self")
    def trace(self, x: Any) -> Any:
        """Return the Jordan trace ``sum_i lambda_i(x)`` of the spectrum."""
        return self._trace_core(x)

    @checked_method(in_space="self")
    def determinant(self, x: Any) -> Any:
        """Return the Jordan determinant ``prod_i lambda_i(x)`` of the spectrum."""
        return self._determinant_core(x)

    @abstractmethod
    def unit(self) -> Any:
        """Return the Jordan-algebraic identity element ``e`` (all-ones spectrum)."""

    @abstractmethod
    def _jordan_core(self, x: Any, y: Any) -> Any: ...

    @abstractmethod
    def _spectrum_core(self, x: Any) -> Any: ...

    @abstractmethod
    def _spectral_decompose_core(self, x: Any) -> Any: ...

    @abstractmethod
    def _from_spectrum_core(self, eigvals: Any, frame: Any) -> Any: ...

    def _spectral_apply_core(self, x: Any, f: Callable) -> Any:
        eigvals, frame = self._spectral_decompose_core(x)
        return self._from_spectrum_core(f(eigvals), frame)

    def _trace_core(self, x: Any) -> Any:
        return self.ops.sum(self._spectrum_core(x), axis=-1)

    def _determinant_core(self, x: Any) -> Any:
        return self.ops.prod(self._spectrum_core(x), axis=-1)


class EuclideanJordanAlgebraSpace(JordanAlgebraSpace, InnerProductSpace):
    """
    Jordan algebra capability with a compatible inner product.

    Parameters
    ----------
    ctx : Context, str, or None, optional
        Context specification used for elements and validation checks.
    check_level : {"none", "cheap", "standard", "strict"}, optional
        Validation level; the ambient default when omitted.
    """
