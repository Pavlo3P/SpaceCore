"""
Base for coordinate-defined functionals: value plus Euclidean coordinate gradient.

:class:`_CoordinateFunctional` turns the coordinate gradient into the Riesz gradient
``domain.riesz_inverse(d phi / d x)`` once. That is the one guard against pairing a
Euclidean gradient with a non-Euclidean metric (ADR-010, ADR-019).
Beck, *First-Order Methods in Optimization*, Remark 3.32; for complex real-valued
objectives use ``DF[h] = Re(g^H R h)`` to obtain the same ``R^{-1}`` conversion.
"""
from __future__ import annotations

from typing import Any, cast

from .._base import Domain, Functional
from ..._batching import _leading_batch_size, _warn_vmap_fallback_once


def lp_value(ops: Any, vec: Any, p: float) -> Any:
    """Return the coordinate ``p``-norm ``(sum_i |vec_i|^p)^{1/p}``."""
    return ops.sum(ops.abs(vec) ** p) ** (1.0 / p)


def lp_coordinate_grad(ops: Any, vec: Any, p: float) -> Any:
    """Return the Euclidean gradient of the ``p``-norm (zero subgradient at 0).

    The gradient is ``sign(v_i) |v_i|^{p-1} / ||v||_p^{p-1}``. The ``safe``
    denominator guard turns the ``0 / 0`` at the origin into the conventional
    zero subgradient: there ``|v_i|^{p-1}`` already vanishes for ``p > 1`` and
    ``sign`` vanishes for ``p == 1``.
    """
    # d/dx_i ||x||_p = sign(x_i) |x_i|^(p-1) / ||x||_p^(p-1); zero at the origin.
    r = ops.abs(vec)
    norm = ops.sum(r**p) ** (1.0 / p)
    safe = ops.where(norm > 0, norm, cast(Any, 1.0))
    return ops.sign(vec) * (r ** (p - 1.0)) * (safe ** (1.0 - p))


class _CoordinateFunctional(Functional[Domain]):
    r"""
    Base for functionals defined by a coordinatewise scalar formula.

    Subclasses implement the scalar :meth:`value` and the Euclidean coordinate
    gradient :meth:`_coordinate_grad` (the array of partials ``d phi / d x_i``).
    This base Riesz-corrects that coordinate gradient with
    ``domain.riesz_inverse`` so :meth:`grad` is the metric gradient required by
    ADR-010 on any geometry (identity on a Euclidean space, division by the
    weights on a diagonal metric, ``G^{-1}`` in general).
    """

    def _grad_core(self, x: Any) -> Any:
        """The Riesz gradient under the domain geometry."""
        return cast(Any, self.domain)._riesz_inverse_core(self._coordinate_grad(x))

    def _vgrad_core(self, xs: Any) -> Any:
        """The gradient core evaluated independently over a leading batch axis."""
        _warn_vmap_fallback_once(self, "vgrad", _leading_batch_size(self.domain, xs))
        return self.ops.vmap(self._grad_core, in_axes=0, out_axes=0)(xs)

    def _differential_core(self, x: Any) -> Any:
        """The coordinate partials directly; they are the differential covector."""
        return self._coordinate_grad(x)

    def _coordinate_grad(self, x: Any) -> Any:
        """Return the Euclidean coordinate gradient ``d phi / d x_i``."""
        raise NotImplementedError(
            f"{type(self).__name__} does not define _coordinate_grad."
        )
