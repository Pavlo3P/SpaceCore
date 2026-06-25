"""Shared base and helpers for coordinate-defined toolbox functionals.

A coordinate functional ``F(x) = phi(x)`` is specified by its scalar value and
its *Euclidean coordinate gradient* ``d phi / d x_i``. The metric-correct Riesz
gradient required by [ADR-010](010_functional_contract.md) is obtained once, in
:class:`_CoordinateFunctional`, by Riesz-correcting that coordinate gradient with
``domain.riesz_inverse``. Centralizing the correction is the single defense
against the [ADR-019](019_everyday_toolbox.md) trap of pairing a Euclidean
gradient with a non-Euclidean metric.
"""
from __future__ import annotations

from typing import Any, cast

from .._base import Domain, Functional
from ..._batching import _leading_batch_size, _warn_vmap_fallback_once
from ..._checks import checked_method


def _inner_core(space: Any, x: Any, y: Any) -> Any:
    """Return ``<x, y>`` via the space's check-free inner product when present."""
    inner = getattr(space, "_inner_core", space.inner)
    return inner(x, y)


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

    @checked_method(in_space="domain", out_space="domain")
    def grad(self, x: Any) -> Any:
        """Return the Riesz gradient under the domain geometry."""
        return cast(Any, self.domain).riesz_inverse(self._coordinate_grad(x))

    @checked_method(in_space="domain", out_space="domain", in_batched=True, out_batched=True)
    def vgrad(self, xs: Any) -> Any:
        """Evaluate :meth:`grad` independently over a leading batch axis."""
        _warn_vmap_fallback_once(self, "vgrad", _leading_batch_size(self.domain, xs))
        return self.ops.vmap(self.grad, in_axes=0, out_axes=0)(xs)

    def _coordinate_grad(self, x: Any) -> Any:
        """Return the Euclidean coordinate gradient ``d phi / d x_i``."""
        raise NotImplementedError(
            f"{type(self).__name__} does not define _coordinate_grad."
        )
