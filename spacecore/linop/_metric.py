from __future__ import annotations

import warnings

from ..space._base import Space
from ..space._inner import InnerProduct


_METRIC_BATCH_FALLBACK_ERRORS = (TypeError, ValueError, NotImplementedError)

# Metric Hermiticity checks apply the operator to every coordinate basis vector.
# Above this size, return "unknown" instead of doing O(n) work implicitly.
_METRIC_HERMITIAN_BASIS_CHECK_MAX_SIZE = 1024


def space_has_riesz_maps(space) -> bool:
    """Return whether ``space`` exposes usable Riesz maps for metric adjoints."""
    parts = getattr(space, "spaces", None)
    if parts is not None:
        return all(part.is_euclidean or space_has_riesz_maps(part) for part in parts)

    geometry_type = type(space.geometry)
    geometry_has_maps = (
        geometry_type.riesz is not InnerProduct.riesz
        and geometry_type.riesz_inverse is not InnerProduct.riesz_inverse
    )
    space_has_maps = (
        type(space).riesz is not Space.riesz
        and type(space).riesz_inverse is not Space.riesz_inverse
    )
    return geometry_has_maps or space_has_maps


def _requires_euclidean_or_riesz(dom, cod, opname: str) -> None:
    """Reject non-Euclidean spaces that cannot define metric adjoints."""
    for space, role in ((dom, "domain"), (cod, "codomain")):
        if space.is_euclidean or space_has_riesz_maps(space):
            continue
        raise TypeError(
            f"{opname} on non-Euclidean {role} {type(space).__name__} "
            "requires Riesz maps via its Space methods or InnerProduct "
            "riesz/riesz_inverse. Use MatrixFreeLinOp with an explicit "
            "adjoint instead."
        )


def _metric_is_hermitian_by_basis(op) -> bool | None:
    """Check self-adjointness by comparing forward and adjoint basis actions."""
    if op.domain != op.codomain:
        return False
    if op.domain.size > _METRIC_HERMITIAN_BASIS_CHECK_MAX_SIZE:
        return None
    try:
        size = op.domain.size
        eye = op.ops.eye(size, dtype=op.dtype)
        apply_cols = []
        adjoint_cols = []
        for i in range(size):
            e_i = op.domain.unflatten(eye[:, i])
            apply_cols.append(op.domain.flatten(op.apply(e_i)))
            adjoint_cols.append(op.domain.flatten(op.rapply(e_i)))
        A = op.ops.stack(tuple(apply_cols), axis=1)
        A_sharp = op.ops.stack(tuple(adjoint_cols), axis=1)
        return bool(op.ops.allclose(A, A_sharp))
    except Exception:
        return None


def _warn_metric_batch_fallback(opname: str, error: Exception) -> None:
    """Warn that batched Riesz maps were unavailable and vmap fallback is used."""
    warnings.warn(
        f"{opname}.rvapply() could not use batched Riesz maps and is falling "
        f"back to vmap(self.rapply). Original error: "
        f"{type(error).__name__}: {error}",
        RuntimeWarning,
        stacklevel=2,
    )


def metric_rapply(domain, codomain, euclidean_rapply, y):
    """Apply the metric adjoint ``R_X^{-1} A^dagger R_Y`` to one element."""
    if domain.is_euclidean and codomain.is_euclidean:
        return euclidean_rapply(y)
    return domain.riesz_inverse(euclidean_rapply(codomain.riesz(y)))


def metric_rvapply(
    domain,
    codomain,
    euclidean_rapply,
    euclidean_rvapply,
    ys,
    *,
    opname: str,
    ops,
):
    """Apply the metric adjoint over a leading batch axis.

    The fast path uses batched Riesz maps, which should broadcast over the
    leading batch axis. If a space does not support batched Riesz maps, this
    falls back to backend ``vmap`` over :func:`metric_rapply` and emits a
    runtime warning.
    """
    if domain.is_euclidean and codomain.is_euclidean:
        return euclidean_rvapply(ys)
    try:
        yd = codomain.riesz(ys)
        tmp = euclidean_rvapply(yd)
        return domain.riesz_inverse(tmp)
    except _METRIC_BATCH_FALLBACK_ERRORS as err:
        _warn_metric_batch_fallback(opname, err)
        per_elem = lambda y: metric_rapply(domain, codomain, euclidean_rapply, y)
        return ops.vmap(per_elem, in_axes=0, out_axes=0)(ys)
