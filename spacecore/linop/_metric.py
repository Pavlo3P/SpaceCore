from __future__ import annotations

from ..space import DenseCoordinateSpace, DenseVectorSpace, ElementwiseJordanSpace
from ..space.base import CoordinateSpace, InnerProductSpace
from ..space._capabilities import require


_VECTOR_SPACES = (DenseCoordinateSpace, DenseVectorSpace, ElementwiseJordanSpace)

# Metric Hermiticity checks apply the operator to every coordinate basis vector.
# Above this size, return "unknown" instead of doing O(n) work implicitly.
_METRIC_HERMITIAN_BASIS_CHECK_MAX_SIZE = 1024


def space_has_riesz_maps(space) -> bool:
    """Return whether ``space`` exposes usable Riesz maps for metric adjoints."""
    # Tree spaces: every leaf must be Euclidean or have Riesz maps.
    leaves = getattr(space, "leaf_spaces", None)
    if leaves is not None:
        return all(leaf.is_euclidean or space_has_riesz_maps(leaf) for leaf in leaves)

    # Riesz maps exist if the space overrides them or its geometry has them.
    return isinstance(space, InnerProductSpace) and (
        type(space).riesz is not InnerProductSpace.riesz or space.geometry.has_riesz_maps
    )


def _requires_euclidean_or_riesz(dom, cod, opname: str) -> None:
    """Reject non-Euclidean spaces that cannot define metric adjoints."""
    # Each endpoint must be Euclidean or provide Riesz maps.
    for space, role in ((dom, "domain"), (cod, "codomain")):
        if space_has_riesz_maps(space):
            continue
        raise TypeError(
            f"{opname} on non-Euclidean {role} {type(space).__name__} "
            "requires Riesz maps via its Space methods or InnerProduct "
            "riesz/riesz_inverse. Use MatrixFreeLinOp with an explicit "
            "adjoint instead."
        )


def _metric_is_hermitian_by_basis(op) -> bool | None:
    """Check self-adjointness by comparing forward and adjoint basis actions."""
    # Small square coordinate operators: compare A with A# column by column.
    if not isinstance(op.domain, CoordinateSpace):
        return None
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


def metric_rapply(domain, codomain, euclidean_rapply, y):
    r"""Apply the metric adjoint ``R_X^{-1} A^dagger R_Y`` to one element.

    Write ``R_X``, ``R_Y`` for the positive-definite Gram matrices and ``A^dagger``
    for the conjugate transpose (Trefethen & Bau, *Numerical Linear Algebra*,
    Lecture 2, pp. 11-12). The defining identity
    ``<Ax, y>_Y = <x, A^# y>_X`` expands to ``(Ax)^dagger R_Y y = x^dagger R_X A^# y``
    for all ``x``, hence ``A^dagger R_Y = R_X A^#`` and
    ``A^# = R_X^{-1} A^dagger R_Y``. The Euclidean short-circuit below is the case
    ``R_X = R_Y = I``, where the metric adjoint is the conjugate transpose.

    Using the coordinate adjoint without the Gram factors can pass Euclidean tests
    while failing the identity on weighted spaces.
    """
    require(domain, InnerProductSpace, "metric_rapply")
    require(codomain, InnerProductSpace, "metric_rapply")
    # Euclidean fast path; otherwise A# y = R_X^-1 A^T R_Y y.
    if domain.is_euclidean and codomain.is_euclidean:
        return euclidean_rapply(y)
    return domain.riesz_inverse(euclidean_rapply(codomain.riesz(y)))


def metric_rvapply(domain, codomain, euclidean_rvapply, ys):
    """Apply the metric adjoint over a leading batch axis; Riesz maps broadcast over it."""
    if domain.is_euclidean and codomain.is_euclidean:
        return euclidean_rvapply(ys)
    return domain.riesz_inverse(euclidean_rvapply(codomain.riesz(ys)))
