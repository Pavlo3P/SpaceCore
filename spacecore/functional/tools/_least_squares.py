"""The :func:`least_squares` constructor (ADR-019).

The ``0.4.0`` ergonomics study found every least-squares persona hand-expanding
``1/2 ||A x - b||^2`` into ``Q = A.H @ A`` and ``c = -A^# b`` to fit a
:class:`~spacecore.functional.LinOpQuadraticForm`. This module provides that
expansion as a single front door. It returns an existing
``LinOpQuadraticForm`` -- no new type -- so the whole quadratic toolbox
(gradient, Hessian action, batched evaluation, composition) applies unchanged.
"""
from __future__ import annotations

import numpy as _np
from typing import Any

from .._linear import InnerProductFunctional
from .._quadratic import LinOpQuadraticForm
from ...linop import DiagonalLinOp, LinOp


def least_squares(
    A: LinOp,
    b: Any,
    *,
    weights: Any = None,
    scale: float = 0.5,
) -> LinOpQuadraticForm:
    r"""
    Build the least-squares objective ``scale * ||A x - b||^2`` as a quadratic form.

    The objective is expanded into the canonical
    :class:`~spacecore.functional.LinOpQuadraticForm`
    ``q(x) = 1/2 <x, Q x>_X + <c, x>_X + a`` with

    * ``Q = 2 scale * (A.H @ A)`` (the normal operator, Hermitian by construction),
    * ``c = -2 scale * A.H(b)`` (a Riesz/metric gradient, the linear term), and
    * ``a = scale * <b, b>_Y`` (the constant residual energy),

    so that ``q(x)`` equals ``scale * ||A x - b||_Y^2`` exactly on real spaces.
    With ``scale = 0.5`` (the default) this is the textbook ``1/2 ||A x - b||^2``
    and ``Q`` is exactly ``A.H @ A``.

    Adjoints and inner products are taken in the operator's declared geometry
    (ADR-009), so the result is metric-correct on non-Euclidean domains and
    codomains without any extra work by the caller.

    Parameters
    ----------
    A : LinOp
        Forward operator ``A : X -> Y``.
    b : array-like
        Observation in the codomain ``Y = A.codomain``.
    weights : array-like or None, optional
        Diagonal residual weights ``w`` with the codomain shape and strictly
        positive, finite entries. When given, the objective is the weighted
        least squares ``scale * <A x - b, W (A x - b)>_Y`` with ``W = diag(w)``.
        Default ``None`` is the unweighted objective.
    scale : float, optional
        Positive scalar multiplying the squared residual. Default ``0.5``.

    Returns
    -------
    LinOpQuadraticForm
        Quadratic form whose ``value`` is the (weighted) least-squares objective
        and whose ``grad`` is the metric gradient ``2 scale * A.H(W (A x - b))``.

    Examples
    --------
    >>> import numpy as np
    >>> import spacecore as sc
    >>> ctx = sc.Context(sc.NumpyOps(), dtype=np.float64)
    >>> X = sc.DenseCoordinateSpace((2,), ctx)
    >>> Y = sc.DenseCoordinateSpace((2,), ctx)
    >>> A = sc.DenseLinOp(ctx.asarray([[1.0, 0.0], [0.0, 2.0]]), X, Y, ctx)
    >>> f = sc.least_squares(A, ctx.asarray([1.0, 4.0]))
    >>> float(f.value(ctx.asarray([1.0, 2.0])))
    0.0
    """
    if not isinstance(A, LinOp):
        raise TypeError(f"least_squares requires A to be a LinOp, got {type(A).__name__}.")

    scale = float(scale)
    if not _np.isfinite(scale):
        raise ValueError(f"least_squares requires a finite scale, got {scale}.")

    X = A.domain
    Y = A.codomain
    ctx = A.ctx
    b = ctx.asarray(b)
    two_scale = 2.0 * scale

    if weights is None:
        gram = A.H @ A
        rhs = A.H.apply(b)
        b_energy = Y.inner(b, b)
    else:
        w = ctx.asarray(weights)
        w_np = _np.asarray(w)
        if tuple(w_np.shape) != tuple(Y.shape):
            raise ValueError(
                f"least_squares weights must have the codomain shape {tuple(Y.shape)}, "
                f"got {tuple(w_np.shape)}."
            )
        if not _np.all(_np.isfinite(w_np)) or not _np.all(w_np > 0):
            raise ValueError("least_squares weights must be strictly positive and finite.")
        W = DiagonalLinOp(w, Y, ctx)
        Wb = W.apply(b)
        gram = A.H @ W @ A
        rhs = A.H.apply(Wb)
        b_energy = Y.inner(b, Wb)

    Q = gram if two_scale == 1.0 else two_scale * gram
    linear = InnerProductFunctional((-two_scale) * rhs, X, ctx)
    offset = scale * ctx.ops.real(b_energy)
    return LinOpQuadraticForm(Q, linear, offset, ctx)
