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
    Build ``scale * ||A x - b||^2`` as a :class:`~spacecore.functional.LinOpQuadraticForm`.

    It is ``1/2 <x, Q x> + <c, x> + a`` with ``Q = 2 scale A.H A``, ``c = -2 scale A.H b``
    and ``a = scale <b, b>``, all in the operator's own geometry. Expanding
    the squared norm and moving ``A`` by the adjoint identity gives this formula
    on real spaces; on complex spaces the cross term must be ``Re<c, x>``.
    The current linear term retains the full pairing, so the squared-norm
    interpretation requires real data. Trefethen & Bau, *Numerical Linear Algebra*,
    Lecture 2, pp. 11-12.

    Parameters
    ----------
    A : LinOp
        Forward operator ``X -> Y``.
    b : array-like
        Observation in ``A.codomain``.
    weights : array-like or None, optional
        Positive diagonal residual weights ``w``, giving ``scale <r, diag(w) r>``.
    scale : float, optional
        Positive factor; ``0.5`` gives the textbook ``1/2 ||A x - b||^2``.

    Returns
    -------
    LinOpQuadraticForm
        The objective; its gradient is ``2 scale A.H (W (A x - b))``.

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
    # Validate A and scale.
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

    # Unweighted: Gram A* A, right-hand side A* b, energy <b, b>.
    if weights is None:
        gram = A.H @ A
        rhs = A.H.apply(b)
        b_energy = Y.inner(b, b)
    # Weighted: the same with W = diag(w) inserted, after validating w.
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

    # Assemble 1/2 <x, Q x> + <c, x> + a with Q = 2s Gram, c = -2s rhs, a = s energy.
    Q = gram if two_scale == 1.0 else two_scale * gram
    linear = InnerProductFunctional((-two_scale) * rhs, X, ctx)
    offset = scale * ctx.ops.real(b_energy)
    return LinOpQuadraticForm(Q, linear, offset, ctx)
