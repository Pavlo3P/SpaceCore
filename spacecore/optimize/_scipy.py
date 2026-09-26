"""
SciPy optimizer adapters (ADR-018): ``scipy.optimize`` on flat real coordinates.

The gradient passed is ``X.riesz(F.grad(x))``. Structure is flattened away, so
``bounds`` and ``constraints`` act on coordinates. SciPy's own models are
Euclidean, and complex domains are rejected.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from ._common import (
    coordinate_gradient,
    domain_with_geometry,
    require_functional,
    require_real_field,
)


def _flat_host(X: Any, element: Any) -> np.ndarray:
    """Return ``element`` of ``X`` as a flat host ``float64`` array for SciPy."""
    return np.asarray(X.flatten(element), dtype=np.float64).reshape(-1)


def _reject_args(kw: dict, fname: str) -> None:
    """Reject SciPy's ``args`` parameter, which is meaningless for a Functional.

    SciPy splats ``args`` into every ``fun(x, *args)``/``jac(x, *args)`` call, but
    a SpaceCore ``Functional`` takes only its domain element. Rejecting here turns
    a cryptic ``takes 1 positional argument but 2 were given`` into an actionable
    error.
    """
    if "args" in kw:
        raise ValueError(
            f"{fname} does not support the SciPy 'args' parameter: a SpaceCore "
            "Functional takes only its domain element. Bake any extra parameters "
            "into F instead."
        )


def _make_value_and_grad(F: Any, X: Any):
    """Build SciPy ``fun``/``jac`` callables over flat host coordinate vectors."""
    ctx = X.ctx

    def fun(v: np.ndarray) -> float:
        x = X.unflatten(ctx.asarray(np.asarray(v)))
        return float(np.real(F.value(x)))

    def jac(v: np.ndarray) -> np.ndarray:
        x = X.unflatten(ctx.asarray(np.asarray(v)))
        return _flat_host(X, coordinate_gradient(F, X, x))

    return fun, jac


def minimize_scipy(
    F: Any,
    x0: Any,
    *,
    method: str = "L-BFGS-B",
    jac: Any = True,
    **kw: Any,
) -> Any:
    r"""
    Minimize a functional with :func:`scipy.optimize.minimize`.

    SciPy sees flat coordinates: the gradient handed over is ``X.riesz(F.grad(x))``,
    which is the identity on a Euclidean space and required on a weighted one.

    Parameters
    ----------
    F : Functional
        Objective on a real inner-product space ``X = F.domain``.
    x0 : array-like
        Initial guess in ``F.domain``.
    method : str, optional
        SciPy method; ``"L-BFGS-B"`` by default.
    jac : bool, str, callable, or None, optional
        ``True`` supplies the coordinate gradient; ``False``/``None`` or a real
        finite-difference string lets SciPy approximate it; a callable is forwarded.
        ``"cs"`` is rejected.
    **kw
        Forwarded to SciPy; ``bounds`` and ``constraints`` act on flat coordinates.
        ``args`` is not supported.

    Returns
    -------
    scipy.optimize.OptimizeResult
        SciPy's result, plus ``x_element``: the minimizer as an element of ``F.domain``.

    Raises
    ------
    TypeError
        If ``F`` is not a :class:`~spacecore.Functional` or its domain has no inner product.
    ValueError
        If the domain is complex, ``jac="cs"``, or ``args`` is passed.

    See Also
    --------
    line_search_scipy : Wolfe line search with the same gradient handoff.

    Examples
    --------
    Minimize ``f(x) = 1/2 x^T diag(3, 1) x - 3 x_0 - 2 x_1`` on a Euclidean
    space; the minimizer is ``(1, 2)``.

    >>> import numpy as np
    >>> import spacecore as sc
    >>> ctx = sc.Context(sc.NumpyOps(), dtype=np.float64)
    >>> X = sc.DenseCoordinateSpace((2,), ctx)
    >>> Q = sc.DenseLinOp(ctx.asarray([[3.0, 0.0], [0.0, 1.0]]), X, X, ctx)
    >>> linear = sc.InnerProductFunctional(ctx.asarray([-3.0, -2.0]), X)
    >>> F = sc.LinOpQuadraticForm(Q, linear)
    >>> result = sc.minimize_scipy(F, X.zeros(), options={"gtol": 1e-10})
    >>> bool(result.success)
    True
    >>> np.allclose(np.asarray(result.x_element), [1.0, 2.0])
    True
    """
    import scipy.optimize as sopt

    F = require_functional(F, "minimize_scipy")
    X = domain_with_geometry(F, "minimize_scipy")
    require_real_field(X, "minimize_scipy")
    _reject_args(kw, "minimize_scipy")

    fun, riesz_jac = _make_value_and_grad(F, X)
    if jac is True:
        jac_arg: Any = riesz_jac
    elif jac is False or jac is None:
        jac_arg = None
    elif jac == "cs":
        raise ValueError(
            "minimize_scipy does not support jac='cs' (complex-step) on a real "
            "domain: SciPy evaluates the objective at complex perturbations the "
            "real backend rejects. Use the default jac=True (analytic "
            "X.riesz(F.grad(x))) or a real finite-difference scheme "
            "(jac='2-point' or '3-point')."
        )
    else:
        jac_arg = jac

    x0_flat = _flat_host(X, x0)
    result = sopt.minimize(fun, x0_flat, method=method, jac=jac_arg, **kw)
    result.x_element = X.unflatten(X.ctx.asarray(np.asarray(result.x)))
    return result


def line_search_scipy(F: Any, x: Any, d: Any, **kw: Any) -> Any:
    r"""
    Wolfe line search along ``d`` with :func:`scipy.optimize.line_search`.

    The coordinate gradient ``X.riesz(F.grad(x))`` makes SciPy's slope equal the
    directional derivative on any geometry.

    Parameters
    ----------
    F : Functional
        Objective on a real inner-product space ``X = F.domain``.
    x : array-like
        Current iterate in ``F.domain``.
    d : array-like
        Search direction; the new iterate is ``x + alpha * d``.
    **kw
        Forwarded to SciPy (``gfk``, ``c1``, ``c2``, ``amax``, ...). ``args`` is not supported.

    Returns
    -------
    tuple
        SciPy's ``(alpha, fc, gc, new_fval, old_fval, new_slope)``; ``alpha`` is ``None``
        when the Wolfe conditions are not met.

    Raises
    ------
    TypeError
        If ``F`` is not a :class:`~spacecore.Functional` or its domain has no inner product.
    ValueError
        If the domain is complex or ``args`` is passed.

    Examples
    --------
    A descent step on a convex quadratic returns a positive step length.

    >>> import numpy as np
    >>> import spacecore as sc
    >>> ctx = sc.Context(sc.NumpyOps(), dtype=np.float64)
    >>> X = sc.DenseCoordinateSpace((2,), ctx)
    >>> Q = sc.DenseLinOp(ctx.asarray([[3.0, 0.0], [0.0, 1.0]]), X, X, ctx)
    >>> F = sc.LinOpQuadraticForm(Q)
    >>> x = ctx.asarray([1.0, 1.0])
    >>> d = X.scale(-1.0, F.grad(x))
    >>> alpha = sc.line_search_scipy(F, x, d)[0]
    >>> bool(alpha > 0.0)
    True
    """
    import scipy.optimize as sopt

    F = require_functional(F, "line_search_scipy")
    X = domain_with_geometry(F, "line_search_scipy")
    require_real_field(X, "line_search_scipy")
    _reject_args(kw, "line_search_scipy")

    fun, jac = _make_value_and_grad(F, X)
    xk = _flat_host(X, x)
    pk = _flat_host(X, d)
    return sopt.line_search(fun, jac, xk, pk, **kw)
