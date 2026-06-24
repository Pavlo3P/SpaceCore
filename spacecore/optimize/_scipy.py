"""SciPy optimizer adapters (ADR-018).

Thin functions that drive ``scipy.optimize`` from a SpaceCore
:class:`~spacecore.Functional`. They marshal between SpaceCore elements and
SciPy's flat real coordinate arrays, evaluate the objective through ``F.value``,
and -- crucially -- convert the metric gradient ``F.grad`` to a coordinate
gradient with ``X.riesz`` before handing it to SciPy (see
:func:`spacecore.optimize._common.coordinate_gradient`). The external optimizer
owns the loop, line search, and convergence; these adapters only translate the
objective and its geometry.

Information lost at the SciPy boundary
--------------------------------------
* **Structure.** Elements are flattened to a 1-D coordinate vector with
  ``X.flatten``/``X.unflatten``; structured (tree/block) layout is not visible to
  SciPy. Any ``bounds`` or ``constraints`` passed through ``**kw`` are therefore
  expressed in *flattened coordinates*, not structured elements.
* **Geometry.** SciPy optimizes in the flat Euclidean coordinate metric. The
  domain inner-product geometry survives only through the ``X.riesz`` gradient
  conversion; SciPy's own trust region / quasi-Newton model is Euclidean.
* **Field.** SciPy works on real vectors, so a complex domain is rejected.
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
    Minimize a SpaceCore functional with :func:`scipy.optimize.minimize`.

    The objective is evaluated through ``F.value`` and the Jacobian through
    ``X.riesz(F.grad(x))`` -- the metric-to-coordinate gradient handoff that is
    the identity on a Euclidean space and mandatory on a weighted one. SpaceCore
    elements are flattened to and from SciPy's flat coordinate array with
    ``F.domain.flatten``/``unflatten``.

    Parameters
    ----------
    F : Functional
        Objective with a real, inner-product domain ``X = F.domain``. ``F.grad``
        must be implemented unless a gradient-free ``jac`` is selected.
    x0 : array-like
        Initial guess, an element of ``F.domain``.
    method : str, optional
        SciPy ``minimize`` method. Default ``"L-BFGS-B"``.
    jac : bool, str, callable, or None, optional
        Gradient policy. ``True`` (default) supplies SpaceCore's
        ``X.riesz(F.grad(x))`` coordinate gradient. ``False`` or ``None`` lets
        SciPy approximate the gradient by finite differences. A real
        finite-difference string (``"2-point"`` or ``"3-point"``) or a callable
        over the flat coordinate vector is forwarded to SciPy unchanged. The
        complex-step mode ``"cs"`` is rejected because it evaluates the objective
        at complex perturbations the real domain cannot represent.
    **kw
        Forwarded to :func:`scipy.optimize.minimize` (e.g. ``bounds``,
        ``constraints``, ``tol``, ``options``). ``bounds`` and ``constraints``
        are interpreted in *flattened coordinates*. The SciPy ``args`` parameter
        is not supported (a ``Functional`` takes only its domain element).

    Returns
    -------
    scipy.optimize.OptimizeResult
        The SciPy result, unchanged, with one added field ``x_element``: the
        minimizer ``result.x`` unflattened back into an element of ``F.domain``.
        ``result.x`` stays the flat coordinate array, by SciPy convention.

    Raises
    ------
    TypeError
        If ``F`` is not a :class:`~spacecore.Functional` or its domain has no
        inner-product geometry.
    ValueError
        If ``F.domain`` is over the complex field, if ``jac="cs"``, or if the
        unsupported SciPy ``args`` parameter is passed.

    See Also
    --------
    line_search_scipy : Wolfe line search along a direction with the same handoff.
    spacecore.optimize.minimize_optax : optax loop with pytree pass-through.

    Notes
    -----
    The external optimizer owns iteration, line search, and convergence; this
    adapter only translates the objective and its geometry. The minimizer
    location of a smooth problem does not depend on the metric (a metric gradient
    and the coordinate gradient vanish together), but SciPy's *steps* do: feeding
    SciPy the raw metric gradient ``F.grad`` on a weighted space is inconsistent
    with ``fun`` and degrades or breaks convergence. ``X.riesz`` removes that
    trap.

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

    Evaluates ``F.value`` and the coordinate gradient ``X.riesz(F.grad(x))`` over
    the flat coordinate representation, so SciPy's slope
    ``coordinate_gradient . d`` is the correct directional derivative
    ``df(d) = <F.grad(x), d>_X`` even on a weighted space.

    Parameters
    ----------
    F : Functional
        Objective with a real, inner-product domain ``X = F.domain``.
    x : array-like
        Current iterate, an element of ``F.domain``.
    d : array-like
        Search direction, an element of ``F.domain`` interpreted as a coordinate
        displacement (the new iterate is ``x + alpha * d``). The adapter does not
        transform ``d``: both ``-F.grad(x)`` (natural/metric steepest descent) and
        ``-X.riesz(F.grad(x))`` (coordinate steepest descent) are valid descent
        directions.
    **kw
        Forwarded to :func:`scipy.optimize.line_search` (e.g. ``gfk``,
        ``old_fval``, ``c1``, ``c2``, ``amax``). The SciPy ``args`` parameter is
        not supported (a ``Functional`` takes only its domain element).

    Returns
    -------
    tuple
        The SciPy ``line_search`` tuple
        ``(alpha, fc, gc, new_fval, old_fval, new_slope)``. ``alpha`` is the step
        length and is ``None`` when the line search fails to satisfy the Wolfe
        conditions.

    Raises
    ------
    TypeError
        If ``F`` is not a :class:`~spacecore.Functional` or its domain has no
        inner-product geometry.
    ValueError
        If ``F.domain`` is over the complex field, or if the unsupported SciPy
        ``args`` parameter is passed.

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
