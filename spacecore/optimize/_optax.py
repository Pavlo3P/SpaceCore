"""optax optimizer adapter (ADR-018).

Drives an `optax <https://optax.readthedocs.io>`_ gradient-transformation
optimizer from a SpaceCore :class:`~spacecore.Functional`. optax operates on JAX
pytrees, and SpaceCore elements already *are* pytrees (a dense array is a leaf; a
``TreeSpace`` element is a registered pytree), so the adapter passes the element
representation straight through with no flatten/unflatten marshalling. As with
the SciPy adapters, the gradient handed to optax is the coordinate gradient
``X.riesz(F.grad(x))`` -- optax applies coordinate updates
(``params - lr * grad``), so the metric-to-coordinate conversion is mandatory on
a weighted space and the identity on a Euclidean one.

``optax`` is an optional dependency; it is imported lazily so that
``import spacecore`` (and doctest collection) does not require it.

Information lost at the optax boundary
--------------------------------------
* **Geometry.** optax updates are taken in the flat coordinate metric; the
  domain geometry survives only through the ``X.riesz`` gradient conversion.
* **Representation.** ``x0`` may be a raw array or a tuple/list/dict of arrays;
  a bound ``TreeElement`` is normalized to its raw pytree at entry so that
  ``optax.apply_updates`` matches it against the converted gradient's structure.
* **Field.** Unlike the SciPy adapters, ``minimize_optax`` does not reject a
  complex domain -- optax's updates are complex-capable -- so the correctness of
  complex-valued descent is the caller's responsibility.
* **Tracing.** Eager evaluation is correct but unfused. To ``jax.jit`` the step,
  wrap it yourself; jitted external solves may require a domain context built with
  ``check_level="none"`` (ADR-018, ergonomics run-02 S14).
"""
from __future__ import annotations

from typing import Any, Callable

from ..space import TreeSpace
from ._common import coordinate_gradient, domain_with_geometry, require_functional


def minimize_optax(
    F: Any,
    x0: Any,
    optimizer: Any,
    *,
    steps: int,
    callback: Callable[[int, Any], None] | None = None,
) -> Any:
    r"""
    Run an optax optimizer on a SpaceCore functional with pytree pass-through.

    Executes the canonical optax update loop for ``steps`` iterations::

        grad = X.riesz(F.grad(params))            # metric -> coordinate gradient
        updates, opt_state = optimizer.update(grad, opt_state, params)
        params = optax.apply_updates(params, updates)

    The optax optimizer owns the update rule (learning rate, momentum,
    preconditioning); this adapter only supplies the coordinate gradient and runs
    the fixed-length loop. There is no convergence test -- optax is a
    gradient-transformation library, so iteration count is the caller's contract.

    Parameters
    ----------
    F : Functional
        Objective with an inner-product domain ``X = F.domain`` and an
        implemented ``F.grad``.
    x0 : pytree
        Initial parameters, an element of ``F.domain``. A raw array or
        tuple/list/dict of arrays is used as-is; a bound ``TreeElement`` is
        normalized to its raw pytree.
    optimizer : optax.GradientTransformation
        An optax optimizer such as ``optax.adam(1e-2)``.
    steps : int
        Number of update steps to run. Must be non-negative.
    callback : callable, optional
        Called as ``callback(step, params)`` after each update, e.g. to record a
        loss trajectory.

    Returns
    -------
    pytree
        The final parameters, an element of ``F.domain``.

    Raises
    ------
    TypeError
        If ``F`` is not a :class:`~spacecore.Functional`, its domain has no
        inner-product geometry, or its domain is not JAX-backed (optax produces
        JAX arrays).
    ValueError
        If ``steps`` is negative.
    ImportError
        If ``optax`` is not installed (``pip install spacecore[optax]``).

    See Also
    --------
    spacecore.optimize.minimize_scipy : SciPy ``minimize`` with the same handoff.

    Notes
    -----
    On a Euclidean space ``X.riesz`` is the identity. On a weighted space it
    multiplies the metric gradient by the diagonal weights, which is exactly the
    coordinate gradient optax's Euclidean update expects (ADR-018).

    Examples
    --------
    .. code-block:: python

        import numpy as np
        import optax
        import spacecore as sc

        ctx = sc.Context(sc.JaxOps(), dtype=np.float32, check_level="none")
        X = sc.DenseCoordinateSpace((2,), ctx)
        Q = sc.DenseLinOp(ctx.asarray([[3.0, 0.0], [0.0, 1.0]]), X, X, ctx)
        linear = sc.InnerProductFunctional(ctx.asarray([-3.0, -2.0]), X)
        F = sc.LinOpQuadraticForm(Q, linear)

        x_star = sc.minimize_optax(F, X.zeros(), optax.adam(1e-1), steps=500)
        # x_star is approximately (1.0, 2.0)
    """
    F = require_functional(F, "minimize_optax")
    X = domain_with_geometry(F, "minimize_optax")
    if getattr(X.ops, "family", None) != "jax":
        raise TypeError(
            "minimize_optax requires a JAX-backed domain (optax produces JAX "
            f"arrays via optax.apply_updates); F.domain uses the "
            f"{getattr(X.ops, 'family', '?')!r} backend. Build the domain with a "
            "JaxOps context, or use minimize_scipy for the NumPy backend."
        )
    steps = int(steps)
    if steps < 0:
        raise ValueError(f"steps must be non-negative, got {steps}.")

    try:
        import optax
    except ImportError as exc:  # pragma: no cover - exercised only without optax
        raise ImportError(
            "minimize_optax requires the optional 'optax' dependency; "
            "install it with `pip install spacecore[optax]`."
        ) from exc

    # Normalize a tree element to the raw pytree representation that
    # ``X.riesz`` (and thus the gradient) returns, so ``optax.apply_updates``
    # sees one consistent pytree structure across params, grad, and updates. A
    # bound ``TreeElement`` is its own registered pytree and would otherwise
    # collide with the raw tuple/dict the gradient carries.
    params = X.unflatten_tree(X.flatten_tree(x0)) if isinstance(X, TreeSpace) else x0
    opt_state = optimizer.init(params)
    for step in range(steps):
        grad = coordinate_gradient(F, X, params)
        updates, opt_state = optimizer.update(grad, opt_state, params)
        params = optax.apply_updates(params, updates)
        if callback is not None:
            callback(step, params)
    return params
