"""
Compiled optax driver for SpaceCore functionals (ADR-018).

The loop runs in one ``jax.jit(jax.lax.while_loop(...))`` on the coordinate gradient
``X.riesz(F.grad(x))``. ``optax`` is imported lazily. Build the domain with
``check_level="none"`` so traced evaluations skip data-dependent checks.
"""
from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Any, Callable, NamedTuple

import numpy as np

from ..space import TreeSpace
from ._common import domain_with_geometry, require_functional

_ROW_FMT = "{i:>7d}  {v:>+18.8e}  {d:>+18.8e}  {g:>18.3e}"
_HEADER = "{:>7}  {:>18}  {:>18}  {:>18}".format("iter", "value", "delta", "grad_norm")
_RULE = "-" * len(_HEADER)


class _OptaxState(NamedTuple):
    """optax state plus statistics cached at ``params`` (computed once/iteration)."""

    params: Any
    opt_state: Any
    value: Any
    grad: Any  # coordinate gradient X.riesz(F.grad)
    grad_norm: Any


class _History(NamedTuple):
    """Preallocated on-device history buffers, sliced to host after the loop."""

    iteration: Any
    value: Any
    value_delta: Any
    grad_norm: Any


class _LoopState(NamedTuple):
    """``while_loop`` carry: counter, optimizer state, finiteness, history, ls-count."""

    iteration: Any
    state: _OptaxState
    finite: Any
    history: _History
    history_index: Any
    nls: Any  # cumulative line-search steps


def _linesearch_steps(opt_state: Any) -> Any:
    """
    Sum every ``num_linesearch_steps`` counter in an optax state, or ``None``.

    Line-search optimizers expose ``num_linesearch_steps`` as *per-update*
    line-search info in their state (backtracking reports ``search_state.iter_num``,
    zoom reports ``final_state.count`` for the current round); gradient-transformation
    optimizers (adam, sgd, ...) do not. The search is structural (walks NamedTuple
    fields / tuples / dicts), and *every* counter found is summed, so a chain of
    several line-search transforms is counted in full rather than only the first.
    Returns ``None`` when no counter is present.
    """
    # Walk the nested optax state; sum every num_linesearch_steps field found.
    found = []
    stack = [opt_state]
    while stack:
        node = stack.pop()
        fields = getattr(node, "_fields", None)
        if fields is not None:  # NamedTuple
            if "num_linesearch_steps" in fields:
                found.append(node.num_linesearch_steps)
            stack.extend(getattr(node, f) for f in fields)
        elif isinstance(node, (tuple, list)):
            stack.extend(node)
        elif isinstance(node, dict):
            stack.extend(node.values())
    if not found:
        return None
    total = found[0]
    for extra in found[1:]:
        total = total + extra
    return total


def _tree_l2_norm(tree: Any) -> Any:
    """
    Euclidean L2 norm of a gradient pytree, correct for complex leaves.

    Uses ``|g|**2 = re**2 + im**2`` per element, so a purely imaginary gradient
    has a nonzero norm (unlike ``real(g)**2``, which would drop the imaginary part).
    """
    import jax
    import jax.numpy as jnp

    # sqrt of the summed squared moduli over all leaves.
    leaves = jax.tree_util.tree_leaves(tree)
    if not leaves:
        return jnp.asarray(0.0)
    squared = [jnp.sum(jnp.abs(leaf) ** 2) for leaf in leaves]
    return jnp.sqrt(jnp.sum(jnp.stack(squared)))


def _tree_all_finite(tree: Any) -> Any:
    """
    Return ``True`` iff every leaf is entirely finite (real and imaginary parts).

    Checks the leaves directly rather than relying on a scalar norm, so a single
    non-finite gradient entry is caught even if it would not surface in the norm.
    """
    import jax
    import jax.numpy as jnp

    leaves = jax.tree_util.tree_leaves(tree)
    if not leaves:
        return jnp.asarray(True)
    return jnp.stack([jnp.all(jnp.isfinite(leaf)) for leaf in leaves]).all()


@dataclass
class OptaxResult:
    """
    Result of :func:`minimize_optax`.

    Parameters
    ----------
    success : bool
        Whether the run reached ``grad_norm <= tol`` and stayed finite.
    status : int
        ``0`` converged, ``1`` iteration cap, ``2`` non-finite.
    message : str
        Readable status.
    num_iters : int
        Iterations executed.
    nfev, njev : int
        Evaluations by the driver itself: ``num_iters + 1`` each.
    n_linesearch_steps : int
        Line-search steps summed over the run; about one extra evaluation each.
    final_value, final_grad_norm : float
        Objective and coordinate-gradient norm at the final point.
    x_element : Any
        The minimizer, an element of ``F.domain``.
    history : dict
        ``iteration``, ``value``, ``value_delta`` and ``grad_norm`` arrays.
    compile_seconds, execution_seconds, average_iteration_ms : float
        Compile time, execution time, and mean time per iteration.
    """

    success: bool
    status: int
    message: str
    num_iters: int
    nfev: int
    njev: int
    n_linesearch_steps: int
    final_value: float
    final_grad_norm: float
    x_element: Any
    history: dict
    compile_seconds: float
    execution_seconds: float
    average_iteration_ms: float


def minimize_optax(
    F: Any,
    x0: Any,
    opt: Any,
    *,
    max_iter: int = 1000,
    tol: float = 1e-6,
    project: Callable[[Any], Any] | None = None,
    verbose: int = 1,
    log_every: int = 50,
    history_every: int | None = None,
    record_history: bool = True,
    progress_callback: Callable[[dict], None] | None = None,
) -> OptaxResult:
    r"""
    Minimize a functional with an optax optimizer inside one ``jax.jit`` while-loop.

    Each iteration evaluates ``F.value_and_grad`` once and hands optax the coordinate
    gradient ``X.riesz(F.grad(x))``. It stops when that gradient's norm is at most
    ``tol``.

    Parameters
    ----------
    F : Functional
        Objective on a JAX-backed inner-product space, with ``grad`` implemented.
    x0 : pytree
        Initial parameters in ``F.domain``.
    opt : optax.GradientTransformation
        The optimizer, e.g. ``optax.adam(1e-2)`` or ``optax.lbfgs()``.
    max_iter : int
        Iteration cap.
    tol : float
        Tolerance on the coordinate-gradient norm.
    project : callable, optional
        Map applied to the parameters after each update, e.g. a projection.
    verbose : int
        ``0`` silent, ``1`` summary, ``2`` a row every ``log_every`` iterations.
    log_every, history_every : int
        Logging and history cadences; neither affects convergence.
    record_history : bool
        Whether to record value, value change and gradient norm.
    progress_callback : callable, optional
        Called with each recorded row after the loop finishes.

    Returns
    -------
    OptaxResult
        Final statistics, the minimizer ``x_element``, the history and timings.

    Raises
    ------
    TypeError
        If ``F`` is not a :class:`~spacecore.Functional`, or its domain has no inner
        product or is not JAX-backed.
    ValueError
        If ``max_iter`` or ``tol`` is negative, or a cadence is not positive.
    ImportError
        If ``optax`` is not installed.

    See Also
    --------
    spacecore.optimize.minimize_scipy : SciPy ``minimize`` with the same handoff.

    Examples
    --------
    .. code-block:: python

        import numpy as np
        import optax
        import spacecore as sc

        ctx = sc.Context(sc.JaxOps(), dtype=np.float32)
        X = sc.DenseCoordinateSpace((2,), ctx, check_level="none")
        Q = sc.DenseLinOp(ctx.asarray([[3.0, 0.0], [0.0, 1.0]]), X, X, ctx)
        linear = sc.InnerProductFunctional(ctx.asarray([-3.0, -2.0]), X)
        F = sc.LinOpQuadraticForm(Q, linear)

        res = sc.minimize_optax(F, X.zeros(), optax.adam(1e-1), max_iter=1000, tol=1e-6)
        # res.x_element is approximately (1.0, 2.0); res.success is True
    """
    # Validate everything before importing optax and jax.
    F = require_functional(F, "minimize_optax")
    X = domain_with_geometry(F, "minimize_optax")
    if getattr(X.ops, "family", None) != "jax":
        raise TypeError(
            "minimize_optax requires a JAX-backed domain (it compiles the loop "
            f"with jax.jit); F.domain uses the {getattr(X.ops, 'family', '?')!r} "
            "backend. Build the domain with a JaxOps context, or use minimize_scipy "
            "for the NumPy backend."
        )
    max_iter = int(max_iter)
    if max_iter < 0:
        raise ValueError(f"max_iter must be non-negative, got {max_iter}.")
    if tol < 0:
        raise ValueError(f"tol must be non-negative, got {tol}.")
    if log_every <= 0:
        raise ValueError(f"log_every must be positive, got {log_every}.")
    history_every = log_every if history_every is None else int(history_every)
    if history_every <= 0:
        raise ValueError(f"history_every must be positive, got {history_every}.")

    try:
        import optax
    except ImportError as exc:  # pragma: no cover - exercised only without optax
        raise ImportError(
            "minimize_optax requires the optional 'optax' dependency; "
            "install it with `pip install spacecore[optax]`."
        ) from exc
    import jax
    import jax.numpy as jnp

    verbose = int(verbose)
    project_fn = (lambda p: p) if project is None else project
    # Tree domains iterate on the canonical pytree of x0.
    params = X.unflatten_tree(X.flatten_tree(x0)) if isinstance(X, TreeSpace) else x0
    # Lets line-search optimizers receive value, grad and value_fn in update().
    opt = optax.with_extra_args_support(opt)

    def evaluate(p: Any):
        value, mgrad = F.value_and_grad(p)  # fused value + metric (Riesz) gradient
        grad = X.riesz(mgrad)  # metric -> coordinate gradient for optax
        return value, grad, _tree_l2_norm(grad)

    def value_fn(p: Any):  # objective for line-search optimizers (optax minimizes)
        return F.value(p)

    # One optimizer update, then projection, then a fused re-evaluation.
    def step(state: _OptaxState) -> _OptaxState:
        updates, opt_state = opt.update(
            state.grad,
            state.opt_state,
            state.params,
            value=state.value,
            grad=state.grad,
            value_fn=value_fn,
        )
        p = project_fn(optax.apply_updates(state.params, updates))
        value, grad, grad_norm = evaluate(p)  # exactly one value_and_grad per step
        return _OptaxState(p, opt_state, value, grad, grad_norm)

    # History rows: one per history_every steps, plus the first and the last.
    value0, grad0, grad_norm0 = evaluate(params)
    init_state = _OptaxState(params, opt.init(params), value0, grad0, grad_norm0)
    real_dtype = jnp.asarray(jnp.real(value0)).dtype
    capacity = max_iter // history_every + 2

    def record(h: _History, index, iteration, value, delta, grad_norm) -> _History:
        return _History(
            iteration=h.iteration.at[index].set(jnp.asarray(iteration, jnp.int32)),
            value=h.value.at[index].set(jnp.real(value)),
            value_delta=h.value_delta.at[index].set(delta),
            grad_norm=h.grad_norm.at[index].set(grad_norm),
        )

    def run_loop(init_state: _OptaxState, tol_val):
        # Preallocated history keeps the loop body's shapes fixed, hence jittable.
        zeros = jnp.zeros((capacity,), real_dtype)
        history = _History(jnp.zeros((capacity,), jnp.int32), zeros, zeros, zeros)
        finite0 = _tree_all_finite(
            (init_state.value, init_state.grad)
        ) & jnp.isfinite(init_state.grad_norm)
        if record_history:
            history = record(
                history, 0, 0, init_state.value, jnp.asarray(0.0, real_dtype),
                init_state.grad_norm,
            )
            history_index = jnp.asarray(1, jnp.int32)
        else:
            history_index = jnp.asarray(0, jnp.int32)
        loop0 = _LoopState(
            jnp.asarray(0, jnp.int32), init_state, finite0, history, history_index,
            jnp.asarray(0, jnp.int32),
        )

        # Continue while under max_iter, above tol, and finite.
        def cond_fn(ls: _LoopState):
            return (ls.iteration < max_iter) & (ls.state.grad_norm > tol_val) & ls.finite

        def body_fn(ls: _LoopState):
            # Step, count line-search steps, and test convergence and divergence.
            prev_value = ls.state.value
            next_state = step(ls.state)
            iteration = ls.iteration + jnp.asarray(1, jnp.int32)
            step_ls = _linesearch_steps(next_state.opt_state)
            nls = ls.nls + (
                jnp.asarray(0, jnp.int32)
                if step_ls is None
                else step_ls.astype(jnp.int32)
            )
            value = next_state.value
            delta = jnp.real(value - prev_value)
            grad_norm = next_state.grad_norm
            finite = _tree_all_finite((value, next_state.grad)) & jnp.isfinite(grad_norm)
            done_after = (iteration >= max_iter) | (grad_norm <= tol_val) | (~finite)

            # Print from inside the compiled loop, on logging steps only.
            if verbose >= 2:
                log_now = ((iteration % log_every) == 0) | done_after
                jax.lax.cond(
                    log_now,
                    lambda: jax.debug.print(
                        _ROW_FMT, i=iteration, v=jnp.real(value), d=delta,
                        g=grad_norm, ordered=True,
                    ),
                    lambda: None,
                )

            # Write a history row on schedule; lax.cond keeps it traced.
            if record_history:
                do_record = ((iteration % history_every) == 0) | done_after

                def write(args):
                    h, hi = args
                    return (
                        record(h, hi, iteration, value, delta, grad_norm),
                        hi + jnp.asarray(1, jnp.int32),
                    )

                history, history_index = jax.lax.cond(
                    do_record, write, lambda a: a, (ls.history, ls.history_index)
                )
            else:
                history, history_index = ls.history, ls.history_index

            return _LoopState(iteration, next_state, finite, history, history_index, nls)

        return jax.lax.while_loop(cond_fn, body_fn, loop0)

    # Compile ahead of time so compile and execution are timed apart.
    run = jax.jit(run_loop)
    tol_arr = jnp.asarray(tol, real_dtype)

    t0 = perf_counter()
    compiled = run.lower(init_state, tol_arr).compile()
    compile_seconds = perf_counter() - t0

    if verbose >= 2:
        print(_HEADER)
        print(_RULE)
        print(_ROW_FMT.format(i=0, v=float(np.real(value0)), d=0.0, g=float(grad_norm0)))

    t1 = perf_counter()
    result = jax.device_get(compiled(init_state, tol_arr))
    execution_seconds = perf_counter() - t1

    # Unpack the result on the host.
    final = result.state
    num_iters = int(result.iteration)
    n_linesearch_steps = int(result.nls)
    # Count only the value_and_grad calls the driver itself makes: one per
    # iteration plus the initial evaluation. A line-search optimizer's internal
    # evaluations are reported separately as n_linesearch_steps and are not folded
    # in here (their exact count is optax-internal, and the driver does not reuse
    # them via value_and_grad_from_state -- see the class docstring).
    nfev = njev = num_iters + 1
    final_value = float(np.real(final.value))
    final_grad_norm = float(final.grad_norm)
    # Return the minimizer as an element of F.domain (bound element for a
    # structured space; the raw array is already an element otherwise).
    x_element = X.element(final.params) if isinstance(X, TreeSpace) else final.params
    # Status 2: diverged; 0: converged; 1: iteration cap.
    finite = bool(np.asarray(result.finite))
    success = finite and final_grad_norm <= tol
    if not finite:
        status, message = 2, "diverged: value or gradient became nonfinite"
    elif success:
        status, message = 0, "converged"
    else:
        status, message = 1, "maximum iterations reached"

    # Trim the preallocated history to the rows actually written.
    if record_history:
        n = int(result.history_index)
        history = {
            "iteration": np.asarray(result.history.iteration[:n], dtype=int),
            "value": np.asarray(result.history.value[:n], dtype=float),
            "value_delta": np.asarray(result.history.value_delta[:n], dtype=float),
            "grad_norm": np.asarray(result.history.grad_norm[:n], dtype=float),
        }
    else:
        history = {}

    # Callbacks run after the compiled loop, replaying the recorded rows.
    if progress_callback is not None and record_history:
        for i in range(len(history["iteration"])):
            progress_callback(
                {
                    "iteration": int(history["iteration"][i]),
                    "value": float(history["value"][i]),
                    "value_delta": float(history["value_delta"][i]),
                    "grad_norm": float(history["grad_norm"][i]),
                }
            )

    average_iteration_ms = (
        execution_seconds * 1000.0 / num_iters if num_iters else 0.0
    )

    if verbose >= 1:
        if verbose >= 2:
            print(_RULE)
        print(f"status   : {message}")
        print(f"iters    : {num_iters}  (nfev={nfev}, njev={njev}, ls={n_linesearch_steps})")
        print(f"value    : {final_value:+.8e}")
        print(f"grad_norm: {final_grad_norm:.3e}")
        print(
            f"compile: {compile_seconds:.2f} s   execution: {execution_seconds:.2f} s"
            f"   avg/iter: {average_iteration_ms:.3f} ms"
        )

    return OptaxResult(
        success=success,
        status=status,
        message=message,
        num_iters=num_iters,
        nfev=nfev,
        njev=njev,
        n_linesearch_steps=n_linesearch_steps,
        final_value=final_value,
        final_grad_norm=final_grad_norm,
        x_element=x_element,
        history=history,
        compile_seconds=compile_seconds,
        execution_seconds=execution_seconds,
        average_iteration_ms=average_iteration_ms,
    )
