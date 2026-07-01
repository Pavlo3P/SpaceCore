"""Compiled optax optimizer driver (ADR-018, 0.4.2 W1).

Minimizes a SpaceCore :class:`~spacecore.Functional` with an
`optax <https://optax.readthedocs.io>`_ optimizer. The whole optimization runs
inside a single ``jax.jit(jax.lax.while_loop(...))``: the fused
``F.value_and_grad`` is evaluated exactly once per iteration and cached in
``_OptaxState``, so the stopping test ``grad_norm <= tol`` and the progress log
reuse cached values with no recomputation and no per-iteration host sync. Cadence
parameters (``log_every`` / ``history_every``) govern logging only, never
convergence.

As with the SciPy adapters, the gradient handed to optax is the coordinate
gradient ``X.riesz(F.grad(x))`` -- optax applies coordinate updates
(``params - lr * grad``), so the metric-to-coordinate conversion is mandatory on
a weighted space and the identity on a Euclidean one (ADR-018).

``optax`` is an optional dependency, imported lazily so that ``import spacecore``
does not require it. The domain must be JAX-backed (the loop is compiled with
``jax.jit``); build it with ``check_level="none"`` so the traced ``F.value`` /
``F.grad`` do not perform data-dependent host checks (ADR-018, ergonomics run-02).

Information lost at the optax boundary
--------------------------------------
* **Geometry.** optax updates are taken in the flat coordinate metric; the domain
  geometry survives only through the ``X.riesz`` gradient conversion.
* **Representation.** ``x0`` may be a raw array or a tuple/list/dict of arrays; a
  bound ``TreeElement`` is normalized to its raw pytree at entry so that
  ``optax.apply_updates`` matches it against the converted gradient's structure.
* **Field.** A complex domain is not rejected -- optax's updates are
  complex-capable -- so the correctness of complex-valued descent is the caller's
  responsibility.
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
    Return an optax state's ``num_linesearch_steps`` counter, or ``None``.

    Line-search optimizers (e.g. ``optax.lbfgs``) expose
    ``info.num_linesearch_steps`` somewhere in their state; gradient-transformation
    optimizers (adam, sgd, ...) do not. The search is structural (walks NamedTuple
    fields / tuples / dicts) so it is robust to how the optimizer nests its state.
    """
    stack = [opt_state]
    while stack:
        node = stack.pop()
        fields = getattr(node, "_fields", None)
        if fields is not None:  # NamedTuple
            if "num_linesearch_steps" in fields:
                return node.num_linesearch_steps
            stack.extend(getattr(node, f) for f in fields)
        elif isinstance(node, (tuple, list)):
            stack.extend(node)
        elif isinstance(node, dict):
            stack.extend(node.values())
    return None


def _tree_l2_norm(tree: Any) -> Any:
    """
    Euclidean L2 norm of a gradient pytree, correct for complex leaves.

    Uses ``|g|**2 = re**2 + im**2`` per element, so a purely imaginary gradient
    has a nonzero norm (unlike ``real(g)**2``, which would drop the imaginary part).
    """
    import jax
    import jax.numpy as jnp

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
        ``True`` iff the run stopped with ``grad_norm <= tol`` and stayed finite.
    status : int
        ``0`` converged, ``1`` maximum iterations, ``2`` nonfinite (diverged).
    message : str
        Human-readable status.
    num_iters : int
        Iterations executed.
    nfev, njev : int
        Value and gradient evaluations the driver itself performs: one fused
        ``value_and_grad`` per iteration plus one initial evaluation, i.e.
        ``num_iters + 1``. This does not include evaluations a line-search
        optimizer makes internally -- those are reported separately as
        ``n_linesearch_steps``.
    n_linesearch_steps : int
        Cumulative line-search iterations reported by the optimizer (from
        ``num_linesearch_steps`` in the optax state, e.g. ``optax.lbfgs``); ``0``
        when the optimizer performs no line search. Each step performs roughly one
        internal objective evaluation. The driver does not reuse these via
        ``optax.value_and_grad_from_state``: that would substitute the autodiff
        gradient of ``F.value`` for the ``X.riesz(F.grad)`` gradient the SpaceCore
        contract requires, so line-search values are recomputed rather than cached.
    final_value, final_grad_norm : float
        Objective and coordinate-gradient norm at the final point.
    x_element : Any
        The minimizer, an element of ``F.domain`` (a bound element for a
        structured space such as ``TreeSpace``; a raw array otherwise).
    history : dict
        Arrays ``iteration``/``value``/``value_delta``/``grad_norm`` (empty when
        ``record_history=False``).
    compile_seconds, execution_seconds, average_iteration_ms : float
        Compile (AOT) time, steady-state execution time, and per-iteration mean.
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
    Minimize a SpaceCore functional with a compiled, convergence-aware optax loop.

    The whole loop runs inside ``jax.jit(jax.lax.while_loop(...))``. The fused
    ``F.value_and_grad`` is evaluated once per iteration and cached, so the
    stopping test ``grad_norm <= tol`` and the progress log reuse cached values
    with no recomputation and no per-iteration host sync. Cadence parameters
    (``log_every`` / ``history_every``) govern logging only, never convergence.

    Progress columns: ``iteration``, functional value ``F(x)``, objective delta
    ``ΔF = F_k - F_{k-1}``, and the coordinate-gradient norm.

    Parameters
    ----------
    F : Functional
        Objective with an inner-product, JAX-backed domain ``X = F.domain`` and an
        implemented ``F.grad`` (hence ``F.value_and_grad``).
    x0 : pytree
        Initial parameters, an element of ``F.domain``. A raw array or
        tuple/list/dict of arrays is used as-is; a bound ``TreeElement`` is
        normalized to its raw pytree.
    opt : optax.GradientTransformation
        An optax optimizer such as ``optax.adam(1e-2)`` or ``optax.lbfgs()``.
    max_iter : int
        Iteration cap (non-negative).
    tol : float
        Convergence tolerance on the coordinate-gradient L2 norm (non-negative).
    project : callable, optional
        Optional retraction applied to the parameters after each optax update
        (e.g. projection onto a constraint set). Default: identity.
    verbose : int
        ``0`` silent, ``1`` final summary, ``2`` per-``log_every`` progress rows.
    log_every, history_every : int
        Live-logging and history-sampling cadences. ``history_every`` defaults to
        ``log_every``.
    record_history : bool
        Whether to record the iteration/value/value_delta/grad_norm history.
    progress_callback : callable, optional
        Called once per recorded row *after* the loop (compiled loops cannot call
        Python callbacks live) as ``progress_callback(row_dict)``.

    Returns
    -------
    OptaxResult
        Final statistics, the minimizer ``x_element``, the recorded history, and
        compile-vs-execution timing.

    Raises
    ------
    TypeError
        If ``F`` is not a :class:`~spacecore.Functional`, its domain has no
        inner-product geometry, or its domain is not JAX-backed.
    ValueError
        If ``max_iter``/``tol`` are negative or ``log_every``/``history_every`` are
        non-positive.
    ImportError
        If ``optax`` is not installed (``pip install spacecore[optax]``).

    See Also
    --------
    spacecore.optimize.minimize_scipy : SciPy ``minimize`` with the same handoff.

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

        res = sc.minimize_optax(F, X.zeros(), optax.adam(1e-1), max_iter=1000, tol=1e-6)
        # res.x_element is approximately (1.0, 2.0); res.success is True
    """
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
    params = X.unflatten_tree(X.flatten_tree(x0)) if isinstance(X, TreeSpace) else x0
    opt = optax.with_extra_args_support(opt)

    def evaluate(p: Any):
        value, mgrad = F.value_and_grad(p)  # fused value + metric (Riesz) gradient
        grad = X.riesz(mgrad)  # metric -> coordinate gradient for optax
        return value, grad, _tree_l2_norm(grad)

    def value_fn(p: Any):  # objective for line-search optimizers (optax minimizes)
        return F.value(p)

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

        def cond_fn(ls: _LoopState):
            return (ls.iteration < max_iter) & (ls.state.grad_norm > tol_val) & ls.finite

        def body_fn(ls: _LoopState):
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
    finite = bool(np.asarray(result.finite))
    success = finite and final_grad_norm <= tol
    if not finite:
        status, message = 2, "diverged: value or gradient became nonfinite"
    elif success:
        status, message = 0, "converged"
    else:
        status, message = 1, "maximum iterations reached"

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
