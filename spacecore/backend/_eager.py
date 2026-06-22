from __future__ import annotations

from typing import Any, Callable, Optional, Sequence, Tuple

from ..types import Carry, R, T, X, Y


class EagerControlFlowMixin:
    """Eager Python implementations of the control-flow and pytree primitives.

    Backends without native staged control flow (NumPy, CuPy, PyTorch) share a
    single eager implementation of :meth:`fori_loop`, :meth:`while_loop`,
    :meth:`scan`, and :meth:`cond`, plus the small private pytree helpers that
    :meth:`scan` relies on. JAX deliberately does *not* use this mixin: it
    overrides every control-flow method with ``jax.lax.*`` so that tracing and
    compilation semantics are preserved.

    The mixin depends only on the portable :class:`~spacecore.backend.BackendOps`
    surface (``self.stack`` / ``self.asarray``), so leaf stacking works on every
    backend without referencing a backend-specific array namespace.
    """

    # -- pytree helpers ----------------------------------------------------

    def _tree_map(self, f: Callable[[Any], Any], tree: Any) -> Any:
        if isinstance(tree, dict):
            return {k: self._tree_map(f, v) for k, v in tree.items()}
        if isinstance(tree, tuple):
            return tuple(self._tree_map(f, v) for v in tree)
        if isinstance(tree, list):
            return [self._tree_map(f, v) for v in tree]
        return f(tree)

    def _tree_multimap(self, f: Callable[..., Any], *trees: Any) -> Any:
        # assumes matching structure
        t0 = trees[0]
        if isinstance(t0, dict):
            return {k: self._tree_multimap(f, *(t[k] for t in trees)) for k in t0.keys()}
        if isinstance(t0, tuple):
            return tuple(self._tree_multimap(f, *(t[i] for t in trees)) for i in range(len(t0)))
        if isinstance(t0, list):
            return [self._tree_multimap(f, *(t[i] for t in trees)) for i in range(len(t0))]
        return f(*trees)

    def _tree_take0(self, xs: Any) -> Any:
        """Grab a representative leaf to infer leading length."""
        if isinstance(xs, dict):
            return self._tree_take0(next(iter(xs.values())))
        if isinstance(xs, (tuple, list)):
            return self._tree_take0(xs[0])
        return xs

    def _tree_index(self, xs: Any, i: int) -> Any:
        """Take per-step slice ``xs[i]`` along axis 0 for each leaf."""

        def _idx(a: Any) -> Any:
            # If it's an ndarray-like with leading axis, slice it; else treat as scalar leaf.
            try:
                return a[i]
            except Exception:
                return a

        return self._tree_map(_idx, xs)

    def _tree_stack(self, ys_list: Sequence[Any]) -> Any:
        """Stack per-step outputs into a single pytree of arrays (axis 0)."""
        if not ys_list:
            # JAX would return empty stacked outputs when length == 0.
            return ()

        def _stack_leaves(*leaves: Any) -> Any:
            # Prefer the portable backend stack; fall back to asarray for ragged leaves.
            try:
                return self.stack(leaves, axis=0)
            except Exception:
                return self.asarray(leaves)

        return self._tree_multimap(_stack_leaves, *ys_list)

    # -- control flow ------------------------------------------------------

    def fori_loop(
        self,
        lower: int,
        upper: int,
        body_fun: Callable[[int, T], T],
        init_val: T,
    ) -> T:
        """Run a counted loop eagerly as a Python ``for`` loop.

        This executes without JAX tracing semantics; ``unroll``-style parameters
        are not accepted because eager backends gain nothing from them.
        """
        val = init_val
        for i in range(int(lower), int(upper)):
            val = body_fun(i, val)
        return val

    def while_loop(
        self,
        cond_fun: Callable[[T], bool],
        body_fun: Callable[[T], T],
        init_val: T,
    ) -> T:
        """Run a while-loop eagerly, converting the predicate to ``bool`` each step."""
        val = init_val
        while bool(cond_fun(val)):
            val = body_fun(val)
        return val

    def scan(
        self,
        f: Callable[[Carry, X], Tuple[Carry, Y]],
        init: Carry,
        xs: X,
        length: Optional[int] = None,
        reverse: bool = False,
        unroll: int = 1,
    ) -> Tuple[Carry, Y]:
        """Run a scan primitive eagerly, stacking per-step outputs at the end.

        ``unroll`` is accepted only for API parity with ``jax.lax.scan``. When
        ``xs`` is ``None`` an explicit ``length`` is required; otherwise the
        length is inferred from the leading axis of the first leaf of ``xs``.
        """
        carry = init

        if xs is None:
            if length is None:
                raise ValueError("scan(xs=None) requires an explicit `length`.")
            n = int(length)
            indices = range(n - 1, -1, -1) if reverse else range(n)
            ys_steps: list[Any] = []
            for _i in indices:
                carry, y = f(carry, None)  # type: ignore[arg-type]
                ys_steps.append(y)
            if reverse:
                ys_steps.reverse()
            return carry, self._tree_stack(ys_steps)

        # infer length from xs if not provided
        if length is None:
            leaf0 = self._tree_take0(xs)
            try:
                n = int(leaf0.shape[0])  # ndarray-like
            except Exception as e:
                raise ValueError(
                    "Could not infer scan length from `xs`; pass `length=` explicitly."
                ) from e
        else:
            n = int(length)

        indices = range(n - 1, -1, -1) if reverse else range(n)
        ys_steps = []
        for i in indices:
            x_i = self._tree_index(xs, i)
            carry, y = f(carry, x_i)
            ys_steps.append(y)

        if reverse:
            ys_steps.reverse()

        return carry, self._tree_stack(ys_steps)

    def cond(
        self,
        pred: bool,
        true_fun: Callable[[T], R],
        false_fun: Callable[[T], R],
        *operands: Any,
    ) -> R:
        """Select a branch eagerly using Python truth-value conversion."""
        return true_fun(*operands) if bool(pred) else false_fun(*operands)
