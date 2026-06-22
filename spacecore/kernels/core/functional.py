"""Concrete core kernels for :mod:`spacecore.functional` objects.

A functional's public ``value`` / ``grad`` / ``vvalue`` / ``vgrad`` validate
their input at the boundary (and, at the ``standard`` level, the scalar/batch
output shape). The check-free *cores* below carry the pure compute, so the
boundary validation runs once and composite functionals can reach their
operands' cores without re-validating intermediates — e.g.
``ComposedFunctional`` evaluates ``F._value_core(A._apply_core(x))`` and
``LinOpQuadraticForm`` reaches ``Q._apply_core`` / ``linear._value_core``.

The functions are duck-typed on the functional. They use the space's check-free
``_inner_core`` / ``_add_core`` when available (matching the LinOp leaf kernels)
and import nothing from :mod:`spacecore.functional`, so no import cycle forms.
"""
from __future__ import annotations

from typing import Any

from ..._batching import _batched_inner
from ...space import TreeSpace
from ._rules import CoreKernelSet, register_core_kernels


def _broadcast_space_element(space: Any, value: Any, n: int) -> Any:
    """Broadcast a single space element to a leading-axis batch of size ``n``."""
    if isinstance(space, TreeSpace):
        return space.unflatten_tree(
            tuple(
                _broadcast_space_element(part, component, n)
                for part, component in zip(space.leaf_spaces, space.flatten_tree(value))
            )
        )
    return space.ops.broadcast_to(value, (n,) + tuple(space.shape))


# ---------------------------------------------------------------------------
# Linear functionals — constant Riesz gradient (shared by all subclasses)
# ---------------------------------------------------------------------------
def linear_grad_core(op: Any, x: Any) -> Any:
    return op.representer


def linear_vgrad_core(op: Any, xs: Any) -> Any:
    dom = op.dom
    sample = dom.flatten_tree(xs)[0] if isinstance(dom, TreeSpace) else xs
    n = int(getattr(sample, "shape", (0,))[0])
    return _broadcast_space_element(dom, op.representer, n)


# ---------------------------------------------------------------------------
# Inner-product functional — value(x) = <c, x>
# ---------------------------------------------------------------------------
def inner_product_value_core(op: Any, x: Any) -> Any:
    inner = getattr(op.domain, "_inner_core", op.domain.inner)
    return inner(op._c, x)


def inner_product_vvalue_core(op: Any, xs: Any) -> Any:
    dom = op.dom
    ops = op.ops
    if dom.is_euclidean and len(tuple(dom.shape)) == 1:
        xs_flat = xs
        c_flat = ops.conj(op._c)
    else:
        c_dual = op._c if dom.is_euclidean else dom.riesz(op._c)
        c_flat = ops.conj(dom.flatten(c_dual))
        xs_flat = dom.flatten_batch(xs)
    return xs_flat @ c_flat


# ---------------------------------------------------------------------------
# Matrix-free linear functional — user callables
# ---------------------------------------------------------------------------
def matrixfree_linear_value_core(op: Any, x: Any) -> Any:
    return op.value_fn(x)


def matrixfree_linear_vvalue_core(op: Any, xs: Any) -> Any:
    if op.vvalue_fn is None:
        return op.ops.vmap(op.value, in_axes=0, out_axes=0)(xs)
    return op.vvalue_fn(xs)


# ---------------------------------------------------------------------------
# LinOp quadratic form — 1/2 <x, Qx> + linear(x) + a
# ---------------------------------------------------------------------------
def linop_quadratic_value_core(op: Any, x: Any) -> Any:
    qx = op.Q._apply_core(x)
    inner = getattr(op.domain, "_inner_core", op.domain.inner)
    value = 0.5 * inner(x, qx)
    if op.linear is not None:
        value = value + op.linear._value_core(x)
    return value + op.a


def linop_quadratic_grad_core(op: Any, x: Any) -> Any:
    grad = op.Q._apply_core(x)
    if op.linear is not None:
        add = getattr(op.domain, "_add_core", op.domain.add)
        grad = add(grad, op.linear.representer)
    return grad


def linop_quadratic_vvalue_core(op: Any, xs: Any) -> Any:
    qxs = op.Q._vapply_core(xs)
    if op.domain.is_euclidean and hasattr(xs, "shape"):
        axes = tuple(range(1, len(tuple(xs.shape))))
        values = 0.5 * op.ops.sum(op.ops.conj(xs) * qxs, axis=axes)
    else:
        values = 0.5 * _batched_inner(op.domain, xs, qxs)
    if op.linear is not None:
        values = values + op.linear._vvalue_core(xs)
    return values + op.a


def linop_quadratic_vgrad_core(op: Any, xs: Any) -> Any:
    grads = op.Q._vapply_core(xs)
    if op.linear is not None:
        grads = op.domain.add_batch(grads, op.linear._vgrad_core(xs))
    return grads


# ---------------------------------------------------------------------------
# Composed functional — F(A x)
# ---------------------------------------------------------------------------
def composed_functional_value_core(op: Any, x: Any) -> Any:
    return op.F._value_core(op.A._apply_core(x))


register_core_kernels(CoreKernelSet(
    "functional-linear", grad=linear_grad_core, vgrad=linear_vgrad_core,
    notes="Constant Riesz gradient of a linear functional.",
))
register_core_kernels(CoreKernelSet(
    "inner-product-functional",
    value=inner_product_value_core, vvalue=inner_product_vvalue_core,
    notes="value(x) = <c, x> via the space's check-free inner product.",
))
register_core_kernels(CoreKernelSet(
    "matrixfree-linear-functional",
    value=matrixfree_linear_value_core, vvalue=matrixfree_linear_vvalue_core,
    notes="Invoke the user-supplied value/vvalue callables directly.",
))
register_core_kernels(CoreKernelSet(
    "linop-quadratic-form",
    value=linop_quadratic_value_core, grad=linop_quadratic_grad_core,
    vvalue=linop_quadratic_vvalue_core, vgrad=linop_quadratic_vgrad_core,
    notes="Reaches Q and linear-term cores without re-validating intermediates.",
))
register_core_kernels(CoreKernelSet(
    "composed-functional", value=composed_functional_value_core,
    notes="Pull-back F(Ax) via the operand cores.",
))
