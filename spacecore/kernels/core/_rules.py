"""Rules that bind *core apply kernels* to operator implementations.

A *core kernel* is the check-free fast path an operator runs once its boundary
has already been validated: the body of ``apply``/``rapply``/``vapply`` minus the
membership checks. Historically each operator hand-wrote those bodies as
``_apply_core``/``_rapply_core``/``_vapply_core`` methods. This module lets the
kernels live in the :mod:`spacecore.kernels` subpackage instead — organized as
concrete functions (the *kernels*) plus a registration/binding mechanism (the
*rules*) — while the operator classes only declare which kernel set they use.

The binding model is deliberately static and zero-overhead:

* A concrete kernel is a plain function ``kernel(op, operand) -> result`` that
  reads what it needs off the operator instance (duck-typed, so the kernel
  module never imports the operator classes and no import cycle forms).
* Related kernels are grouped into a :class:`CoreKernelSet` and registered by a
  stable name.
* An operator opts in with the :func:`core_kernels` class decorator. The
  decorator installs the set's functions as the class's
  ``_apply_core``/``_rapply_core``/``_vapply_core`` methods *once*, at class
  definition time. At call time ``op._apply_core(x)`` is therefore an ordinary
  bound-method call — there is no per-call dispatch, lookup, or applicability
  scan, so routing through the kernel registry costs nothing relative to a
  hand-written method.

This separates "what is the optimized core" (concrete kernels in
:mod:`spacecore.kernels.algebra`) from "which operator uses which core" (the
:func:`core_kernels` declarations on the operator classes), so a new kernel can
be added or swapped without editing operator bodies.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional, TypeVar

CoreFn = Callable[[Any, Any], Any]
"""Signature of a core kernel: ``(operator, operand) -> result``."""


# Each settable field maps to the dunder-free core method the decorator installs.
# LinOp operators use the apply/rapply/vapply/rvapply cores; Functional objects
# use the value/grad/vvalue/vgrad cores. A kernel set fills in whichever family
# its operator family needs.
_FIELD_TO_METHOD: dict[str, str] = {
    "apply": "_apply_core",
    "rapply": "_rapply_core",
    "vapply": "_vapply_core",
    "rvapply": "_rvapply_core",
    "value": "_value_core",
    "grad": "_grad_core",
    "vvalue": "_vvalue_core",
    "vgrad": "_vgrad_core",
}


@dataclass(frozen=True)
class CoreKernelSet:
    """A named bundle of an operator's check-free core kernels.

    A kernel set fills in the cores its operator family needs and leaves the rest
    ``None``. :class:`~spacecore.linop.LinOp` operators use
    ``apply``/``rapply``/``vapply``/``rvapply``; :class:`~spacecore.functional.Functional`
    objects use ``value``/``grad``/``vvalue``/``vgrad``. Each settable core
    ``(op, operand) -> result`` is installed by :func:`core_kernels` as the
    matching ``_*_core`` method; an unset core leaves the inherited one in place
    (the base classes fall back to the boundary-checked public method).

    Attributes
    ----------
    name
        Stable identifier referenced by :func:`core_kernels`.
    apply, rapply, vapply, rvapply
        LinOp cores: forward / adjoint / batched-forward / batched-adjoint.
    value, grad, vvalue, vgrad
        Functional cores: value / Riesz-gradient and their batched forms.
    notes
        One-line description of what the kernel set fuses or skips.
    """

    name: str
    apply: Optional[CoreFn] = None
    rapply: Optional[CoreFn] = None
    vapply: Optional[CoreFn] = None
    rvapply: Optional[CoreFn] = None
    value: Optional[CoreFn] = None
    grad: Optional[CoreFn] = None
    vvalue: Optional[CoreFn] = None
    vgrad: Optional[CoreFn] = None
    notes: str = ""

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("CoreKernelSet requires a non-empty name.")
        provided = [f for f in _FIELD_TO_METHOD if getattr(self, f) is not None]
        if not provided:
            raise ValueError(f"core kernel set {self.name!r}: at least one core is required")
        for field in _FIELD_TO_METHOD:
            value = getattr(self, field)
            if value is not None and not callable(value):
                raise TypeError(f"core kernel set {self.name!r}: {field} must be callable or None")


_CORE_KERNEL_SETS: dict[str, CoreKernelSet] = {}


def register_core_kernels(kernel_set: CoreKernelSet) -> CoreKernelSet:
    """Record ``kernel_set`` so operators can bind it by name.

    Re-registering the identical object is a no-op; a different set under an
    existing name is a programming error and raises.
    """
    existing = _CORE_KERNEL_SETS.get(kernel_set.name)
    if existing is not None and existing is not kernel_set:
        raise ValueError(f"core kernel set name collision: {kernel_set.name!r}")
    _CORE_KERNEL_SETS[kernel_set.name] = kernel_set
    return kernel_set


def get_core_kernels(name: str) -> CoreKernelSet:
    """Return the registered :class:`CoreKernelSet` named ``name``."""
    return _CORE_KERNEL_SETS[name]


def core_kernel_names() -> tuple[str, ...]:
    """Return every registered core-kernel-set name, in registration order."""
    return tuple(_CORE_KERNEL_SETS)


_C = TypeVar("_C", bound=type)


def core_kernels(name: str) -> Callable[[_C], _C]:
    """Class decorator that binds the named core-kernel set onto an operator.

    Installs the set's ``apply``/``rapply``/``vapply`` functions as the class's
    ``_apply_core``/``_rapply_core``/``_vapply_core`` methods. Functions left as
    ``None`` in the set are not bound, so the class keeps whatever it inherits
    (the base ``LinOp`` cores fall back to the boundary-checked public methods).

    Raises ``KeyError`` if ``name`` is not registered — kernel modules must be
    imported (registering their sets) before the operator classes that use them
    are defined.
    """
    kernel_set = _CORE_KERNEL_SETS[name]

    def decorate(cls: _C) -> _C:
        for field, method in _FIELD_TO_METHOD.items():
            fn = getattr(kernel_set, field)
            if fn is not None:
                setattr(cls, method, fn)
        setattr(cls, "_core_kernel_set", name)
        return cls

    return decorate
