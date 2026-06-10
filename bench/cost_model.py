from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from typing import Any

import numpy as np

from .harness import time_op


class _Dispatch(Enum):
    A = auto()


@dataclass(frozen=True, slots=True)
class PrimitiveCosts:
    wrapper_ns: float
    frame_ns: float
    attr_ns: float
    enum_dispatch_ns: float
    reshape_fixed_ns: float
    check_member_ns: float
    riesz_fixed_ns: float
    riesz_per_element_ns: float
    vmap_fallback_per_element_ns: float

    def to_json(self) -> dict[str, float]:
        return {
            "wrapper_ns": self.wrapper_ns,
            "frame_ns": self.frame_ns,
            "attr_ns": self.attr_ns,
            "enum_dispatch_ns": self.enum_dispatch_ns,
            "reshape_fixed_ns": self.reshape_fixed_ns,
            "check_member_ns": self.check_member_ns,
            "riesz_fixed_ns": self.riesz_fixed_ns,
            "riesz_per_element_ns": self.riesz_per_element_ns,
            "vmap_fallback_per_element_ns": self.vmap_fallback_per_element_ns,
        }


def _choose_number(fn, target_ns: float = 10_000_000.0) -> int:
    one = max(time_op(fn, repeat=3, number=1, warmup=1)["best_ns"], 1.0)
    return max(1, min(100_000, int(target_ns / one)))


def _measure(fn) -> float:
    number = _choose_number(fn)
    return time_op(fn, repeat=9, number=number, warmup=2)["best_ns"]


def calibrate_primitives() -> PrimitiveCosts:
    """Measure machine-local Python primitive costs used by the model."""

    class Box:
        value = 1

        def method(self):
            return self.value

    box = Box()

    def direct():
        return 1

    def frame():
        return direct()

    def attr():
        return box.value

    def enum_dispatch():
        return _Dispatch.A is _Dispatch.A

    arr = np.ones(4096)
    weights = 1.0 + np.arange(4096) / 4096.0

    def reshape_roundtrip():
        return arr.reshape((64, 64)).reshape((-1,))

    def check_member():
        if arr.shape != (4096,):
            raise ValueError
        return arr

    def riesz():
        return weights * arr

    xs = np.ones((16, 16))

    def vmap_fallback():
        return np.stack([x + 1.0 for x in xs], axis=0)

    class Wrapper:
        _enable_checks = False
        _mode = _Dispatch.A

        def core(self, x):
            if self._mode is _Dispatch.A:
                return x
            return x

        def apply(self, x):
            checks = self._enable_checks
            if checks:
                raise AssertionError
            y = self.core(x)
            if checks:
                raise AssertionError
            return y

    wrapper = Wrapper()
    scalar = np.ones(())

    direct_ns = _measure(direct)
    frame_ns = max(_measure(frame) - direct_ns, 1.0)
    attr_ns = max(_measure(attr) - direct_ns, 1.0)
    enum_ns = max(_measure(enum_dispatch) - direct_ns, 1.0)
    reshape_ns = max(_measure(reshape_roundtrip) - direct_ns, 1.0)
    check_ns = max(_measure(check_member) - direct_ns, 1.0)
    riesz_total = max(_measure(riesz) - _measure(lambda: arr), 1.0)
    fallback_total = max(_measure(vmap_fallback), 1.0)
    wrapper_ns = max(_measure(lambda: wrapper.apply(scalar)) - _measure(lambda: scalar), 1.0)
    return PrimitiveCosts(
        wrapper_ns=wrapper_ns,
        frame_ns=frame_ns,
        attr_ns=attr_ns,
        enum_dispatch_ns=enum_ns,
        reshape_fixed_ns=reshape_ns,
        check_member_ns=check_ns,
        riesz_fixed_ns=min(riesz_total, 500.0),
        riesz_per_element_ns=max((riesz_total - min(riesz_total, 500.0)) / arr.size, 0.001),
        vmap_fallback_per_element_ns=fallback_total / xs.shape[0],
    )


def _component(name: str, count: float, unit_ns: float) -> dict[str, float]:
    return {"name": name, "count": float(count), "ns": float(max(count, 0.0) * unit_ns)}


def predict_overhead(case: Any, costs: PrimitiveCosts) -> tuple[float, list[dict[str, float]]]:
    """Predict fixed SpaceCore-vs-bare overhead for a benchmark case."""
    components: list[dict[str, float]] = []
    batch = case.batch or 1
    size = max(case.size, 1)
    operation = case.operation
    op_type = case.operator_type
    geometry = case.geometry

    wrappers = 1.0
    frames = 0.5
    attrs = 5.0
    enum = 1.0 if op_type in {"DenseLinOp", "SparseLinOp", "DiagonalLinOp"} else 0.5
    reshapes = 0.0
    checks = 2.0 if case.checks else 0.0
    riesz = 0.0
    vmap = 0.0

    if (
        "vapply" in operation
        or "rvapply" in operation
        or "vvalue" in operation
        or "vgrad" in operation
    ):
        frames += 0.5
        attrs += 2.0
        if "fallback" in case.mode:
            vmap += batch
    if (
        operation in {"vvalue", "vgrad"}
        and case.backend in {"numpy-eager"}
        and op_type not in {"InnerProductFunctional", "LinOpQuadraticForm"}
    ):
        vmap += batch
    if case.shape_kind == "tensor":
        reshapes += 2.0
    if geometry in {"weighted", "general-metric"} and operation in {"rapply", "rvapply", "grad"}:
        riesz += 2.0
    if op_type in {"ComposedLinOp", "SumLinOp", "ScaledLinOp", "ProductLinOp"}:
        wrappers += 2.0
        attrs += 4.0
    if "Functional" in op_type:
        wrappers += 1.0
        attrs += 2.0

    components.append(_component("wrapper", wrappers, costs.wrapper_ns))
    components.append(_component("frame", frames, costs.frame_ns))
    components.append(_component("attr", attrs, costs.attr_ns))
    components.append(_component("enum_dispatch", enum, costs.enum_dispatch_ns))
    components.append(_component("reshape", reshapes, costs.reshape_fixed_ns))
    components.append(_component("check_member", checks, costs.check_member_ns))
    if riesz:
        riesz_ns = riesz * (costs.riesz_fixed_ns + costs.riesz_per_element_ns * size * batch)
        components.append({"name": "riesz", "count": float(riesz), "ns": float(riesz_ns)})
    if vmap:
        components.append(_component("vmap_fallback", vmap, costs.vmap_fallback_per_element_ns))

    predicted = sum(c["ns"] for c in components)
    if batch > 1 and operation in {"vapply", "rvapply", "vvalue"}:
        components.append(
            {"name": "amortized_per_element", "count": float(batch), "ns": float(predicted / batch)}
        )
    return float(predicted), components


def classify_gap(measured_ns: float, predicted_ns: float) -> str:
    measured = max(measured_ns, 0.0)
    predicted = max(predicted_ns, 250.0)
    if measured <= max(predicted * 1.5 + 500.0, 2_000.0):
        return "ok"
    if measured <= max(predicted * 2.0 + 1_000.0, 10_000.0):
        return "slightly_high"
    return "anomalous"
