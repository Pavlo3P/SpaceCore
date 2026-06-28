"""Per-backend device enumeration.

Returns the list of physical devices each backend can run on in the
current environment. The runner uses this list to iterate every
``(backend, device)`` combination per probe.

Device labels are SpaceCore-side strings, *not* backend-native device
objects: ``"cpu"``, ``"cuda"``, ``"mps"``, ``"gpu"``, ``"tpu"``. The
helper :func:`device_object` resolves a label back to the backend
native object when one is needed (e.g. to call ``jax.device_put``).
"""
from __future__ import annotations

from typing import Any


def devices_for(backend: str) -> tuple[str, ...]:
    """Return every device label the backend can target."""
    if backend == "numpy":
        return ("cpu",)
    if backend == "jax":
        return _jax_devices()
    if backend == "torch":
        return _torch_devices()
    if backend == "cupy":
        return ("cuda",) if _cupy_available() else ()
    return ()


def device_object(backend: str, device: str) -> Any:
    """Resolve a label to the backend-native device object.

    ``None`` is returned when the backend selects a device implicitly
    (e.g. NumPy is always CPU and has no concept of device objects).
    """
    if backend == "numpy":
        return None
    if backend == "jax":
        try:
            import jax

            if device == "cpu":
                return jax.devices("cpu")[0]
            if device in {"gpu", "cuda"}:
                return jax.devices("gpu")[0]
            if device == "tpu":
                return jax.devices("tpu")[0]
        except (ImportError, RuntimeError, IndexError):
            return None
    if backend == "torch":
        try:
            import torch

            if device == "cpu":
                return torch.device("cpu")
            if device in {"gpu", "cuda"}:
                return torch.device("cuda")
            if device == "mps":
                return torch.device("mps")
        except ImportError:
            return None
    return None


def _jax_devices() -> tuple[str, ...]:
    try:
        import jax
    except ImportError:
        return ()
    devices: list[str] = []
    try:
        if jax.devices("cpu"):
            devices.append("cpu")
    except (RuntimeError, ValueError):
        pass
    for label in ("gpu", "tpu"):
        try:
            if jax.devices(label):
                devices.append(label)
        except (RuntimeError, ValueError):
            continue
    return tuple(devices)


def _torch_devices() -> tuple[str, ...]:
    try:
        import torch
    except ImportError:
        return ()
    devices = ["cpu"]
    if torch.cuda.is_available():
        devices.append("cuda")
    # MPS is Apple's Metal backend. Skip if torch was built without it
    # or the device is not present.
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available() and mps.is_built():
        devices.append("mps")
    return tuple(devices)


def _cupy_available() -> bool:
    try:
        import cupy
        return bool(cupy.cuda.runtime.getDeviceCount())
    except (ImportError, Exception):
        return False


def all_available_devices() -> dict[str, tuple[str, ...]]:
    """Return ``{backend: (device, ...)}`` for every backend present."""
    return {
        backend: devices_for(backend)
        for backend in ("numpy", "jax", "torch", "cupy")
        if devices_for(backend)
    }
