"""Macrobenchmark suite.

The macro suite measures *algorithm-level* SpaceCore performance:
end-to-end CG, eigenvalue iteration, PDHG, QOT gradients, density
transforms, operator-algebra stress, and JAX full-loop. It complements
:mod:`bench._operations` which measures per-call micro overhead.

Macrobenchmarks are timed in four modes:

* ``bare`` — pure backend kernels, no SpaceCore objects.
* ``spacecore_public_none`` — public SpaceCore API with
  ``check_level="none"``.
* ``spacecore_public_cheap`` — public SpaceCore API with
  ``check_level="cheap"``.
* ``spacecore_lowered`` — SpaceCore lowers the workload to a
  backend-native or JAX-jitted kernel and executes that.

For JAX the runner additionally separates ``compile_time_ns`` from
``run_time_ns`` so steady-state runtimes are not contaminated by
one-time tracing cost.

The result schema is documented in :mod:`bench.macro._schema` and is
the JSON contract every macrobenchmark must emit.
"""
from __future__ import annotations

from ._registry import MacroBenchmark, MacroRegistry, registry
from ._schema import MacroResult, RUN_MODES, ModeName
from ._runner import run_benchmarks
from ._aggregate import group_summaries

# Importing the benchmark modules registers them.
from . import cg_poisson  # noqa: F401
from . import power_lanczos  # noqa: F401
from . import pdhg  # noqa: F401
from . import qot_barycenter  # noqa: F401
from . import density_pipeline  # noqa: F401
from . import operator_stress  # noqa: F401
from . import jax_full_loop  # noqa: F401

__all__ = [
    "MacroBenchmark",
    "MacroRegistry",
    "MacroResult",
    "ModeName",
    "RUN_MODES",
    "registry",
    "run_benchmarks",
    "group_summaries",
]
