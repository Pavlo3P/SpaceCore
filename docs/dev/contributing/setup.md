# Contributor setup

SpaceCore supports Python 3.11 and newer. The minimal contributor setup uses the core NumPy/SciPy dependency set plus development tools.

## Clone and install

```bash
git clone https://github.com/Pavlo3P/SpaceCore
cd SpaceCore
pip install -e ".[dev]"
```

`.[dev]` is the minimal contributor install. It installs SpaceCore in editable mode, the default NumPy-backed runtime dependencies, and contributor tools such as pytest and Ruff.

Use the fuller optional-backend install when you are working on backend-specific behavior:

```bash
pip install -e ".[jax,torch,dev]"
```

The `[jax,torch,dev]` install is for contributors changing JAX or Torch paths. Optional backend tests should skip cleanly when JAX or Torch is absent, so they are not required for ordinary NumPy-backed changes.

## Verify the environment

Start with collection. It is faster than the full suite and catches collection-time import errors before running tests:

```bash
pytest --co -q
```

Run the normal test gate:

```bash
pytest tests/ -x -q
```

Run a focused space-layer check when working on spaces, geometry, checks, or batching:

```bash
pytest tests/spaces/ -v
```

Run linting:

```bash
ruff check .
```

The minimal `.[dev]` install should run the NumPy-backed test suite. Tests for optional backends should either run when their packages are installed or skip without causing the NumPy contributor workflow to fail.
