# Contributing

SpaceCore is a mathematical software library for typed vector spaces, operators,
functionals, backend-independent array code, and geometry-aware algorithms. It
gives mathematical structure to objects; it does not hide problem-specific
mathematics from contributors.

## Setup

Use [docs/dev/contributing/setup.md](docs/dev/contributing/setup.md) for the
full environment setup. The minimal contributor workflow is:

```bash
pip install -e ".[dev]"
pytest --co -q
pytest tests/ -x -q
ruff check .
```

Optional JAX, Torch, and CuPy backend instructions live in the detailed setup
guide.

## Architecture

Read [docs/dev/contributing/architecture.md](docs/dev/contributing/architecture.md)
before changing core behavior. It explains the backend, context, space, LinOp,
functional, linalg, batching, and cross-cutting infrastructure layers.

## Prerequisites

Read [docs/dev/contributing/prerequisites.md](docs/dev/contributing/prerequisites.md)
to judge the mathematical background needed for a change. Docstring and test
fixes may need little background; geometry, adjoint, spectral-method, and linalg
changes require mathematical review. Mathematical correctness is part of
contribution review even when tests pass.

## Process

Follow [docs/dev/contributing/process.md](docs/dev/contributing/process.md) for
branch, PR, and review expectations. PRs should include tests, updated docs or
docstrings when relevant, a changelog entry under `[Unreleased]`, and a
mathematical invariant statement for changes touching geometry, adjoints,
spectral methods, fields, or batching.

## Beginner-Safe Issues

Start with
[good-first-issue](https://github.com/Pavlo3P/SpaceCore/labels/good-first-issue).
That label is reserved for issues with a specific file to change, an example to
follow, and a concrete done condition.

## Current Project State

Check [docs/dev/current.md](docs/dev/current.md) before opening design-heavy
PRs. Unsettled questions should not be implemented without maintainer agreement.
