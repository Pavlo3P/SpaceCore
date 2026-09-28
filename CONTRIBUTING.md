# Contributing

## Setup and checks

Requires Python 3.11+.

```bash
pip install -e ".[dev]"
pytest --co -q
pytest tests/ -x -q
ruff check .
```

See [setup](docs/dev/contributing/setup.md) for optional backends and focused tests,
and [docs/README.md](docs/README.md) to build the documentation.

## Before opening a PR

- Add tests for changed behavior and update relevant docs or docstrings.
- Add a changelog entry under `[Unreleased]`.
- For geometry, adjoints, spectral methods, scalar fields, or batching changes,
  state the mathematical invariant the change preserves. Passing tests alone
  does not establish mathematical correctness.
- Check [current project decisions](docs/dev/current.md) before design changes;
  unresolved questions need maintainer agreement.

Read the [architecture](docs/dev/contributing/architecture.md) before changing
core behavior. See [prerequisites](docs/dev/contributing/prerequisites.md) for
mathematical background and [process](docs/dev/contributing/process.md) for
branch, review, and release rules.

## First contribution

Choose a [good-first-issue](https://github.com/Pavlo3P/SpaceCore/labels/good-first-issue).
Each should name a file, an example to follow, and a clear completion condition.

## Typing regression checks

With the `jax`, `torch`, and `dev` extras installed:

```bash
pyright --project tests/typing/pyrightconfig.json
mypy --follow-imports=silent --warn-unused-ignores tests/typing/*.py
```

For a NumPy-only install, pass `tests/typing/arrays.py` to both commands
(keep the Pyright project option). These consumer tests check inferred result
classes, native array interoperability, and intentional invalid calls. The
unused-ignore checks ensure errors cannot disappear through `Any`. Pytest
excludes this directory because its negative examples must never execute.
See [the typing investigation](docs/dev/backend_typing.md) for the design and
remaining limits.
