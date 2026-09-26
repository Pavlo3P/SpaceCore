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
