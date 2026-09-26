# Codex Task: Write the root `CONTRIBUTING.md`

## Goal

Create or replace the repository-root `CONTRIBUTING.md` with a concise contributor entry point for SpaceCore.

The file should be short enough to read quickly, but complete enough that a new contributor knows:

* what kind of project SpaceCore is;
* what mathematical background helps;
* how to set up the development environment;
* where to find architecture, prerequisite, and process documentation;
* how to find beginner-safe issues;
* what the project is currently working on.

## File to edit

```text
CONTRIBUTING.md
```

## Required content

Write the file as a one-page contributor guide with these sections.

### 1. Project description

Add a short opening paragraph explaining that SpaceCore is a mathematical software library for typed vector spaces, operators, functionals, backend-independent array code, and geometry-aware algorithms.

The paragraph must be honest: SpaceCore gives mathematical structure to objects; it does **not** hide problem-specific mathematics from contributors.

### 2. Setup link

Add a section pointing to:

```text
docs/dev/contributing/setup.md
```

The root file should not duplicate every command, but it should mention the expected minimal workflow:

```bash
pip install -e ".[dev]"
pytest --co -q
pytest tests/ -x -q
ruff check .
```

If the detailed setup guide includes optional backends, say that JAX/Torch/CuPy instructions live there.

### 3. Architecture link

Add a section pointing to:

```text
docs/dev/contributing/architecture.md
```

Briefly say that it explains the backend, context, space, LinOp, functional, linalg, batching, and cross-cutting infrastructure layers.

### 4. Prerequisites link

Add a section pointing to:

```text
docs/dev/contributing/prerequisites.md
```

State that different contribution types require different mathematical background. For example, docstring/test fixes may need little background, while geometry, adjoints, spectral methods, and linalg changes require mathematical review.

### 5. Process link

Add a section pointing to:

```text
docs/dev/contributing/process.md
```

Mention that PRs should include tests, updated docs/docstrings when relevant, a changelog entry under `[Unreleased]`, and a mathematical invariant statement for changes touching geometry, adjoints, spectral methods, fields, or batching.

### 6. Beginner-safe issues

Add a section linking to the repository’s `good-first-issue` label.

Use a relative or GitHub URL if the repository URL is already known in project metadata. Otherwise use a placeholder that is easy to replace:

```markdown
[good-first-issue](https://github.com/Pavlo3P/SpaceCore/labels/good-first-issue)
```

Explain that this label is reserved for issues with a specific file, example to follow, and concrete done condition.

### 7. Current project state

Add a section linking to:

```text
docs/dev/current.md
```

Explain that contributors should check this before opening design-heavy PRs, because unsettled questions should not be implemented without maintainer agreement.

## Style requirements

* Keep the file approximately one page.
* Use plain Markdown.
* Keep tone direct and practical.
* Do not oversell the project.
* Do not describe SpaceCore as a general-purpose replacement for NumPy, SciPy, PyTorch, or JAX.
* Do not promise that contributors can avoid the mathematics.
* Avoid long duplicated command blocks; link to the detailed setup guide instead.

## Consistency checks

After editing, verify:

```bash
python -m pip install -e ".[dev]"
pytest --co -q
ruff check .
```

Also check all links in `CONTRIBUTING.md` manually or through the project’s docs/link checker if available.

## Done condition

The task is complete when:

* `CONTRIBUTING.md` exists at the repository root;
* it is approximately one page;
* it links to setup, architecture, prerequisites, process, `good-first-issue`, and `docs/dev/current.md`;
* install commands match the detailed setup guide;
* no stale `[dev,docs]` install recipe remains if the current guide uses `.[dev]`;
* the file clearly states that mathematical correctness is part of contribution review;
* `pytest --co -q` and `ruff check .` pass.
