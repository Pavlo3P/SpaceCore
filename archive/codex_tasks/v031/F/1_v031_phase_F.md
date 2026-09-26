# Task: Complete SpaceCore 0.3.1 Phase F release-candidate checks

## Goal

Complete the final release-candidate verification for SpaceCore `0.3.1`.

This task is the last gate before release. It should not introduce new features. It should verify that the repository is internally consistent, installable from clean artifacts, documented, and ready for publication.

The task must cover:

* execution of all tutorials and the end-to-end worked example;
* public API documentation audit;
* linting, tests, audits, and strict documentation build;
* source distribution and wheel build;
* inspection of release artifacts;
* final changelog and release notes cleanup.

Do not perform SDPLab-specific downstream validation in this task. If any remaining failure is caused by downstream SDPLab integration, record it separately as a follow-up issue and keep this release-candidate task focused on SpaceCore itself.

---

## Scope

This task corresponds to Phase F of the `0.3.1` plan.

Phase F consists of:

* F1: execute all tutorials and the worked example;
* F2: run public API documentation audit;
* F3: run lint, tests, audits, and docs build;
* F4: build and inspect release artifacts;
* F5: finalize changelog and release notes.

The expected output is a release-candidate branch where all checks pass from a clean environment.

---

## Out of scope

Do not add new abstractions.

Do not redesign APIs.

Do not add new solvers.

Do not optimize performance unless a trivial fix is required to make a release check pass.

Do not run or fix SDPLab-specific downstream migration checks.

Do not silently weaken tests, skip examples, or remove documentation warnings just to make the release pass. If something is intentionally excluded, explain why in a release note or follow-up issue.

---

## Step 1: Start from a clean environment

Create a fresh virtual environment and install the package from the repository.

Suggested commands:

```bash
python -m venv .venv-rc
source .venv-rc/bin/activate

python -m pip install --upgrade pip build twine
python -m pip install -e ".[dev,docs]"
```

If the project uses a different extras layout, inspect `pyproject.toml` and use the correct development/documentation extras.

Record the Python version used. Prefer Python `3.11` or `3.12`, matching the supported release matrix.

Done condition:

* package imports successfully;
* development dependencies install cleanly;
* no hidden dependency on local editable state except the intended editable install.

---

## Step 2: Execute all tutorials

Find all tutorial files in the documentation tree, examples tree, notebooks tree, or any equivalent tutorial directory.

Run every tutorial from start to finish.

Acceptable tutorial formats include:

* `.py` scripts;
* `.ipynb` notebooks;
* documentation examples executed through Sphinx;
* examples referenced from README or documentation landing pages.

For notebooks, use a deterministic execution command such as:

```bash
jupyter nbconvert --to notebook --execute path/to/tutorial.ipynb --output /tmp/tutorial-check.ipynb
```

or the project’s existing notebook-test command if one exists.

For Python tutorials, run:

```bash
python path/to/tutorial.py
```

For documentation examples, use the strict docs build in Step 5.

While checking tutorials, verify that they use the current `0.3.x` API:

* no stale `BatchSpace`;
* no stale `Space.batch`;
* no stale `batch_space=`;
* no stale context-policy APIs;
* no stale dtype-policy APIs;
* no obsolete `VectorSpace` meaning if the hierarchy was renamed;
* no outdated examples of adjoints or weighted inner products;
* no examples relying on private implementation details.

If a tutorial is obsolete but intentionally kept for historical context, move it out of the active tutorial path or mark it clearly as historical so it is not presented as current usage.

Done condition:

* every active tutorial executes successfully;
* every tutorial imports only public APIs unless explicitly testing internals;
* all tutorial outputs are deterministic enough for CI;
* every failure is either fixed or converted into a narrowly described follow-up issue.

---

## Step 3: Execute the end-to-end worked example

Run the worked example selected for `0.3.1`.

The example should exercise the main SpaceCore abstractions together:

* spaces;
* elements;
* linear operators;
* adjoints;
* at least one algorithm or solver;
* documentation explaining the mathematical structure.

The example should be run from a clean environment, not from an interactive state.

Verify that the example:

* constructs spaces explicitly;
* declares domain and codomain correctly;
* uses the intended inner product or geometry;
* does not silently flatten structured elements unless that is the point of the example;
* checks output against a reference solution or independently computed baseline;
* documents numerical tolerances;
* does not depend on unavailable optional libraries unless the optional dependency is declared and guarded.

Do not require SDPLab for this worked example. If the current worked example depends on SDPLab, either:

1. replace it with a SpaceCore-only example for the release candidate, or
2. split the downstream-dependent part into a separate non-blocking follow-up.

Done condition:

* the worked example executes from start to finish;
* the example validates its result numerically;
* the example is integrated into documentation or CI;
* the example uses SpaceCore abstractions honestly, without bypassing the library’s object model.

---

## Step 4: Audit public API documentation

Run a public API documentation audit.

The audit should answer:

* Which modules expose public classes, functions, or constants?
* Which public objects lack docstrings?
* Which docstrings are stale?
* Which docstrings mention removed APIs?
* Which docstrings contain mathematically false or misleading claims?
* Which public objects are intentionally undocumented?
* Which private objects are accidentally exposed?

At minimum, audit:

* `spacecore.space`;
* `spacecore.linop`;
* `spacecore.functional`;
* `spacecore.linalg`;
* `spacecore.backend`;
* `spacecore.context`;
* top-level `spacecore.__init__`.

For every public object, check:

* signature;
* parameters;
* return type or returned mathematical object;
* domain/codomain when applicable;
* shape conventions;
* dtype/backend assumptions;
* error conditions;
* whether the operation is JIT/autodiff-compatible if the docstring claims so;
* whether non-Euclidean geometry is handled correctly.

Pay special attention to adjoint documentation. The documentation must distinguish:

* coordinate conjugate transpose;
* Euclidean adjoint;
* metric adjoint/Riesz-represented adjoint.

For a linear operator (A : X \to Y), the metric adjoint should be described as an operator (A^\sharp : Y \to X), with the defining relation

```text
<Ax, y>_Y = <x, A^sharp y>_X.
```

Do not document `rapply` as merely “transpose multiplication” unless the relevant spaces are Euclidean coordinate spaces.

Done condition:

* no unintentionally undocumented public API remains;
* any intentionally undocumented public object is listed with justification;
* stale API references are removed;
* docstrings and generated API pages agree with the current implementation.

---

## Step 5: Run lint, tests, audits, and strict docs build

Run the full quality gate used for release.

Use the project’s configured tools. Likely commands include some subset of:

```bash
ruff check .
ruff format --check .
pytest
pytest --doctest-modules
mypy .
python -m build
sphinx-build -W -b html docs docs/_build/html
```

Adjust the commands to match the project configuration in `pyproject.toml`, `tox.ini`, `noxfile.py`, or CI files.

The strict documentation build must fail on warnings. Fix warnings rather than suppressing them unless the warning is known to be false-positive and the suppression is local and documented.

The test suite must be run as a whole, not only file-by-file. If the monolithic test suite has instability, diagnose whether the cause is:

* global mutable backend/context state;
* random seeds;
* test order dependence;
* JAX/Torch compilation side effects;
* leaked warnings;
* floating-point tolerance too strict;
* optional dependency detection;
* path/import problems.

Do not skip flaky tests without creating a precise issue explaining the cause.

Done condition:

* lint passes;
* formatting check passes;
* tests pass as a full suite;
* doctests pass if configured;
* strict docs build passes;
* no new warning is ignored without explanation.

---

## Step 6: Build release artifacts

Build both source distribution and wheel.

Suggested command:

```bash
rm -rf dist build *.egg-info
python -m build
```

Inspect the artifacts:

```bash
python -m twine check dist/*
```

Also inspect the contents manually:

```bash
tar -tzf dist/*.tar.gz | less
python -m zipfile -l dist/*.whl | less
```

Verify that the source distribution includes all necessary files:

* package source;
* pyproject/build metadata;
* license;
* README;
* changelog/release notes if intended;
* documentation files needed for source users;
* py.typed if the package declares typing support;
* test files if the project ships them in sdist.

Verify that the wheel does not include unwanted files:

* local caches;
* build artifacts;
* notebooks with large outputs unless intentional;
* private scratch files;
* generated documentation build output unless intentionally packaged.

Done condition:

* `python -m build` succeeds;
* `twine check dist/*` passes;
* wheel contents are minimal and correct;
* sdist contents are complete;
* version metadata is correct and consistent.

---

## Step 7: Test clean installation from built wheel

Create a second clean environment and install from the built wheel, not from the repository.

Suggested commands:

```bash
python -m venv .venv-wheel-check
source .venv-wheel-check/bin/activate

python -m pip install --upgrade pip
python -m pip install dist/*.whl
python -c "import spacecore; print(spacecore.__version__)"
```

Then run a minimal smoke test using only installed package files.

The smoke test should check:

* top-level import;
* construction of a basic dense coordinate/vector space;
* construction of a product space if public;
* construction and application of a linear operator;
* adjoint application in a Euclidean case;
* one linalg routine on a small deterministic problem.

Example smoke-test shape:

```python
import spacecore as sc

# Construct small spaces.
# Construct an operator A : X -> Y.
# Apply A and its adjoint.
# Solve or run one tiny linalg routine.
# Assert expected numerical result.
```

Adapt this to the actual public API.

Done condition:

* package installs from wheel in a clean environment;
* imports succeed;
* smoke test passes without relying on repository files.

---

## Step 8: Finalize changelog and release notes

Update the changelog and release notes for `0.3.1`.

The notes should clearly distinguish:

* fixes;
* documentation improvements;
* test/CI improvements;
* examples/tutorials;
* known limitations;
* any backward-incompatible cleanup if present.

The release notes should mention the stabilization scope. They should not overclaim new features.

Include known limitations honestly, for example:

* solver set remains intentionally narrow;
* some external solver adapters are planned/evaluated but not part of this release unless implemented;
* ecosystem-positioning note is a living design note if it is not complete;
* optional backend support depends on installed optional dependencies;
* any tutorial intentionally excluded from CI should be named.

If any release blocker remains, do not mark the release candidate complete. Instead write a precise blocker issue.

Done condition:

* changelog has a `0.3.1` section;
* release notes summarize the actual merged scope;
* known limitations are explicit;
* no planned-but-unimplemented feature is described as shipped.

---

## Step 9: Final release-candidate report

At the end, add a short release-candidate report to the PR description or a checked-in release checklist.

The report should include:

```text
SpaceCore 0.3.1 release-candidate report

Environment:
- Python:
- OS:
- Backend dependencies installed:
- Optional dependencies installed:

Checks:
- Tutorials:
- Worked example:
- Public API documentation audit:
- Lint:
- Formatting:
- Tests:
- Doctests:
- Strict docs build:
- Build artifacts:
- Twine check:
- Clean wheel installation:
- Smoke test:

Artifacts:
- sdist:
- wheel:

Known limitations:
- ...

Follow-up issues:
- ...
```

Done condition:

* the release-candidate state is reproducible;
* the report states exactly which checks passed;
* any remaining non-blocking work is listed as a follow-up;
* the branch is ready for tag/release review.

---

## Acceptance criteria

This task is complete only when all of the following are true:

1. All active tutorials execute successfully from a clean environment.
2. The end-to-end worked example executes and validates its result.
3. Public API documentation has been audited.
4. No unintentionally undocumented public API remains.
5. Lint and formatting checks pass.
6. The full test suite passes.
7. Strict documentation build passes without warnings.
8. Source distribution and wheel build successfully.
9. Release artifacts pass metadata checks.
10. The built wheel installs in a clean environment.
11. A minimal smoke test passes against the installed wheel.
12. Changelog and release notes accurately describe `0.3.1`.
13. Known limitations are recorded honestly.
14. No SDPLab-specific downstream validation is required for completion of this task.

---

## Expected deliverables

Submit one PR containing:

* fixes required to make Phase F pass;
* updated release notes/changelog;
* any documentation corrections discovered during audit;
* release-candidate report;
* follow-up issues for any non-blocking problems.

The PR should not contain unrelated refactors or new features.
