# Task: Fulfill Phase B — Stabilization, CI validation, packaging, and baseline performance for SpaceCore 0.3.1

## Goal

Complete the non-SDPLab part of Phase B for the SpaceCore 0.3.1 release.

This task is about stabilizing the repository itself: tests, CI, documentation build, Python-version compatibility, package build, clean installation, and performance baseline capture.

Do not perform the SDPLab audit or SDPLab migration work in this task.

## Scope

Implement and verify the following Phase B items:

* B1: Run the full test suite with a timing report.
* B2: Diagnose monolithic-suite instability.
* B3: Verify the full suite under Python 3.11.
* B4: Verify the full suite under Python 3.12.
* B5: Verify strict documentation build.
* B10: Capture a representative performance baseline.
* B11: Verify wheel build and clean installation.

Explicitly exclude:

* B6: Run API audit against SDPLab.
* B7: Fix Space-related SDPLab migration failures.
* B8: Fix LinOp/Functional SDPLab migration failures.
* B9: Fix linalg/remaining SDPLab migration failures.

If any SDPLab-related failure appears indirectly, record it in a note, but do not fix it here.

## General constraints

Keep the task focused on release stabilization. Avoid API redesign unless a failure proves that the current API is internally inconsistent.

Prefer small, mechanical fixes over broad refactors.

Do not hide failures by weakening tests unless the test is clearly obsolete. If a test is obsolete, document why and update it to the current 0.3.x API.

All changes must preserve public API intent for SpaceCore 0.3.1.

## Step 1 — Run full test suite with timing report

Run the complete test suite locally.

Use pytest timing output, for example:

```bash
pytest --durations=25
```

or the repository’s existing test command if it already includes timing.

Record:

* total number of tests;
* total runtime;
* slowest 25 tests;
* tests taking unexpectedly long;
* tests that pass individually but fail in the full suite;
* warnings that may become CI failures later.

If the suite has markers, also run relevant grouped subsets, for example:

```bash
pytest tests/spaces --durations=15
pytest tests/linop --durations=15
pytest tests/linalg --durations=15
pytest tests/functional --durations=15
```

Expected result:

* the full suite either passes, or every failure is categorized and assigned to a precise fix;
* slow tests are identified, but optimization is not required unless a test stalls or is pathologically slow.

## Step 2 — Diagnose monolithic-suite instability

Check whether tests behave differently when run:

1. individually;
2. by module;
3. as the full suite;
4. in random order if pytest-randomly or similar tooling is available.

Look especially for:

* leaked global context/backend state;
* tests mutating shared fixtures;
* dtype/backend assumptions leaking between tests;
* JAX/Torch/CuPy availability assumptions;
* tests relying on execution order;
* global random seeds;
* cached operators or spaces reused across incompatible contexts;
* tolerance differences between NumPy, JAX, Torch, and CuPy backends.

If instability is found, fix the cause.

If the cause is too large for this task, create a precise follow-up issue describing:

* failing command;
* minimal reproduction;
* suspected cause;
* affected modules;
* whether it blocks 0.3.1.

Done condition:

* the full suite is stable, or the remaining instability is documented as a narrow issue with reproduction commands.

## Step 3 — Verify Python 3.11

Create or use a clean Python 3.11 environment.

Install the package with test dependencies.

Example:

```bash
python3.11 -m venv .venv311
source .venv311/bin/activate
python -m pip install -U pip
python -m pip install -e ".[test,docs]"
pytest --durations=25
```

Adapt extras to the actual project configuration.

Fix Python 3.11 failures unless they are caused by unavailable optional dependencies. Optional-dependency failures should be skipped cleanly with explicit skip reasons.

Done condition:

* full test suite passes on Python 3.11;
* any optional backend skips are intentional and visible.

## Step 4 — Verify Python 3.12

Repeat the clean-environment validation under Python 3.12.

Example:

```bash
python3.12 -m venv .venv312
source .venv312/bin/activate
python -m pip install -U pip
python -m pip install -e ".[test,docs]"
pytest --durations=25
```

Fix Python 3.12-specific issues.

Pay attention to:

* typing changes;
* deprecated standard-library APIs;
* dependency version constraints;
* build backend compatibility;
* doc build warnings caused by newer Sphinx or docutils.

Done condition:

* full test suite passes on Python 3.12;
* CI configuration includes Python 3.11 and Python 3.12, or there is a clear note explaining where that is configured.

## Step 5 — Verify strict documentation build

Run the documentation build in strict mode.

Use the repository’s documented docs command if present. Otherwise use something like:

```bash
sphinx-build -W --keep-going -b html docs docs/_build/html
```

Fix all documentation warnings that are treated as errors.

Check especially:

* broken cross-references;
* stale API paths;
* removed classes or methods;
* stale batching references;
* stale dtype/context-policy references;
* malformed math directives;
* invalid doctest examples;
* autosummary/import failures;
* missing modules in the public API docs.

Do not rewrite the full documentation here. This task only needs the strict build to pass.

Done condition:

* strict docs build passes without warnings;
* generated docs do not contain obvious broken API references introduced by 0.3.x changes.

## Step 6 — Capture performance baseline

Capture representative timings without attempting broad optimization.

The goal is to preserve a baseline for future regressions, not to make SpaceCore faster in this task.

Select a small but representative set of operations, preferably covering:

* dense vector space operations;
* product space operations;
* matrix-backed LinOp apply/rapply;
* matrix-free LinOp apply/rapply;
* composition/sum/scaled LinOps;
* inner-product and norm computations;
* one or two linalg routines such as CG, LSQR, Lanczos, or power method;
* at least NumPy backend;
* JAX and Torch if already supported and available in the environment.

For each benchmark, record:

* operation name;
* backend;
* shape/problem size;
* dtype;
* number of runs;
* median time;
* mean time if useful;
* whether JIT compilation time is included or excluded;
* hardware summary.

Do not mix JAX compile time with steady-state execution time unless explicitly labeled.

Recommended output file:

```text
docs/dev/performance_baseline_0.3.1.md
```

or, if the repository has a better location:

```text
benchmarks/baseline_0.3.1.md
```

Keep benchmark code simple. If adding scripts, prefer:

```text
benchmarks/baseline_spacecore.py
```

The benchmark script should be easy to run manually and should not make CI fragile.

Done condition:

* a baseline timing report is committed;
* timings are representative enough to detect future regressions;
* no optimization work is required unless a severe accidental slowdown is discovered.

## Step 7 — Build wheel and source distribution

Verify package build from a clean checkout.

Example:

```bash
python -m pip install -U build twine
python -m build
python -m twine check dist/*
```

Fix packaging errors.

Check that the resulting artifacts include the expected package files and exclude obvious junk:

* no local caches;
* no build artifacts inside the package;
* no accidental large files;
* no missing py.typed file if typing is intended;
* no missing documentation/license files required by project metadata.

Done condition:

* wheel builds;
* source distribution builds;
* `twine check dist/*` passes.

## Step 8 — Verify clean installation and import

Create a fresh virtual environment that does not use editable install.

Install the built wheel:

```bash
python -m venv .venv-wheel-test
source .venv-wheel-test/bin/activate
python -m pip install -U pip
python -m pip install dist/*.whl
```

Then verify basic imports:

```bash
python - <<'PY'
import spacecore as sc
print(sc.__version__)
PY
```

Also verify that the main public modules import:

```bash
python - <<'PY'
import spacecore
import spacecore.space
import spacecore.linop
import spacecore.functional
import spacecore.linalg
PY
```

If the package exposes optional backends, test that missing optional dependencies fail gracefully. Importing SpaceCore without JAX, Torch, or CuPy should not fail unless they are declared hard dependencies.

Done condition:

* wheel installs in a clean environment;
* `import spacecore` works;
* public modules import;
* missing optional dependencies are handled cleanly.

## Step 9 — Update CI if needed

Ensure CI covers the non-SDPLab Phase B requirements.

At minimum, CI should include:

* Python 3.11 test job;
* Python 3.12 test job;
* strict documentation build job;
* packaging/build check job.

If runtime is too high, keep expensive jobs separate from quick PR checks, but ensure they can be run before release.

Do not add unstable GPU CI in this task unless the repository already has stable GPU infrastructure.

Done condition:

* CI expresses the release checks needed for Phase B;
* local commands and CI commands are consistent;
* failing optional backend availability is handled through skips, not silent pass/fail ambiguity.

## Step 10 — Produce final stabilization report

Add a short report, either in the PR description or in a committed developer note.

The report must include:

* test commands run;
* Python versions tested;
* docs build command;
* package build command;
* clean install command;
* summary of failures fixed;
* remaining known issues, if any;
* location of the performance baseline;
* confirmation that SDPLab audit/migration was intentionally excluded.

Suggested structure:

```markdown
# Phase B stabilization report

## Commands run

## Python 3.11 result

## Python 3.12 result

## Documentation build result

## Package build and clean install result

## Performance baseline

## Fixed issues

## Remaining issues

## Explicitly excluded

SDPLab audit and SDPLab migration items B6--B9 were excluded from this task.
```

## Acceptance criteria

This task is complete when all of the following are true:

* The full SpaceCore test suite runs with timing information.
* Monolithic-suite instability is either fixed or documented with a precise reproduction.
* Tests pass under Python 3.11.
* Tests pass under Python 3.12.
* Strict documentation build passes.
* A representative performance baseline is stored.
* Wheel and source distribution build successfully.
* Built wheel installs in a clean environment.
* `import spacecore` and main public module imports work from the clean wheel install.
* CI is updated or confirmed to cover these checks.
* A stabilization report summarizes what was run and what remains.
* No SDPLab audit or SDPLab migration work is included.

## Non-goals

Do not do the following in this task:

* Do not run or fix the SDPLab audit.
* Do not migrate SDPLab.
* Do not redesign SpaceCore APIs.
* Do not add new solver functionality.
* Do not optimize performance beyond fixing accidental severe regressions.
* Do not rewrite broad documentation sections except where needed to make strict docs pass.
* Do not add new backend support.
* Do not introduce broad compatibility layers or external adapters.

## Suggested commit structure

Use small commits if possible:

1. Test-suite stability fixes.
2. Python 3.11/3.12 compatibility fixes.
3. Strict docs-build fixes.
4. Packaging and clean-install fixes.
5. Performance baseline and stabilization report.
6. CI updates.

## Final expected output

A pull request that makes SpaceCore internally release-stable for Phase B, excluding SDPLab-specific validation, and leaves behind enough commands and baseline data that the checks can be repeated before the 0.3.1 release.
