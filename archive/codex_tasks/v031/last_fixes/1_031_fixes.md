# Task: Finish the 0.3.1 documentation and release-gate cleanup

## Goal

Close the remaining 0.3.1 gaps after the documentation/stabilization pass.

This task explicitly excludes:

* SDPLab-specific examples or downstream validation;
* ecosystem/library comparison work.

The 0.3.1 worked example should be SpaceCore-native. It only needs to show spaces, LinOps, geometry, an algorithm, and an independent direct reference implementation.

## Current problems to fix

The previous 0.3.1 pass added substantial documentation and a weighted Tikhonov example, but several release-gate issues remain:

1. Some old notebooks in `tutorials/` still describe stale API concepts, especially old concrete `VectorSpace` usage.
2. `tutorials/README.md` is outdated and lists only the first five notebooks.
3. The regularized OT tutorial is marked as retained/unvalidated. It should either be revalidated or clearly moved out of the 0.3.1 release gate.
4. Release verification is described in release notes, but there is no single obvious script/command sequence that performs the release-candidate check.
5. Documentation should not imply that 0.3.1 requires an SDPLab-driven example. That requirement was removed.

## Required changes

### 1. Update stale notebooks

Review all notebooks under `tutorials/`:

* `1_BackendOps.ipynb`
* `2_Context.ipynb`
* `3_Space.ipynb`
* `4_LinOp.ipynb`
* `5_Conversion_Policy.ipynb`
* `6_Regularized_Opt_Transport.ipynb`
* `7_Quadratic_Program.ipynb`
* `8_Linalg_MatrixFree.ipynb`
* `9_Linalg_Comparison.ipynb`
* `weighted_tikhonov.ipynb`

Fix stale API language and code.

In particular:

* Do not describe `VectorSpace` as the concrete dense array space.
* Use `DenseCoordinateSpace` for general dense arrays.
* Use `DenseVectorSpace` only for one-dimensional dense coordinate vectors.
* Keep `VectorSpace` as the abstract linear-space layer.
* Check that all examples use current LinOp, Functional, Context, and linalg APIs.
* Remove or update outputs that no longer match current code.

Done condition:

```bash
jupyter nbconvert --to notebook --execute tutorials/*.ipynb --inplace
```

or an equivalent scripted notebook execution must pass for the notebooks that are part of the active tutorial set.

### 2. Update `tutorials/README.md`

Replace the current short folder listing with a complete tutorial index.

It must include every notebook in `tutorials/`, with:

* one-sentence purpose;
* status: active, advanced, retained, or deprecated;
* required optional dependencies, if any;
* recommended reading order.

The README must match `docs/source/tutorials/index.rst`.

### 3. Decide the status of the regularized OT tutorial

For `tutorials/6_Regularized_Opt_Transport.ipynb` and `docs/source/tutorials/regularized_ot.rst`, choose one of two paths.

Preferred path:

* re-execute it;
* fix broken code;
* list required extras, likely `jax`, `optax`, and plotting dependencies;
* keep it as an advanced active tutorial.

Fallback path:

* mark it explicitly as retained but not part of the 0.3.1 release gate;
* move release-gate language away from it;
* ensure docs do not claim it was verified if it was not.

Do not leave ambiguous language such as “should be revalidated later” inside the 0.3.1 release baseline.

### 4. Add a release-candidate verification script

Add one script, for example:

```text
scripts/verify_release_candidate.sh
```

The script should run the core release checks in one place.

It should include at least:

```bash
python -m pip install -e ".[dev,docs,examples]"
pytest tests/ spacecore/ -x -q
ruff check .
python scripts/docstring_audit.py
python scripts/api_audit.py
sphinx-build -W -b html docs/source docs/source/_build/html
python -m build
python -m twine check dist/*
```

If notebook execution is supported, include it or document why it is separate.

The script should be simple, readable, and safe to run locally.

Also add a short section to `docs/source/release_notes.rst` or `docs/dev/0.3.1-docs-inventory.md` saying that this script is the source of truth for the release-candidate gate.

### 5. Remove SDPLab from the 0.3.1 task/gate language

Search docs and release notes for 0.3.1 language that implies SDPLab is required.

Allowed:

* saying that SDPLab validation is out of scope;
* mentioning SDPLab in future release plans if relevant.

Not allowed:

* saying that the 0.3.1 worked example must be SDPLab-driven;
* treating SDPLab validation as a release gate.

The accepted 0.3.1 example is the SpaceCore-native weighted Tikhonov example, provided it remains tested against an independent dense NumPy reference.

### 6. Strengthen the weighted Tikhonov example gate

Check:

* `examples/weighted_tikhonov.py`
* `tutorials/weighted_tikhonov.ipynb`
* `docs/source/tutorials/weighted_tikhonov.rst`
* related tests

Ensure the example demonstrates:

* a non-Euclidean domain or codomain;
* metric adjoint behavior, not merely Euclidean transpose behavior;
* SpaceCore LinOp algebra;
* a SpaceCore algorithm, preferably CG;
* comparison with a direct dense NumPy reference.

Add or tighten tests if any of these are only described but not checked.

### 7. Final verification

Run:

```bash
python -m pip install -e ".[dev,docs,examples]"
pytest tests/ spacecore/ -x -q
ruff check .
sphinx-build -W -b html docs/source docs/source/_build/html
```

If optional notebook dependencies are installed, also run the active notebooks.

## Acceptance criteria

The task is complete when:

* no active tutorial notebook uses stale 0.2.x/early-0.3.0 API language;
* `tutorials/README.md` is complete and agrees with the Sphinx tutorial index;
* regularized OT is either revalidated or explicitly excluded from the 0.3.1 gate;
* one release-candidate verification script exists and is documented;
* 0.3.1 docs do not require SDPLab;
* weighted Tikhonov remains the official 0.3.1 worked example;
* tests, ruff, and strict Sphinx docs build pass.
