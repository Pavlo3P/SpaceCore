# Docstring migration progress

Baseline (2026-05-27):

- Ruff pydocstyle: 60 `D` violations from `ruff check --select D spacecore/`.
- Numpydoc validation: 133 actionable issues from `python scripts/docstring_audit.py`
  after the Phase 0 allow-list (`ES01`, `EX01`, `SA01`, `GL08`).
- Numpydoc validation, raw: 306 issues from
  `python scripts/docstring_audit.py --include-allowed`.
- Doctest: 0 doctest examples collected from
  `pytest --doctest-modules spacecore/ -x` under the initial ignore list.

Notes:

- The installed `numpydoc` command validates import paths, not package
  directories, so `scripts/docstring_audit.py` records and reports the public
  SpaceCore API baseline.
- Ruff docstring rules are run as a separate non-blocking CI warning while the
  migration is in progress, so the existing strict `ruff check .` step remains
  unchanged.
