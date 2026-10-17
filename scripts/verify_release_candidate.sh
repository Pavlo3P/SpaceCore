#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON:-python}"

run() {
  printf '\n==> %s\n' "$*"
  "$@"
}

run "$PYTHON_BIN" -m pip install -e ".[dev,docs]"

run "$PYTHON_BIN" -m pytest tests/ spacecore/ -x -q
run "$PYTHON_BIN" -m ruff check .
run "$PYTHON_BIN" -m ruff check --select D spacecore/
run "$PYTHON_BIN" scripts/docstring_audit.py --check
run "$PYTHON_BIN" scripts/api_audit.py
run "$PYTHON_BIN" -m sphinx -W -b html docs/source docs/source/_build/html
run "$PYTHON_BIN" -m build
run "$PYTHON_BIN" -m twine check dist/*

run "$PYTHON_BIN" scripts/check_tutorials.py
