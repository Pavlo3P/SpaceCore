#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON:-python}"

run() {
  printf '\n==> %s\n' "$*"
  "$@"
}

run "$PYTHON_BIN" -m pip install -e ".[dev,docs,examples]"

run "$PYTHON_BIN" -m pytest tests/ spacecore/ -x -q
run "$PYTHON_BIN" -m ruff check .
run "$PYTHON_BIN" -m ruff check --select D spacecore/
run "$PYTHON_BIN" scripts/docstring_audit.py --check
run "$PYTHON_BIN" scripts/api_audit.py
run "$PYTHON_BIN" -m sphinx -W -b html docs/source docs/source/_build/html
run "$PYTHON_BIN" -m build
run "$PYTHON_BIN" -m twine check dist/*

if [[ "${SPACECORE_VERIFY_NOTEBOOKS:-1}" != "0" ]]; then
  run "$PYTHON_BIN" -c "import nbconvert, ipykernel"
  export PYTHONPATH="$ROOT_DIR${PYTHONPATH:+:$PYTHONPATH}"
  run "$PYTHON_BIN" -m nbconvert \
    --to notebook \
    --execute \
    tutorials/01_backend_and_context.ipynb \
    tutorials/02_linear_algebra.ipynb \
    tutorials/03_functionals.ipynb \
    tutorials/04_tree_spaces.ipynb \
    tutorials/05_weighted_tikhonov.ipynb \
    tutorials/06_optimal_transport.ipynb \
    tutorials/07_manifold_descent.ipynb \
    tutorials/08_pdhg_conic_program.ipynb \
    --inplace
else
  printf '\n==> Skipping active tutorial notebooks because SPACECORE_VERIFY_NOTEBOOKS=0\n'
fi
