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
run "$PYTHON_BIN" scripts/docstring_audit.py
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
    tutorials/1_BackendOps.ipynb \
    tutorials/2_Context.ipynb \
    tutorials/3_Space.ipynb \
    tutorials/4_LinOp.ipynb \
    tutorials/5_Conversion_Policy.ipynb \
    tutorials/7_Quadratic_Program.ipynb \
    tutorials/8_Linalg_MatrixFree.ipynb \
    tutorials/9_Linalg_Comparison.ipynb \
    tutorials/weighted_tikhonov.ipynb \
    --inplace
else
  printf '\n==> Skipping active tutorial notebooks because SPACECORE_VERIFY_NOTEBOOKS=0\n'
fi
