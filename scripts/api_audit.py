from __future__ import annotations

import argparse
import ast
import json
from dataclasses import dataclass
from pathlib import Path
import re
from typing import Iterable


REMOVED_CALLS = {
    "VectorSpace",
    "ProductInnerProductSpace",
    "ProductStarSpace",
    "ProductJordanAlgebraSpace",
    "ProductEuclideanJordanAlgebraSpace",
    "StackedInnerProductSpace",
    "StackedStarSpace",
    "StackedJordanAlgebraSpace",
    "StackedEuclideanJordanAlgebraSpace",
}

REMOVED_METHODS = {"eigh"}

SKIP_DIRS = {
    ".git",
    ".idea",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".venv",
    ".venv312",
    "__pycache__",
    "build",
    "dist",
    "html",
    "codex_tasks",
}

SKIP_FILES = {
    "playpit.ipynb",
    "sphere_descent.ipynb",
    "sphere_descent_fixed.ipynb",
    "test_speed_good.ipynb",
}

TEXT_SUFFIXES = {".md", ".rst", ".txt"}
PYTHON_SUFFIXES = {".py"}
NOTEBOOK_SUFFIXES = {".ipynb"}
MIGRATION_DOCS = {"CHANGELOG.md", "release_notes.rst"}


@dataclass(frozen=True)
class Finding:
    path: Path
    location: str
    pattern: str
    detail: str


def _iter_files(root: Path) -> Iterable[Path]:
    for path in root.rglob("*"):
        if any(part in SKIP_DIRS or part.startswith(".venv") for part in path.parts):
            continue
        if not path.is_file():
            continue
        if path.name in SKIP_FILES:
            continue
        if path.suffix in PYTHON_SUFFIXES | NOTEBOOK_SUFFIXES | TEXT_SUFFIXES:
            yield path


def _call_name(node: ast.AST) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return None


def _is_backend_eigh_call(node: ast.Attribute) -> bool:
    value: ast.AST = node.value
    while isinstance(value, ast.Attribute):
        if value.attr in {"ops", "xp", "linalg", "torch"}:
            return True
        value = value.value
    return isinstance(value, ast.Name) and value.id in {"ops", "xp", "torch"}


def _scan_python(path: Path) -> list[Finding]:
    text = path.read_text(encoding="utf-8")
    try:
        tree = ast.parse(text, filename=str(path))
    except SyntaxError as exc:
        return [Finding(path, f"line {exc.lineno}", "syntax", str(exc))]

    findings: list[Finding] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            name = _call_name(node.func)
            if name == "VectorSpace" and not node.args and not node.keywords:
                continue
            if name in REMOVED_CALLS:
                findings.append(
                    Finding(path, f"line {node.lineno}", f"{name}(", "removed constructor")
                )
            if (
                isinstance(node.func, ast.Attribute)
                and node.func.attr in REMOVED_METHODS
                and not _is_backend_eigh_call(node.func)
            ):
                findings.append(
                    Finding(
                        path,
                        f"line {node.lineno}",
                        f".{node.func.attr}(",
                        "removed method",
                    )
                )
    return findings


def _scan_text(path: Path) -> list[Finding]:
    if path.name in MIGRATION_DOCS:
        return []
    patterns = [(name, re.compile(rf"\b{name}\s*\(")) for name in sorted(REMOVED_CALLS)]
    patterns.extend((f".{name}", re.compile(rf"\.{name}\s*\(")) for name in REMOVED_METHODS)

    findings: list[Finding] = []
    for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        for label, pattern in patterns:
            if pattern.search(line):
                findings.append(Finding(path, f"line {lineno}", label, line.strip()))
    return findings


def _scan_notebook(path: Path) -> list[Finding]:
    data = json.loads(path.read_text(encoding="utf-8"))
    findings: list[Finding] = []
    for cell_index, cell in enumerate(data.get("cells", []), start=1):
        if cell.get("cell_type") != "code":
            continue
        source = "".join(cell.get("source", []))
        try:
            tree = ast.parse(source, filename=f"{path}:cell {cell_index}")
        except SyntaxError as exc:
            findings.append(Finding(path, f"cell {cell_index}", "syntax", str(exc)))
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                name = _call_name(node.func)
                if name == "VectorSpace" and not node.args and not node.keywords:
                    continue
                if name in REMOVED_CALLS:
                    findings.append(
                        Finding(path, f"cell {cell_index}", f"{name}(", "removed constructor")
                    )
                if (
                    isinstance(node.func, ast.Attribute)
                    and node.func.attr in REMOVED_METHODS
                    and not _is_backend_eigh_call(node.func)
                ):
                    findings.append(
                        Finding(path, f"cell {cell_index}", f".{node.func.attr}(", "removed method")
                    )
    return findings


def scan(root: Path) -> list[Finding]:
    findings: list[Finding] = []
    for path in _iter_files(root):
        if path.suffix in PYTHON_SUFFIXES:
            findings.extend(_scan_python(path))
        elif path.suffix in NOTEBOOK_SUFFIXES:
            findings.extend(_scan_notebook(path))
        else:
            findings.extend(_scan_text(path))
    return findings


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit removed SpaceCore 0.3.0 APIs.")
    parser.add_argument("roots", nargs="*", type=Path, default=[Path.cwd()])
    args = parser.parse_args()

    all_findings: list[Finding] = []
    for root in args.roots:
        if root.exists():
            all_findings.extend(scan(root.resolve()))
        else:
            all_findings.append(Finding(root, "-", "missing", "path does not exist"))

    for finding in all_findings:
        print(f"{finding.path}:{finding.location}: {finding.pattern}: {finding.detail}")
    return 1 if all_findings else 0


if __name__ == "__main__":
    raise SystemExit(main())
