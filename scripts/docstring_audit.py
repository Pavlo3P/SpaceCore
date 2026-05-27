"""Report numpydoc validation issues for SpaceCore's public API."""

from __future__ import annotations

import argparse
import inspect
from collections.abc import Iterable
from dataclasses import dataclass

from numpydoc.validate import validate

import spacecore

ALLOWED_CODES = frozenset({"ES01", "EX01", "SA01", "GL08"})


@dataclass(frozen=True)
class ValidationIssue:
    """A single numpydoc validation issue."""

    target: str
    code: str
    message: str


def _iter_public_targets() -> Iterable[str]:
    """Yield public import paths exported by the top-level package."""
    for name in getattr(spacecore, "__all__", ()):
        if name.startswith("_"):
            continue
        target = f"spacecore.{name}"
        try:
            obj = getattr(spacecore, name)
        except AttributeError:
            continue
        if inspect.ismodule(obj):
            continue
        yield target


def _validate_target(target: str, *, include_allowed: bool) -> list[ValidationIssue]:
    """Validate one import path and normalize numpydoc's result shape."""
    try:
        result = validate(target)
    except Exception as exc:  # pragma: no cover - defensive reporting path
        return [ValidationIssue(target, "IMPORT", f"{type(exc).__name__}: {exc}")]

    issues = []
    for code, message in result.get("errors", []):
        if not include_allowed and code in ALLOWED_CODES:
            continue
        issues.append(ValidationIssue(target, code, message))
    return issues


def collect_issues(*, include_allowed: bool = False) -> list[ValidationIssue]:
    """Collect numpydoc issues for exported public symbols."""
    issues: list[ValidationIssue] = []
    for target in sorted(set(_iter_public_targets())):
        issues.extend(_validate_target(target, include_allowed=include_allowed))
    return issues


def main() -> int:
    """Run the audit command."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="exit non-zero when validation issues are present",
    )
    parser.add_argument(
        "--max-lines",
        type=int,
        default=200,
        help="maximum number of individual issues to print",
    )
    parser.add_argument(
        "--include-allowed",
        action="store_true",
        help="include issues allowed during the migration baseline",
    )
    args = parser.parse_args()

    issues = collect_issues(include_allowed=args.include_allowed)
    for issue in issues[: args.max_lines]:
        print(f"{issue.target}:{issue.code}:{issue.message}")
    if len(issues) > args.max_lines:
        print(f"... {len(issues) - args.max_lines} more issues omitted")
    print(f"numpydoc issues: {len(issues)}")
    return 1 if args.check and issues else 0


if __name__ == "__main__":
    raise SystemExit(main())
