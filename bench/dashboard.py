"""CLI entry point for the unified bench dashboard.

Usage::

    python -m bench.dashboard --input MICRO.json [MACRO.json ...]
        [--out HTML] [--open]

Each ``--input`` path is loaded and inspected. JSON files written by
:mod:`bench._io` (microprobe runs) are recognized by their
``results`` rows shaped like :class:`bench._probes.ProbeResult`; files
written by :mod:`bench.run_macro` are recognized by their
``results`` rows shaped like :class:`bench.macro.MacroResult`. Multiple
files may be passed in any combination — the dashboard renders one HTML
page with both the micro and macro sections populated where data is
present.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

# Make sure the bench package is importable when run as a script.
if __package__ in {None, ""}:  # pragma: no cover - script-mode shim
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


_DEFAULT_OUT = "bench/out/dashboard.html"


_MICRO_MARKERS = ("speedup", "sc_median_ns", "bare_median_ns", "family")
_MACRO_MARKERS = ("benchmark_name", "mode", "run_time_ns", "size_label")


def _classify_row(row: dict) -> str | None:
    """Return ``'micro'``, ``'macro'``, or ``None`` for one result row."""
    if not isinstance(row, dict):
        return None
    if all(k in row for k in _MACRO_MARKERS):
        return "macro"
    if all(k in row for k in _MICRO_MARKERS):
        return "micro"
    return None


def _classify_payload(payload: Any) -> str | None:
    """Heuristic classification of a loaded JSON payload."""
    if not isinstance(payload, dict):
        return None
    rows = payload.get("results")
    if not isinstance(rows, list) or not rows:
        return None
    # Look at the first row that classifies; mixed files are unlikely
    # in practice (the two CLIs write different shapes).
    for row in rows:
        kind = _classify_row(row)
        if kind is not None:
            return kind
    return None


def _load_micro(path: Path):
    """Reconstruct :class:`ProbeResult` objects via :mod:`bench._io`."""
    from bench._io import load

    return load(path)


def _load_macro(path: Path):
    """Reconstruct :class:`MacroResult` objects from a run_macro artifact."""
    from bench.macro._schema import MacroResult

    raw = json.loads(path.read_text())
    rows = raw.get("results") or []
    out = []
    for row in rows:
        try:
            out.append(MacroResult.from_dict(row))
        except Exception:
            # Tolerate slightly older shapes by keeping the raw dict;
            # the dashboard's ``_macro_row_to_dict`` accepts both.
            out.append(row)
    return out


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="bench.dashboard",
        description="Render the unified bench dashboard from JSON artifacts.",
    )
    parser.add_argument(
        "--input", action="append", required=True,
        help=(
            "Path to a micro or macro JSON artifact (repeatable). At "
            "least one of the inputs must contain results."
        ),
    )
    parser.add_argument(
        "--out", default=_DEFAULT_OUT,
        help="Where to write the dashboard HTML (default: %(default)s).",
    )
    parser.add_argument(
        "--open", action="store_true",
        help="Open the rendered dashboard in the default browser.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    from bench._dashboard import open_in_browser, render_dashboard

    args = _build_parser().parse_args(argv)

    micro_rows = []
    macro_rows = []
    for raw in args.input:
        path = Path(raw)
        if not path.exists():
            print(f"no such input: {path}", file=sys.stderr)
            return 2
        try:
            payload = json.loads(path.read_text())
        except json.JSONDecodeError as err:
            print(f"failed to parse {path}: {err}", file=sys.stderr)
            return 2
        kind = _classify_payload(payload)
        if kind == "micro":
            micro_rows.extend(_load_micro(path))
        elif kind == "macro":
            macro_rows.extend(_load_macro(path))
        else:
            print(
                f"warning: {path} has no recognizable micro or macro rows; "
                f"skipping",
                file=sys.stderr,
            )

    if not micro_rows and not macro_rows:
        print("no usable results across the supplied inputs", file=sys.stderr)
        return 2

    out_path = render_dashboard(
        micro_rows,
        args.out,
        macro_results=macro_rows or None,
    )
    print(f"wrote {out_path}")
    if args.open:
        open_in_browser(out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
