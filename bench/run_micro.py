"""Thin CLI wrapper around the existing ``bench run`` subcommand.

Provides ``python -m bench.run_micro`` as a peer to
``python -m bench.run_macro`` and ``python -m bench.dashboard``. The
implementation delegates to :func:`bench.__main__._cmd_run` so the
microprobe CLI stays defined in exactly one place.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Make sure the bench package is importable when run as a script.
if __package__ in {None, ""}:  # pragma: no cover - script-mode shim
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


_DEFAULT_OUT = "bench/out/micro.json"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="bench.run_micro",
        description="Run the SpaceCore microprobe suite.",
    )
    parser.add_argument(
        "--out", default=_DEFAULT_OUT,
        help="Path of the JSON artifact to write (default: %(default)s).",
    )
    parser.add_argument(
        "--backend", action="append", default=None,
        choices=["numpy", "jax", "torch", "cupy"],
        help="Run only on these backend(s) (repeatable).",
    )
    parser.add_argument(
        "--family", action="append", default=None,
        choices=["space", "linop", "functional"],
        help="Run only probes in this family (repeatable).",
    )
    parser.add_argument(
        "--regime", action="append", default=None,
        choices=["baseline", "dispatch", "dispatch_cache", "verify"],
        help="Dispatch/cache regime(s) to sweep for linop probes (repeatable). "
             "Default: baseline + dispatch_cache.",
    )
    parser.add_argument(
        "--match", default=None,
        help="Substring filter on probe name.",
    )
    parser.add_argument(
        "--max-size", type=int, default=None,
        help="Run each probe only at sizes <= this value (keeps small "
             "instances; lighter on CPU/memory). Default: every size.",
    )
    parser.add_argument(
        "--quiet", action="store_true",
        help="Suppress per-probe progress.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Entry point: build a Namespace and forward to ``_cmd_run``."""
    from bench.__main__ import _cmd_run

    args = _build_parser().parse_args(argv)
    # ``_cmd_run`` consumes a richer Namespace than this CLI exposes;
    # build the missing fields with their ``bench run`` defaults so the
    # downstream code path doesn't have to care which entry point it
    # was invoked from.
    ns = argparse.Namespace(
        json=args.out,
        html=None,
        open=False,
        family=args.family,
        backend=args.backend,
        device=None,
        match=args.match,
        max_size=args.max_size,
        quiet=args.quiet,
    )
    return _cmd_run(ns)


if __name__ == "__main__":
    raise SystemExit(main())
