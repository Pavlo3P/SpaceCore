"""CLI entry point for the macrobenchmark suite.

Usage::

    python -m bench.run_macro [--out PATH] [--benchmark NAME ...]
        [--backend ...] [--check-level none|cheap|both]
        [--seeds 0,1,2,3] [--sizes small,medium,large|all]
        [--quick] [--quiet]

Writes a single JSON artifact with ``meta``, the flat list of
:class:`bench.macro.MacroResult` rows, and per-group summaries computed
by :func:`bench.macro.group_summaries`.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Make sure the bench package is importable when run as a script.
if __package__ in {None, ""}:  # pragma: no cover - script-mode shim
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


_DEFAULT_OUT = "bench/out/macro.json"


# --- argument parsing helpers ------------------------------------------------


def _parse_int_csv(value: str) -> tuple[int, ...]:
    """Parse a comma-separated list of ints like ``"0,1,2,3"``."""
    if not value:
        return ()
    parts = [p.strip() for p in value.split(",") if p.strip()]
    try:
        return tuple(int(p) for p in parts)
    except ValueError as err:
        raise argparse.ArgumentTypeError(
            f"--seeds expects comma-separated integers, got {value!r}: {err}"
        ) from err


def _parse_str_csv(value: str) -> tuple[str, ...]:
    """Parse a comma-separated list of strings."""
    if not value:
        return ()
    return tuple(p.strip() for p in value.split(",") if p.strip())


def _modes_for_check_level(level: str):
    """Return the macro RUN_MODES subset for a given ``--check-level``."""
    from bench.macro import RUN_MODES  # local import to keep startup cheap
    from bench.macro._schema import ModeName  # noqa: F401  (type hint only)

    if level == "none":
        return ("bare", "spacecore_public_none", "spacecore_lowered")
    if level == "cheap":
        return ("bare", "spacecore_public_cheap", "spacecore_lowered")
    if level == "both":
        return RUN_MODES
    raise argparse.ArgumentTypeError(
        f"--check-level must be one of none|cheap|both, got {level!r}"
    )


def _resolve_sizes(sizes_arg: tuple[str, ...], benchmarks) -> tuple[str, ...] | None:
    """Translate ``--sizes`` into the runner's ``sizes`` filter.

    The runner takes either ``None`` (run all sizes per benchmark) or a
    tuple of exact size labels to keep. We support the three friendly
    aliases ``small`` / ``medium`` / ``large`` by matching against any
    configured size label that contains the alias as a substring (so
    e.g. ``"medium-2d"`` matches ``medium``). Explicit labels are
    passed through verbatim, and ``all`` becomes ``None``.
    """
    if not sizes_arg or "all" in sizes_arg:
        return None
    aliases = {"small", "medium", "large"}
    wanted: set[str] = set()
    for benchmark in benchmarks:
        size_labels = list(benchmark.sizes)
        for token in sizes_arg:
            if token in aliases:
                wanted.update(label for label in size_labels if token in label)
            else:
                if token in size_labels:
                    wanted.add(token)
    return tuple(sorted(wanted)) if wanted else tuple(sizes_arg)


# --- argument parser ---------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="bench.run_macro",
        description="Run the SpaceCore macrobenchmark suite.",
    )
    parser.add_argument(
        "--out", default=_DEFAULT_OUT,
        help="Path of the JSON artifact to write (default: %(default)s).",
    )
    parser.add_argument(
        "--benchmark", action="append", default=None,
        help="Run only this benchmark (repeatable, by exact name).",
    )
    parser.add_argument(
        "--backend", action="append", default=None,
        choices=["numpy", "jax", "torch", "cupy"],
        help="Run only on these backend(s) (repeatable).",
    )
    parser.add_argument(
        "--check-level", default="both", choices=["none", "cheap", "both"],
        help=(
            "Which SpaceCore check_level modes to run. "
            "'none' -> bare + spacecore_public_none + spacecore_lowered. "
            "'cheap' -> bare + spacecore_public_cheap + spacecore_lowered. "
            "'both' (default) -> all four modes."
        ),
    )
    parser.add_argument(
        "--seeds", default="0,1,2,3", type=_parse_int_csv,
        help="Comma-separated list of seeds (default: %(default)s).",
    )
    parser.add_argument(
        "--sizes", default="all", type=_parse_str_csv,
        help=(
            "Comma-separated size selector. 'all' (default) runs every "
            "configured size; 'small'/'medium'/'large' match by "
            "substring; exact size labels are passed through verbatim."
        ),
    )
    parser.add_argument(
        "--quick", action="store_true",
        help=(
            "Override --sizes/--seeds with each benchmark's "
            "quick_size_labels() and seed (0,)."
        ),
    )
    parser.add_argument(
        "--quiet", action="store_true",
        help="Suppress per-benchmark progress output.",
    )
    return parser


# --- entry point -------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    from bench._io import _metadata
    from bench.macro import group_summaries, registry, run_benchmarks

    args = _build_parser().parse_args(argv)

    # Filter the benchmark set first so size resolution can use the
    # actual configured size labels.
    if args.benchmark:
        benchmarks = registry.filter(names=tuple(args.benchmark))
        if not benchmarks:
            print(
                f"no benchmarks matched {args.benchmark!r}; "
                f"available: {registry.names()}",
                file=sys.stderr,
            )
            return 2
    else:
        benchmarks = registry.all()
    if not benchmarks:
        print("no macrobenchmarks registered", file=sys.stderr)
        return 2

    modes = _modes_for_check_level(args.check_level)
    backends = tuple(args.backend) if args.backend else None
    sizes = _resolve_sizes(args.sizes, benchmarks) if not args.quick else None
    seeds = tuple(args.seeds) if args.seeds else (0, 1, 2, 3)

    results = run_benchmarks(
        benchmarks,
        backends=backends,
        seeds=seeds,
        sizes=sizes,
        modes=modes,
        quick=args.quick,
        progress=not args.quiet,
    )

    payload = {
        "meta": {
            **_metadata(),
            "cli": {
                "benchmark": list(args.benchmark) if args.benchmark else None,
                "backend": list(backends) if backends else None,
                "check_level": args.check_level,
                "seeds": list(seeds),
                "sizes": list(sizes) if sizes else "all",
                "quick": args.quick,
                "modes": list(modes),
            },
        },
        "results": [r.to_dict() for r in results],
        "summaries": group_summaries(results),
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, allow_nan=True))
    if not args.quiet:
        print(f"wrote {out_path} ({len(results)} macro rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
