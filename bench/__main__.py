"""Unified bench CLI: ``python -m bench {run, compare, plot, summary, list}``.

Each subcommand maps to a small wrapper over the modules under
:mod:`bench`. ``run`` is the most common entry point and produces both
a JSON artifact and (optionally) a dashboard PNG.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Make sure the bench package is importable when run as a script.
if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def _cmd_run(args: argparse.Namespace) -> int:
    from bench import _operations  # noqa: F401  (registers probes)
    from bench._dashboard import render_dashboard
    from bench._io import save
    from bench._probes import registry
    from bench._run import run_probes
    from bench._verdict import make_verdict, render_text

    probes = registry.all()
    if args.family:
        probes = registry.filter(families=tuple(args.family))
    if args.match:
        probes = tuple(p for p in probes if args.match in p.name)
    if not probes:
        print("no probes matched the filter", file=sys.stderr)
        return 2

    backends = tuple(args.backend) if args.backend else None
    devices = tuple(args.device) if args.device else None
    max_size = getattr(args, "max_size", None)
    results = run_probes(
        probes,
        backends=backends,
        devices=devices,
        max_size=max_size,
        progress=not args.quiet,
    )
    if not results and max_size is not None:
        print(
            f"no cases ran: every matched probe's smallest size is > --max-size {max_size}",
            file=sys.stderr,
        )
        return 2
    save(results, args.json)
    print(f"wrote {args.json}")

    if args.html:
        render_dashboard(results, args.html)
        print(f"wrote {args.html}")
        if args.open:
            from bench._dashboard import open_in_browser
            open_in_browser(args.html)

    verdict = make_verdict(results)
    print(render_text(results, verdict))
    return 0


def _cmd_compare(args: argparse.Namespace) -> int:
    from bench._dashboard import open_in_browser, render_dashboard
    from bench._io import load
    from bench._verdict import compare_to_baseline, render_text

    current = load(args.current)
    baseline = load(args.baseline)
    verdict, lines = compare_to_baseline(current, baseline)
    print(render_text(current, verdict))
    if args.html:
        render_dashboard(current, args.html, baseline=baseline)
        print(f"\nwrote {args.html}")
        if args.open:
            open_in_browser(args.html)
    if lines:
        print(f"\n{len(lines)} regression(s) vs baseline:")
        for line in lines:
            print(f"  - {line}")
        return 1
    print("\nno regressions vs baseline")
    return 0


def _cmd_plot(args: argparse.Namespace) -> int:
    from bench._dashboard import open_in_browser, render_dashboard
    from bench._io import load

    results = load(args.json)
    out = render_dashboard(results, args.out)
    print(f"wrote {out}")
    if args.open:
        open_in_browser(out)
    return 0


def _cmd_summary(args: argparse.Namespace) -> int:
    from bench._io import load
    from bench._verdict import make_verdict, render_text

    results = load(args.json)
    verdict = make_verdict(results)
    print(render_text(results, verdict))
    return 0


def _cmd_list(args: argparse.Namespace) -> int:
    from bench import _operations  # noqa: F401  (registers probes)
    from bench._probes import registry

    probes = registry.all()
    if args.family:
        probes = registry.filter(families=tuple(args.family))
    print(f"{len(probes)} probe(s):")
    print(f"  {'family':<12s}  {'name':<40s}  {'sizes':<22s}  notes")
    print("  " + "-" * 90)
    for p in probes:
        sizes_str = ",".join(str(s) for s in p.sizes)
        print(
            f"  {p.family:<12s}  {p.name:<40s}  {sizes_str:<22s}  {p.notes}"
        )
    return 0


_FORWARDED_SUBCOMMANDS = {
    "run_macro": "bench.run_macro",
    "dashboard": "bench.dashboard",
}


def _forward_to(module_name: str, argv: list[str]) -> int:
    """Import ``module_name`` and call its ``main(argv)`` entry point.

    Used to make ``python -m bench run_macro`` and ``python -m bench
    dashboard`` behave the same as invoking the dedicated module
    scripts directly, without duplicating the argument parsers.
    """
    import importlib

    module = importlib.import_module(module_name)
    return int(module.main(argv))


def main(argv: list[str] | None = None) -> int:
    if argv is None:
        argv = sys.argv[1:]
    # Forward ``run_macro`` and ``dashboard`` before argparse sees them
    # so we don't have to duplicate their option schemas here. The
    # original subcommands (run, compare, plot, summary, list) keep
    # working through the argparse path below.
    if argv and argv[0] in _FORWARDED_SUBCOMMANDS:
        return _forward_to(_FORWARDED_SUBCOMMANDS[argv[0]], argv[1:])

    parser = argparse.ArgumentParser(prog="bench", description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)

    p_run = sub.add_parser("run", help="Run the bench suite and save results.")
    p_run.add_argument("--json", default="bench/out/bench.json")
    p_run.add_argument(
        "--html", default=None,
        help="Write a self-contained interactive Plotly dashboard to this path.",
    )
    p_run.add_argument(
        "--open", action="store_true",
        help="Open the HTML dashboard in the default browser after writing.",
    )
    p_run.add_argument(
        "--family", action="append", default=None,
        choices=["space", "linop", "functional", "linalg", "kernel"],
        help="Run only probes in this family (repeatable).",
    )
    p_run.add_argument(
        "--backend", action="append", default=None,
        choices=["numpy", "jax", "torch", "cupy"],
        help="Run only on these backend(s) (repeatable). "
             "Defaults to every backend each probe declares.",
    )
    p_run.add_argument(
        "--device", action="append", default=None,
        choices=["cpu", "cuda", "mps", "gpu", "tpu"],
        help="Run only on these device(s) (repeatable). "
             "Defaults to every available device per backend.",
    )
    p_run.add_argument("--match", default=None, help="Substring filter on probe name.")
    p_run.add_argument(
        "--max-size", type=int, default=None,
        help="Run each probe only at sizes <= this value. Keeps the run on "
             "small instances (lighter on CPU/memory). Default: every size.",
    )
    p_run.add_argument("--quiet", action="store_true", help="Suppress per-probe progress.")
    p_run.set_defaults(func=_cmd_run)

    p_cmp = sub.add_parser("compare", help="Compare a current run against a baseline.")
    p_cmp.add_argument("current")
    p_cmp.add_argument("baseline")
    p_cmp.add_argument(
        "--html", default=None,
        help="Write a dashboard with current-vs-baseline regression plot.",
    )
    p_cmp.add_argument("--open", action="store_true")
    p_cmp.set_defaults(func=_cmd_compare)

    p_plot = sub.add_parser("plot", help="Render a dashboard from a JSON artifact.")
    p_plot.add_argument("json")
    p_plot.add_argument("--out", default="bench/out/dashboard.html")
    p_plot.add_argument("--open", action="store_true")
    p_plot.set_defaults(func=_cmd_plot)

    p_sum = sub.add_parser("summary", help="Print the verdict from a JSON artifact.")
    p_sum.add_argument("json")
    p_sum.set_defaults(func=_cmd_summary)

    p_list = sub.add_parser("list", help="List registered probes.")
    p_list.add_argument(
        "--family", action="append", default=None,
        choices=["space", "linop", "functional", "linalg", "kernel"],
    )
    p_list.set_defaults(func=_cmd_list)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
