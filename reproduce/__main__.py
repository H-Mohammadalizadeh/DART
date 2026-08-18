"""Command-line entry point for reproducing the paper's evaluation.

    python -m reproduce                 # everything: batteries, then figures
    python -m reproduce --figures E2,E3 # only these figures
    python -m reproduce --list          # what can be built

Results are cached under ``.cache/``, keyed by the exact configuration text
of each run, so re-rendering after a styling change costs nothing. Delete
that directory to force a full recomputation.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from . import figures, generalization, runner

ROOT = Path(__file__).resolve().parent.parent
FIGURE_DIR = ROOT / "figures"
REPORT = FIGURE_DIR / "generalization.json"


def _list() -> None:
    print(f"{'figure':8}{'section':10}description")
    for name, (section, description, _) in figures.FIGURES.items():
        print(f"{name:8}{section:10}{description}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="reproduce", description=__doc__)
    parser.add_argument("--list", action="store_true", help="list the figures and exit")
    parser.add_argument(
        "--figures",
        metavar="NAMES",
        help="comma-separated subset to build (default: all)",
    )
    parser.add_argument(
        "--skip-generalization",
        action="store_true",
        help="reuse the existing generalization report instead of rerunning the batteries",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=FIGURE_DIR,
        help=f"output directory (default: {FIGURE_DIR.relative_to(ROOT)}/)",
    )
    args = parser.parse_args(argv)

    if args.list:
        _list()
        return 0

    try:
        runner.require_binaries()
    except runner.SimulatorMissing as error:
        print(error, file=sys.stderr)
        return 1

    selected = list(figures.FIGURES)
    if args.figures:
        selected = [name.strip() for name in args.figures.split(",") if name.strip()]
        unknown = [name for name in selected if name not in figures.FIGURES]
        if unknown:
            print(f"unknown figure(s): {', '.join(unknown)}", file=sys.stderr)
            return 2

    report_path = args.output / REPORT.name
    if any(name in figures.NEEDS_GENERALIZATION for name in selected):
        if args.skip_generalization and report_path.exists():
            print(f"reusing {report_path.name}")
        else:
            print("running generalization batteries ...")
            started = time.monotonic()
            report = generalization.run(report_path)
            print(generalization.format_report(report))
            print(f"\nwrote {report_path.name}  [{time.monotonic() - started:.1f}s]")

    for name in selected:
        started = time.monotonic()
        figures.build(name, args.output, report_path)
        section = figures.FIGURES[name][0]
        print(f"{name:5} (sec. {section})  [{time.monotonic() - started:.1f}s]")

    print(f"\nfigures written to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
