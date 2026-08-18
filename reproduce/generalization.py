"""Generalization batteries: does the result hold off the six stress graphs?

Three batteries, all on graphs DART was never tuned against:

* **random** — twelve random sparse six-configuration graphs.
* **scaling** — the same structure grown from 4 to 10 configurations.
* **regime** — a 5x5 grid over weight dominance `w_0/w_i` and
  reconfiguration asymmetry `r/d`.

For each case the four policies run under one protocol and the reported
number is DART's margin over the *best* baseline on that case, which is the
conservative comparison: the baseline is allowed to be whichever of the
three happens to suit the graph.

Results are written to ``figures/generalization.json`` (read by the regime
figure) and printed as a table.
"""

from __future__ import annotations

import json
from pathlib import Path

from . import runner, scenarios
from .protocol import (
    BASELINES,
    DART,
    GENERALIZATION_HORIZON,
    GENERALIZATION_WARMUP,
    POLICIES,
    REPLICATIONS,
    SEED,
)

#: A margin inside this band counts as a tie rather than a win or a loss.
TIE_BAND_PCT = 1.0

PROTOCOL = dict(
    horizon=GENERALIZATION_HORIZON,
    warmup=GENERALIZATION_WARMUP,
    replications=REPLICATIONS,
    seed=SEED,
)


def _run_battery(battery: str) -> list[dict]:
    cases = scenarios.generalization(battery)
    jobs = [
        ((case.label, policy.key), runner.summary, (case, policy.block), PROTOCOL)
        for case in cases
        for policy in POLICIES
    ]
    reports = runner.run_all(jobs)

    rows = []
    for case in cases:
        values = {
            policy.key: runner.weighted_p99(reports[(case.label, policy.key)])[0]
            for policy in POLICIES
        }
        best_baseline = min(values[policy.key] for policy in BASELINES)
        rows.append(
            {
                "case": case.label,
                "weighted_p99": values,
                "best_baseline": best_baseline,
                "margin_pct": 100.0 * (best_baseline - values[DART.key]) / best_baseline,
            }
        )
    return rows


def _summarise(rows: list[dict]) -> dict:
    margins = [row["margin_pct"] for row in rows]
    return {
        "cases": len(margins),
        "wins": sum(1 for m in margins if m > TIE_BAND_PCT),
        "ties": sum(1 for m in margins if -TIE_BAND_PCT <= m <= TIE_BAND_PCT),
        "losses": sum(1 for m in margins if m < -TIE_BAND_PCT),
        "mean_margin_pct": sum(margins) / len(margins),
        "worst_margin_pct": min(margins),
        "best_margin_pct": max(margins),
    }


def run(output: Path) -> dict:
    """Run all three batteries and write the report to `output`."""
    report = {
        "protocol": PROTOCOL,
        "tie_band_pct": TIE_BAND_PCT,
        "batteries": {},
    }
    for battery in scenarios.BATTERIES:
        rows = _run_battery(battery)
        report["batteries"][battery] = {"cases": rows, "summary": _summarise(rows)}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    return report


def format_report(report: dict) -> str:
    """The report as a plain-text table."""
    lines = []
    for battery, block in report["batteries"].items():
        name = scenarios.BATTERIES[battery]
        summary = block["summary"]
        lines.append("")
        lines.append(f"--- {name} ({summary['cases']} cases) ---")
        header = f"{'case':22}" + "".join(f"{p.label:>11}" for p in POLICIES)
        lines.append(header + f"{'best base':>11}{'margin':>9}")
        for row in block["cases"]:
            cells = "".join(f"{row['weighted_p99'][p.key]:>11.0f}" for p in POLICIES)
            lines.append(
                f"{row['case']:22}{cells}{row['best_baseline']:>11.0f}"
                f"{row['margin_pct']:>8.1f}%"
            )
        lines.append(
            f"  -> wins={summary['wins']}/{summary['cases']}  ties={summary['ties']}  "
            f"losses={summary['losses']}  mean margin={summary['mean_margin_pct']:.1f}%  "
            f"worst={summary['worst_margin_pct']:.1f}%  best={summary['best_margin_pct']:.1f}%"
        )
    return "\n".join(lines)


def regime_grid(report: dict) -> tuple[list[float], list[float], dict[tuple[float, float], float]]:
    """The regime battery as ``(dominance values, asymmetry values, margins)``.

    Case labels have the form ``dom_<w0/wi>__asym_<r/d>``.
    """
    margins = {}
    for row in report["batteries"]["regime"]["cases"]:
        head, _, tail = row["case"].partition("__asym_")
        dominance = float(head[len("dom_"):])
        asymmetry = float(tail)
        margins[(dominance, asymmetry)] = row["margin_pct"]
    dominances = sorted({d for d, _ in margins})
    asymmetries = sorted({a for _, a in margins})
    return dominances, asymmetries, margins
