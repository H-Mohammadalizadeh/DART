"""The eight figures of the paper's evaluation.

Each entry below names the figure as the paper does, so a figure in the
PDF can be traced to the code that drew it and to the data behind it. The
grouping follows the paper's own subsections.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

from . import balance, robustness, tail, topologies

#: ``name -> (paper section, one-line description, builder)``. The builder
#: takes the output directory; the regime map additionally takes the
#: generalization report it reads.
FIGURES: dict[str, tuple[str, str, Callable]] = {
    "E1": ("V-B", "the six stress topologies and their reconfiguration matrices", topologies.build),
    "E2": ("V-D", "system-wide weighted P99 on the six topologies", tail.build_e2),
    "E3": ("V-D", "complementary CDF of the weighted sojourn", tail.build_e3),
    "E4": ("V-E", "per-class weighted P99 across the six topologies", balance.build_e4),
    "E5": ("V-F", "weighted P99 against offered load on topology F", robustness.build_e5),
    "E6": ("V-E", "low- against high-priority tail on the two hardest graphs", balance.build_e6),
    "E8": ("V-F", "margin over the best baseline across the regime grid", robustness.build_e8),
    "E10": ("V-F", "each method swept over its own knob", robustness.build_e10),
}

#: The one figure that needs the generalization batteries to have run.
NEEDS_GENERALIZATION = {"E8"}


def build(name: str, output: Path, report_path: Path) -> None:
    """Build one figure into `output`."""
    _, _, builder = FIGURES[name]
    if name in NEEDS_GENERALIZATION:
        builder(output, report_path)
    else:
        builder(output)
