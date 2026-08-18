"""Figures E2 and E3 — tail performance.

E2 is the headline: the system-wide weighted P99 of each policy on each of
the six topologies. E3 shows where that number comes from, plotting the
complementary CDF of the weighted sojourn so the body and the tail of the
distribution can be read separately.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from .. import runner, scenarios, style
from ..protocol import POLICIES

#: Rows kept per replication for the CCDF. The curve is drawn on a log-log
#: axis down to 1e-4, so a few thousand points per replication resolve the
#: tail without making the vector figure enormous.
CCDF_CAP = 1500


def _legend_handles():
    return [Patch(facecolor=p.color, edgecolor="none", label=p.label) for p in POLICIES]


def build_e2(output: Path) -> None:
    """E2 — system-wide weighted P99 on the six topologies."""
    style.apply()
    stress = scenarios.stress()
    labels = list(stress)

    reports = runner.run_all(
        ((label, policy.key), runner.summary, (stress[label], policy.block), {})
        for label in labels
        for policy in POLICIES
    )

    fig, ax = plt.subplots(figsize=(style.COLUMN_WIDTH, 1.9))
    centres = None
    for index, policy in enumerate(POLICIES):
        centres, positions = style.bar_positions(len(labels), index)
        values, errors = zip(
            *(runner.weighted_p99(reports[(label, policy.key)]) for label in labels)
        )
        ax.bar(
            positions,
            values,
            style.BAR_WIDTH,
            color=policy.color,
            linewidth=0,
            yerr=errors,
            error_kw=style.ERROR_BAR,
        )

    ax.set_xticks(centres)
    ax.set_xticklabels(labels)
    ax.set_xlim(centres[0] - style.GROUP_STEP / 2, centres[-1] + style.GROUP_STEP / 2)
    ax.set_xlabel("Topology")
    ax.set_ylabel("Weighted P99 sojourn (s)")
    ax.set_yscale("log")
    ax.set_ylim(100, 3000)
    style.box(ax, n_decades=3)

    fig.tight_layout()
    style.frame(
        ax.legend(
            handles=_legend_handles(),
            loc="upper right",
            bbox_to_anchor=(0.995, 0.995),
            ncol=2,
            columnspacing=0.8,
            **style.LEGEND_KW,
        )
    )
    style.save(fig, output, "E2_wp99_bars")


def build_e3(output: Path) -> None:
    """E3 — complementary CDF of the weighted sojourn, one panel per graph."""
    style.apply()
    stress = scenarios.stress()
    labels = list(stress)

    samples = runner.run_all(
        (
            (label, policy.key),
            runner.weighted_sojourn_samples,
            (stress[label], policy.block, CCDF_CAP),
            {},
        )
        for label in labels
        for policy in POLICIES
    )

    fig, axes = plt.subplots(2, 3, figsize=(style.COLUMN_WIDTH, 2.05))
    for index, (ax, label) in enumerate(zip(axes.flat, labels)):
        # Focus the x-range on the region where the policies differ: from
        # the lowest median to a little past the highest P99.9.
        low = min(runner.quantile(samples[(label, p.key)], 0.50) for p in POLICIES)
        high = max(runner.quantile(samples[(label, p.key)], 0.999) for p in POLICIES) * 1.3
        for policy in POLICIES:
            values = samples[(label, policy.key)]
            if not values:
                continue
            survival = [1.0 - (i + 1) / len(values) for i in range(len(values))]
            ax.plot(values, survival, color=policy.color, lw=1.1)

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_ylim(1e-4, 1.05)
        ax.set_xlim(max(low, 1.0), high)
        style.box(ax, n_decades=3)
        ax.set_box_aspect(0.5)
        if index % 3 != 0:
            ax.tick_params(axis="y", labelleft=False)
        style.panel_letter(ax, f"({label})")

    fig.supylabel(r"Pr(weighted sojourn $> x$)", fontsize=7.0, x=0.01)
    fig.supxlabel("Weighted sojourn (s)", fontsize=7.0, y=0.02)
    left, right = 0.13, 0.985
    style.frame(
        fig.legend(
            handles=_legend_handles(),
            loc="upper center",
            bbox_to_anchor=((left + right) / 2, 1.00),
            ncol=4,
            columnspacing=0.6,
            **style.LEGEND_KW,
        )
    )
    # Explicit margins: tight_layout does not know about the panel letters
    # drawn below each axis, and would clip them.
    fig.subplots_adjust(left=left, right=right, top=0.93, bottom=0.18, hspace=0.30, wspace=0.08)
    style.save(fig, output, "E3_weighted_ccdf")
