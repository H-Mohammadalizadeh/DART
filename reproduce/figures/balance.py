"""Figures E4 and E6 — where the gain comes from.

The objective pools every job's weighted sojourn into one distribution, so
the system P99 is set by whichever class puts the largest values into the
upper tail. Protecting one class is not enough if the other builds a heavy
tail. These two figures separate the classes.

E6 plots the two class tails against each other on the two hardest graphs,
so a policy's position relative to the diagonal shows directly whether it
balanced them. E4 gives the same split across all six graphs as bars.

Both take the per-class P99 one replication at a time and average over
replications, so their error bars use the same Student-t method as every
other figure.
"""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from .. import runner, scenarios, style
from ..protocol import POLICIES

#: Sojourn samples kept per configuration per replication.
SAMPLE_CAP = 2000

#: Two-sided 95% Student-t critical values, by degrees of freedom. Only the
#: entries the replication counts here can reach are listed.
_T_CRITICAL = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 10: 2.228, 15: 2.131, 20: 2.086}


def _t_critical(df: int) -> float:
    if df in _T_CRITICAL:
        return _T_CRITICAL[df]
    return 1.96 if df >= 30 else 2.131


def class_p99(samples, scenario, configurations):
    """Weighted P99 of one class: ``(mean over replications, 95% half-width)``.

    Each replication contributes one exact quantile of the class's own
    weighted-sojourn population; the interval is taken over those.
    """
    replications = sorted({replication for replication, _ in samples})
    per_replication = []
    for replication in replications:
        weighted = [
            scenario.weights[i] * s
            for i in configurations
            for s in samples.get((replication, i), [])
        ]
        if weighted:
            per_replication.append(runner.quantile(weighted, 0.99))
    if not per_replication:
        return float("nan"), 0.0
    mean = sum(per_replication) / len(per_replication)
    if len(per_replication) < 2:
        return mean, 0.0
    variance = sum((x - mean) ** 2 for x in per_replication) / (len(per_replication) - 1)
    half_width = _t_critical(len(per_replication) - 1) * math.sqrt(variance) / math.sqrt(
        len(per_replication)
    )
    return mean, half_width


def _collect(labels):
    stress = scenarios.stress()
    return runner.run_all(
        (
            (label, policy.key),
            runner.per_queue_samples,
            (stress[label], policy.block, SAMPLE_CAP),
            {},
        )
        for label in labels
        for policy in POLICIES
    )


def build_e4(output: Path) -> None:
    """E4 — per-class weighted P99 across the six topologies."""
    style.apply()
    stress = scenarios.stress()
    labels = list(stress)
    samples = _collect(labels)

    fig, (ax_high, ax_low) = plt.subplots(1, 2, figsize=(style.COLUMN_WIDTH, 2.0))
    # One shared y-range, so dropping the right panel's tick labels is
    # honest rather than a saving.
    y_limits = (40, 4000)
    for ax, members in (
        (ax_high, lambda s: s.high_priority()),
        (ax_low, lambda s: s.low_priority()),
    ):
        centres = None
        for index, policy in enumerate(POLICIES):
            centres, positions = style.bar_positions(len(labels), index)
            values, errors = [], []
            for label in labels:
                scenario = stress[label]
                mean, half = class_p99(samples[(label, policy.key)], scenario, members(scenario))
                values.append(mean)
                errors.append(half)
            ax.bar(
                positions,
                values,
                style.BAR_STEP,  # bars abut exactly within a group
                color=policy.color,
                linewidth=0,
                yerr=errors,
                error_kw=style.ERROR_BAR,
            )
        ax.set_xticks(centres)
        ax.set_xticklabels(labels)
        ax.set_xlim(centres[0] - style.GROUP_STEP / 2, centres[-1] + style.GROUP_STEP / 2)
        ax.set_xlabel("Topology")
        ax.set_yscale("log")
        ax.set_ylim(*y_limits)
        style.box(ax, n_decades=3)

    ax_high.set_ylabel("Weighted P99 (s)")
    ax_low.tick_params(axis="y", labelleft=False)

    left, right, top, bottom = 0.12, 0.99, 0.82, 0.20
    fig.subplots_adjust(left=left, right=right, top=top, bottom=bottom, wspace=0.08)
    style.frame(
        fig.legend(
            handles=[Patch(facecolor=p.color, edgecolor="none", label=p.label) for p in POLICIES],
            loc="lower center",
            bbox_to_anchor=((left + right) / 2, top + 0.01),
            ncol=4,
            columnspacing=0.6,
            **style.LEGEND_KW,
        )
    )
    style.save(fig, output, "E4_per_class")


def build_e6(output: Path) -> None:
    """E6 — low-priority against high-priority tail on the two hardest graphs."""
    style.apply()
    stress = scenarios.stress()
    panels = [("C", "return-trap"), ("F", "multipath trap")]
    labels = [label for label, _ in panels]
    samples = _collect(labels)

    fig, axes = plt.subplots(1, 2, figsize=(style.COLUMN_WIDTH, 2.05))
    for ax, (label, description) in zip(axes, panels):
        scenario = stress[label]
        points = {}
        for policy in POLICIES:
            per_queue = samples[(label, policy.key)]
            points[policy.key] = (
                class_p99(per_queue, scenario, scenario.high_priority())[0],
                class_p99(per_queue, scenario, scenario.low_priority())[0],
            )

        flat = [v for pair in points.values() for v in pair]
        low, high = min(flat) * 0.8, max(flat) * 1.25
        ax.plot([low, high], [low, high], ls="--", color="#888888", lw=0.8, zorder=1)
        for policy in POLICIES:
            ax.scatter(
                *points[policy.key],
                s=34,
                color=policy.color,
                edgecolor=style.INK,
                linewidth=0.6,
                zorder=3,
                clip_on=False,
            )
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(low, high)
        ax.set_ylim(low, high)
        ax.set_box_aspect(1)
        style.box(ax, n_decades=3)
        ticks = style.doubling_ticks(low, high)
        style.integer_ticks(ax.xaxis, ticks)
        style.integer_ticks(ax.yaxis, ticks)
        style.panel_letter(ax, f"({label}) {description}", y=-0.30, size=7.2)

    axes[0].text(
        0.30,
        0.40,
        "balanced",
        transform=axes[0].transAxes,
        rotation=45,
        ha="center",
        va="center",
        fontsize=6.2,
        color="#888888",
    )
    fig.supxlabel("High-priority weighted P99 (s)", fontsize=7.0, y=0.04)
    fig.supylabel("Low-priority weighted P99 (s)", fontsize=7.0, x=0.005)

    left, right, top, bottom = 0.12, 0.985, 0.80, 0.20
    fig.subplots_adjust(left=left, right=right, top=top, bottom=bottom, wspace=0.30)
    style.frame(
        fig.legend(
            handles=[
                Line2D(
                    [0],
                    [0],
                    marker="o",
                    ls="none",
                    color=p.color,
                    mec=style.INK,
                    mew=0.6,
                    ms=4.2,
                    label=p.label,
                )
                for p in POLICIES
            ],
            loc="lower center",
            bbox_to_anchor=((left + right) / 2, top + 0.01),
            ncol=4,
            columnspacing=0.7,
            **{k: v for k, v in style.LEGEND_KW.items() if k != "handleheight"},
        )
    )
    style.save(fig, output, "E6_class_balance")
