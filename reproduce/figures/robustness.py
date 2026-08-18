"""Figures E5, E8 and E10 — robustness, tuning fairness, and the limits.

E5 sweeps the offered load on the hardest graph, showing that the policies
separate as the system approaches saturation.

E10 answers the tuning objection. DART has a knob (`β`) and so does Tian-T
(`η`); comparing a swept DART against a fixed Tian-T would be worthless.
Both are therefore swept over their own knob and shown as vertical spreads,
so the claim becomes "DART's worst setting still beats Tian-T's best".

E8 maps where the advantage ends: DART's margin over the best baseline
across a grid of weight dominance and reconfiguration asymmetry.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from .. import generalization, runner, scenarios, style
from ..protocol import (
    DART,
    DART_BETA,
    DVO,
    POLICIES,
    TIAN,
    TIAN_T,
    TIAN_T_ETA,
    dart_block,
    tian_transit_block,
)

#: Offered loads swept on topology F. The top of the range sits at that
#: scenario's own load, just below saturation.
LOADS = [0.6, 0.7, 0.8, 0.9, 0.94]

#: Knob values swept for the fairness figure. `β` is DART's delay-guard
#: strength; `η` is Tian-T's reluctance to pause in transit.
BETA_SWEEP = [0, 1, 2, 4, 6, 8, 12]
ETA_SWEEP = [0.5, 1.0, 2.0, 5.0, 10.0]

#: Colour scale limit for the regime map, in percent margin.
REGIME_SCALE = 30.0


def build_e5(output: Path) -> None:
    """E5 — weighted P99 against offered load on topology F."""
    style.apply()
    scenario = scenarios.stress()["F"]
    base = scenario.arrival_rates
    base_load = scenario.offered_load

    jobs = []
    for load in LOADS:
        factor = load / base_load
        # Rounded so the scaled rates are exactly representable in the
        # scenario text and the run is reproducible from the file alone.
        scaled = scenario.with_arrival_rates([round(x * factor, 6) for x in base])
        for policy in POLICIES:
            jobs.append(((load, policy.key), runner.summary, (scaled, policy.block), {}))
    reports = runner.run_all(jobs)

    # Square plot region so this figure pairs with the regime map; the
    # legend goes inside the empty upper-left so the figure stays square.
    fig, ax = plt.subplots(figsize=(1.62, 1.62))
    extremes = []
    for policy in POLICIES:
        values, errors = zip(*(runner.weighted_p99(reports[(load, policy.key)]) for load in LOADS))
        ax.errorbar(
            LOADS,
            values,
            yerr=errors,
            color=policy.color,
            lw=1.1,
            marker="o",
            ms=2.4,
            capsize=1.2,
            elinewidth=0.6,
            capthick=0.6,
        )
        extremes += [v - e for v, e in zip(values, errors)]
        extremes += [v + e for v, e in zip(values, errors)]

    ax.set_xlabel(r"$\rho$", fontsize=9.0)
    ax.set_ylabel("Weighted P99 (s)")
    ax.set_yscale("log")
    style.box(ax, n_decades=3)
    low, high = min(extremes), max(extremes)
    ax.set_ylim(low * 0.9, high * 1.12)
    style.integer_ticks(ax.yaxis, style.doubling_ticks(low, high, keep_below=0.95))
    ax.set_box_aspect(1)
    fig.subplots_adjust(left=0.215, right=0.965, top=0.965, bottom=0.175)
    style.frame(
        ax.legend(
            handles=[Patch(facecolor=p.color, edgecolor="none", label=p.label) for p in POLICIES],
            loc="upper left",
            bbox_to_anchor=(0.02, 0.985),
            ncol=2,
            columnspacing=0.6,
            **{**style.LEGEND_KW, "labelspacing": 0.2},
        )
    )
    style.save(fig, output, "E5_load_sweep")


def build_e10(output: Path) -> None:
    """E10 — each method swept over its own knob, against unmodified Tian."""
    style.apply()
    stress = scenarios.stress()
    labels = list(stress)

    jobs = []
    for label in labels:
        scenario = stress[label]
        for policy in POLICIES:
            jobs.append(((label, "fixed", policy.key), runner.summary, (scenario, policy.block), {}))
        for beta in BETA_SWEEP:
            jobs.append(((label, "beta", beta), runner.summary, (scenario, dart_block(beta=beta)), {}))
        for eta in ETA_SWEEP:
            jobs.append(((label, "eta", eta), runner.summary, (scenario, tian_transit_block(eta)), {}))
    reports = runner.run_all(jobs)

    fig, ax = plt.subplots(figsize=(style.COLUMN_WIDTH, 2.05))
    offset = 0.17
    all_values = []
    for position, label in enumerate(labels):
        reference = runner.weighted_p99(reports[(label, "fixed", TIAN.key)])[0]

        def reduction(key):
            return 100.0 * (reference - runner.weighted_p99(reports[key])[0]) / reference

        for side, knob, values, default, colour in (
            (-1, "beta", BETA_SWEEP, DART_BETA, DART.color),
            (+1, "eta", ETA_SWEEP, TIAN_T_ETA, TIAN_T.color),
        ):
            reductions = [reduction((label, knob, v)) for v in values]
            all_values += reductions
            x = position + side * offset
            # The full sweep as a vertical span, every setting as a faint
            # dot, and the default setting as a filled marker.
            ax.plot(
                [x, x],
                [min(reductions), max(reductions)],
                color=colour,
                lw=1.1,
                alpha=0.40,
                zorder=2,
                solid_capstyle="round",
            )
            ax.plot(
                [x] * len(reductions),
                reductions,
                marker="o",
                ms=2.0,
                ls="none",
                color=colour,
                alpha=0.55,
                markeredgecolor="none",
                zorder=3,
            )
            ax.plot(
                x,
                reduction((label, knob, default)),
                marker="o",
                ms=4.2,
                color=colour,
                markeredgecolor="white",
                markeredgewidth=0.6,
                zorder=5,
            )

        # DVO has no knob, so it is a single point.
        dvo_value = reduction((label, "fixed", DVO.key))
        all_values.append(dvo_value)
        ax.plot(
            position,
            dvo_value,
            marker="D",
            ms=3.0,
            color=DVO.color,
            markeredgecolor="white",
            markeredgewidth=0.4,
            zorder=4,
        )

    low, high = min(all_values), max(all_values)
    span = high - low
    ax.set_xlim(-0.6, len(labels) - 0.4)
    ax.set_ylim(min(low, 0.0) - 0.06 * span, high + 0.16 * span)
    ax.axhline(0, color=TIAN.color, lw=1.0, ls="--", zorder=1)
    ax.text(
        len(labels) - 0.45,
        0.4,
        TIAN.label,
        fontsize=6.0,
        color=TIAN.color,
        va="bottom",
        ha="right",
    )
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_xlabel("Topology")
    ax.set_ylabel(f"Weighted-P99 reduction vs {TIAN.label} (%)")
    style.box(ax, log_y=False)

    handles = [
        Line2D([0], [0], color=DART.color, lw=1.1, alpha=0.7, marker="o", ms=2.2,
               label=rf"{DART.label} (sweep $\beta$)"),
        Line2D([0], [0], color=TIAN_T.color, lw=1.1, alpha=0.7, marker="o", ms=2.2,
               label=rf"{TIAN_T.label} (sweep $\eta$)"),
        Line2D([0], [0], color=DVO.color, ls="none", marker="D", ms=3.0, mec="white", mew=0.4,
               label=DVO.label),
    ]
    style.frame(
        ax.legend(
            handles=handles,
            loc="lower center",
            bbox_to_anchor=(0.5, 1.0),
            ncol=3,
            frameon=True,
            fancybox=False,
            handletextpad=0.35,
            columnspacing=0.8,
            handlelength=1.1,
            borderpad=0.3,
            labelspacing=0.25,
        )
    )
    # Fixed margins rather than tight_layout, so the rendered width matches
    # the other column-width figures exactly.
    fig.subplots_adjust(left=0.13, right=0.985, top=0.88, bottom=0.155)
    style.save(fig, output, "E10_knob_single")


def build_e8(output: Path, report_path: Path) -> None:
    """E8 — margin over the best baseline across the regime grid."""
    style.apply()
    report = json.loads(report_path.read_text())
    dominances, asymmetries, margins = generalization.regime_grid(report)

    grid = np.full((len(dominances), len(asymmetries)), np.nan)
    for (dominance, asymmetry), margin in margins.items():
        grid[dominances.index(dominance), asymmetries.index(asymmetry)] = margin

    # Sized like one panel of E4, roughly half a column. With equal aspect
    # the square grid sets the final width; insert it in LaTeX at that same
    # physical width so the type renders at its true 7 pt.
    fig, ax = plt.subplots(figsize=(2.1, 1.85))
    image = ax.imshow(
        grid, origin="lower", cmap="RdBu", vmin=-REGIME_SCALE, vmax=REGIME_SCALE, aspect="equal"
    )
    for row in range(len(dominances)):
        for column in range(len(asymmetries)):
            value = grid[row, column]
            if np.isnan(value):
                continue
            ax.text(
                column,
                row,
                f"{value:+.0f}",
                ha="center",
                va="center",
                fontsize=6.2,
                color="white" if abs(value) > 0.6 * REGIME_SCALE else style.INK,
            )

    ax.set_xticks(range(len(asymmetries)))
    ax.set_xticklabels([f"{a:g}" for a in asymmetries])
    ax.set_yticks(range(len(dominances)))
    ax.set_yticklabels([f"{d:g}" for d in dominances])
    ax.set_xlabel(r"Switchover asymmetry  $r/d$")
    ax.set_ylabel(r"Weight dominance  $w_0/w_i$")
    style.box(ax, log_y=False)
    ax.tick_params(length=0)  # a categorical grid gains nothing from ticks

    bar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    bar.set_ticks([-30, -15, 0, 15, 30])
    bar.set_label("Margin over best baseline (%)", fontsize=7.0, y=0.42)
    bar.outline.set_edgecolor(style.INK)
    bar.outline.set_linewidth(0.8)
    bar.ax.tick_params(length=2.4, width=0.9, color=style.INK)

    fig.tight_layout()
    style.save(fig, output, "E8_regime_heatmap")
