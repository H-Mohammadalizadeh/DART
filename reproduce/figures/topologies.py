"""Figure E1 — the six stress topologies and their reconfiguration matrices.

Two rows of three pairs. Each pair is a mean reconfiguration-time matrix
beside the graph it describes: the top row holds the single-path graphs
A–C, the bottom row the multi-path graphs D–F. Nodes are labelled
``(w_i, λ_i)``; since μ_i = 1 throughout, λ_i is also the per-configuration
offered load. High-priority nodes are filled, low-priority ones left blank.

Edge values live only in the matrix. Drawing them on the graph as well
doubles the ink for no extra information and, on the denser graphs, forces
edge labels to overlap.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, Rectangle

from .. import scenarios, style

#: Node coordinates for the multi-path graphs. Layout is a presentation
#: choice, so it lives here rather than in the scenario files.
LAYOUT = {
    "D": [(1.0, 1.7), (0.0, 0.85), (2.0, 0.85), (1.0, 0.0)],
    "E": [(0.0, 1.6), (1.0, 1.6), (2.0, 1.6), (2.0, 0.0), (1.0, 0.0), (0.0, 0.0)],
    "F": [(0.4, 1.6), (-0.4, 0.0), (1.5, 0.0), (3.4, 0.0), (2.6, 1.6)],
}

#: F naturally spans 3.8 units in x against 2.0 for D and E; squeezing it
#: horizontally makes all three fill their panels about equally.
X_SQUEEZE = {"F": 2.0 / 3.8}

#: Where to put a node's ``(w, λ)`` label when the automatic radial
#: placement would collide with an edge or a neighbour.
_CLOSE = 0.19
LABEL_PLACEMENT = {
    # D: push the two side nodes' labels below rather than beside them.
    "D": {1: ("center", "top", 0.0, -0.30), 2: ("center", "top", 0.0, -0.30)},
    # E and F: one shared height above the top row, one below the bottom.
    "E": {
        0: ("center", "bottom", 0.0, _CLOSE),
        1: ("center", "bottom", 0.0, _CLOSE),
        2: ("center", "bottom", 0.0, _CLOSE),
        3: ("center", "top", 0.0, -_CLOSE),
        4: ("center", "top", 0.0, -_CLOSE),
        5: ("center", "top", 0.0, -_CLOSE),
    },
    "F": {
        0: ("center", "bottom", 0.0, _CLOSE),
        4: ("center", "bottom", 0.0, _CLOSE),
        1: ("center", "top", 0.0, -_CLOSE),
        2: ("center", "top", 0.0, -_CLOSE),
        3: ("center", "top", 0.0, -_CLOSE),
    },
}

HIGH_PRIORITY_FILL = "#BBD3E8"
LOW_PRIORITY_FILL = "white"
EDGE_INK = "#333333"
ABSENT_CELL = "#D9D9D9"


def _apply_style() -> None:
    """The shared style, plus what this figure alone needs: a math font for
    the node labels and smaller tick and legend type."""
    style.apply()
    plt.rcParams.update(
        {
            "mathtext.fontset": "stixsans",
            "xtick.labelsize": 5.6,
            "ytick.labelsize": 5.6,
            "legend.fontsize": 6.4,
        }
    )


def _node_label(weight: float, arrival_rate: float) -> str:
    return rf"$({weight:g},\,{arrival_rate:g})$"


def _arrow(ax, start, end, *, shrink=0.0, both=False, scale=4.0):
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="<|-|>" if both else "-|>",
            mutation_scale=scale,
            linewidth=0.45,
            color=EDGE_INK,
            shrinkA=shrink,
            shrinkB=shrink,
            zorder=1,
        )
    )


def _draw_line(ax, scenario) -> None:
    """Horizontal layout for the single-path graphs A–C."""
    matrix = scenario.switchover_times
    n = scenario.n
    high = set(scenario.high_priority())
    step, inset, y = 0.50, 0.13, 0.0
    xs = [i * step for i in range(n)]

    for i in range(n - 1):
        forward = matrix[i][i + 1] >= 0
        backward = matrix[i + 1][i] >= 0
        left = (xs[i] + inset, y)
        right = (xs[i + 1] - inset, y)
        if forward and backward:
            _arrow(ax, left, right, both=True, scale=3.5)
        elif forward:
            _arrow(ax, left, right, scale=3.5)
        elif backward:
            _arrow(ax, right, left, scale=3.5)

    for i in range(n):
        fill = HIGH_PRIORITY_FILL if i in high else LOW_PRIORITY_FILL
        ax.scatter([xs[i]], [y], s=130, facecolors=fill, edgecolors="black", linewidths=0.9, zorder=3)
        ax.text(xs[i], y, str(i), ha="center", va="center", fontsize=7.6, fontweight="bold", zorder=4)
        # Stagger the labels above and below the line so that same-side
        # labels sit two node-steps apart and never collide.
        below = i % 2 == 0
        ax.text(
            xs[i],
            y - 0.10 if below else y + 0.10,
            _node_label(scenario.weights[i], scenario.arrival_rates[i]),
            ha="center",
            va="top" if below else "bottom",
            fontsize=6.8,
            fontweight="bold",
        )

    ax.set_xlim(xs[0] - 0.24, xs[-1] + 0.24)
    ax.set_ylim(-0.42, 0.42)
    ax.set_axis_off()


def _draw_graph(ax, scenario, positions, placement=None, scale=0.65) -> None:
    """Spatial layout for the multi-path graphs D–F."""
    matrix = scenario.switchover_times
    n = scenario.n
    high = set(scenario.high_priority())
    positions = [(x * scale, y * scale) for x, y in positions]

    for i in range(n):
        for j in range(i + 1, n):
            forward = matrix[i][j] >= 0
            backward = matrix[j][i] >= 0
            if not forward and not backward:
                continue
            if forward and backward:
                _arrow(ax, positions[i], positions[j], shrink=10, both=True)
            elif forward:
                _arrow(ax, positions[i], positions[j], shrink=10)
            else:
                _arrow(ax, positions[j], positions[i], shrink=10)

    centre_x = sum(x for x, _ in positions) / n
    centre_y = sum(y for _, y in positions) / n
    label_points = []
    for i, (x, y) in enumerate(positions):
        fill = HIGH_PRIORITY_FILL if i in high else LOW_PRIORITY_FILL
        ax.scatter([x], [y], s=110, facecolors=fill, edgecolors="black", linewidths=0.9, zorder=3)
        ax.text(x, y, str(i), ha="center", va="center", fontsize=7.4, fontweight="bold", zorder=4)

        override = (placement or {}).get(i)
        if override is not None:
            ha, va, dx, dy = override
            label_x, label_y = x + dx, y + dy
        else:
            # Default: push the label radially outward from the centroid.
            dx, dy = x - centre_x, y - centre_y
            norm = (dx * dx + dy * dy) ** 0.5
            if norm < 1e-6:
                label_x, label_y, ha, va = x, y - 0.16, "center", "top"
            else:
                ux, uy = dx / norm, dy / norm
                label_x, label_y = x + ux * 0.16, y + uy * 0.16
                ha = "left" if ux > 0.4 else ("right" if ux < -0.4 else "center")
                va = "bottom" if uy > 0.3 else ("top" if uy < -0.3 else "center")
        ax.text(
            label_x,
            label_y,
            _node_label(scenario.weights[i], scenario.arrival_rates[i]),
            ha=ha,
            va=va,
            fontsize=6.4,
            fontweight="bold",
            zorder=5,
        )
        label_points.append((label_x, label_y))

    xs = [x for x, _ in positions] + [x for x, _ in label_points]
    ys = [y for _, y in positions] + [y for _, y in label_points]
    ax.set_xlim(min(xs) - 0.18, max(xs) + 0.18)
    ax.set_ylim(min(ys) - 0.18, max(ys) + 0.18)
    ax.set_aspect("equal")
    ax.set_axis_off()


def _draw_matrix(ax, scenario) -> None:
    """The N x N mean reconfiguration-time grid.

    Present edges carry their mean on a white cell; the diagonal and the
    absent edges are shaded, so the graph's shape is readable from the
    matrix alone.
    """
    matrix = np.asarray(scenario.switchover_times, dtype=float)
    n = matrix.shape[0]
    for i in range(n):
        for j in range(n):
            absent = i == j or matrix[i, j] < 0
            ax.add_patch(
                Rectangle(
                    (j, n - 1 - i),
                    1,
                    1,
                    facecolor=ABSENT_CELL if absent else "white",
                    edgecolor="black",
                    linewidth=0.8,
                )
            )
            if not absent:
                ax.text(
                    j + 0.5,
                    n - 1 - i + 0.5,
                    f"{matrix[i, j]:g}",
                    ha="center",
                    va="center",
                    fontsize=8.0,
                    fontweight="bold",
                )
    for k in range(n):
        ax.text(k + 0.5, n + 0.22, str(k), ha="center", va="bottom", fontsize=8.0, fontweight="bold")
        ax.text(-0.22, n - 1 - k + 0.5, str(k), ha="right", va="center", fontsize=8.0, fontweight="bold")

    ax.set_xlim(-0.7, n + 0.05)
    ax.set_ylim(-0.05, n + 0.75)
    ax.set_aspect("equal")
    ax.set_axis_off()


def build(output: Path) -> None:
    _apply_style()
    stress = scenarios.stress()

    # Eight columns rather than six: [matrix, graph, gap] x 3 (the trailing
    # gap dropped). The narrow gap columns separate adjacent pairs so each
    # (matrix, graph) reads as one group.
    fig = plt.figure(figsize=(8.2, 3.4))
    grid = fig.add_gridspec(
        2,
        8,
        width_ratios=[1.35, 1.55, 0.20, 1.35, 1.55, 0.20, 1.35, 1.55],
        height_ratios=[1.0, 1.0],
        left=0.03,
        right=0.99,
        top=0.94,
        bottom=0.05,
        wspace=0.10,
        hspace=0.18,
    )
    matrix_columns = [0, 3, 6]
    pairs = []

    for row, labels in enumerate((["A", "B", "C"], ["D", "E", "F"])):
        for column, label in zip(matrix_columns, labels):
            scenario = stress[label]
            ax_matrix = fig.add_subplot(grid[row, column])
            _draw_matrix(ax_matrix, scenario)
            ax_graph = fig.add_subplot(grid[row, column + 1])
            if label in LAYOUT:
                squeeze = X_SQUEEZE.get(label, 1.0)
                positions = [(x * squeeze, y) for x, y in LAYOUT[label]]
                _draw_graph(ax_graph, scenario, positions, LABEL_PLACEMENT.get(label))
            else:
                _draw_line(ax_graph, scenario)
            pairs.append((row, ax_matrix, ax_graph, label))

    # Label each pair below its own two panels, with one shared baseline
    # per row so the labels line up despite differing panel heights.
    fig.canvas.draw()
    for row in (0, 1):
        in_row = [p for p in pairs if p[0] == row]
        baseline = min(
            min(m.get_position().y0, g.get_position().y0) for _, m, g, _ in in_row
        ) + 0.005
        for _, ax_matrix, ax_graph, label in in_row:
            centre = (ax_matrix.get_position().x0 + ax_graph.get_position().x1) / 2.0
            fig.text(
                centre,
                baseline,
                f"({label})",
                ha="center",
                va="top",
                fontsize=9.0,
                fontweight="bold",
            )

    style.save(fig, output, "E1_topologies", png_dpi=None)
