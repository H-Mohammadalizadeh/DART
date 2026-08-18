"""Shared figure styling.

Every figure is sized for an IEEE two-column page and rendered at its final
physical width, so it is inserted into the paper without rescaling: type
set at 7 pt here stays 7 pt on the page. A figure that is drawn small and
then stretched by LaTeX magnifies its fonts and line weights and stops
matching its neighbours, which is why the sizes below are exact rather than
convenient.

The look is a closed dark box, no gridlines, and a framed legend.
"""

from __future__ import annotations

import math

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import FixedLocator, FuncFormatter, LogLocator  # noqa: E402

#: Width of one IEEE column, in inches. Figures meant for `\columnwidth`
#: are drawn at exactly this width.
COLUMN_WIDTH = 3.5

#: Ink colour for spines, ticks and annotation.
INK = "#222222"

#: Error-bar styling, kept identical across every figure that shows one.
ERROR_BAR = dict(elinewidth=0.55, capthick=0.55, capsize=0.9, ecolor=INK)

#: Grouped-bar geometry: distance between topology groups, between bars
#: within a group, and the bar width itself.
GROUP_STEP = 0.35
BAR_STEP = 0.05
BAR_WIDTH = 0.032


def apply() -> None:
    """Install the shared rcParams. Call once at the start of every figure.

    Matplotlib's rcParams are global, so a figure that adjusts a setting for
    its own needs would otherwise change every figure drawn after it in the
    same process. Resetting to the library defaults first makes each figure
    independent of the order the set is built in.
    """
    plt.rcParams.update(plt.rcParamsDefault)
    plt.rcParams.update(
        {
            "figure.dpi": 160,
            "savefig.dpi": 600,
            "font.family": "sans-serif",
            "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
            "font.size": 7.0,
            "axes.titlesize": 7.4,
            "axes.labelsize": 7.0,
            "xtick.labelsize": 7.0,
            "ytick.labelsize": 7.0,
            "legend.fontsize": 7.0,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.pad_inches": 0.02,
        }
    )


def box(ax, *, log_y: bool = True, n_decades: int = 4) -> None:
    """Closed dark box, prominent ticks, no gridlines."""
    for side in ("top", "right", "left", "bottom"):
        ax.spines[side].set_visible(True)
        ax.spines[side].set_color(INK)
        ax.spines[side].set_linewidth(1.0)
    ax.grid(False)
    if log_y:
        ax.yaxis.set_major_locator(LogLocator(base=10.0, numticks=n_decades))
    ax.tick_params(axis="x", length=2.4, width=0.9, color=INK, pad=2.0)
    ax.tick_params(axis="y", which="major", length=2.4, width=0.9, color=INK)
    ax.tick_params(axis="y", which="minor", length=1.4, width=0.6, color="#444444")


def frame(legend):
    """Apply the shared framed-legend styling in place."""
    patch = legend.get_frame()
    patch.set_edgecolor(INK)
    patch.set_linewidth(0.8)
    patch.set_facecolor("white")
    patch.set_alpha(0.95)
    return legend


#: Legend styling shared by the swatch-style legends. `columnspacing` is
#: deliberately absent: it is the one setting each figure tunes to its own
#: width, and is passed at the call site.
LEGEND_KW = dict(
    frameon=True,
    fancybox=False,
    handletextpad=0.25,
    handlelength=0.9,
    handleheight=0.7,
    labelspacing=0.25,
    borderpad=0.3,
)


def bar_positions(n_groups: int, index: int, n_bars: int = 4):
    """``(group centres, bar centres)`` for bar `index` of `n_bars`."""
    centres = [g * GROUP_STEP for g in range(n_groups)]
    offset = (index - (n_bars - 1) / 2.0) * BAR_STEP
    return centres, [c + offset for c in centres]


def doubling_ticks(low: float, high: float, *, keep_below: float = 0.9) -> list[float]:
    """Constant-ratio tick values covering ``[low, high]``.

    A doubling sequence is evenly spaced on a log axis and still reads as
    round numbers (250, 500, 1000, ...), which matplotlib's default log
    locator does not manage on the narrow ranges these panels use.

    `keep_below` is how far under `low` a tick may fall and still be kept:
    a value just outside the data range anchors the axis, but one much
    further out wastes a label. Panels with room for a leading tick use a
    looser value than tight ones.
    """
    value = 10 ** math.floor(math.log10(low))
    ticks = []
    while value <= high * 1.05:
        if value >= low * keep_below:
            ticks.append(value)
        value *= 2
    return ticks


def _round_step_ticks(low: float, high: float, count: int) -> list[float]:
    """About `count` round, evenly spaced values inside ``[low, high]``."""
    if low <= 0 or high <= low:
        return [low, high]
    step = (high - low) / count
    magnitude = 10 ** math.floor(math.log10(step))
    for factor in (1, 2, 2.5, 5, 10):
        if (high - low) / (factor * magnitude) <= count + 0.5:
            step = factor * magnitude
            break
    start = math.ceil(low / step) * step
    ticks = []
    value = start
    while value <= high + 1e-9:
        ticks.append(value)
        value += step
    return ticks or [low, high]


def linear_ticks_on_log_axis(ax, count: int = 3, *, scientific: bool = False) -> None:
    """Round, evenly spaced y-ticks on a log axis spanning under a decade.

    Automatic log ticks either vanish or clutter on such ranges. With
    `scientific`, a common power of ten is factored out into a small tag at
    the top-left so the labels read 1.5, 1.75, 2.0 rather than 1500, 1750,
    2000.
    """
    low, high = ax.get_ylim()
    ticks = _round_step_ticks(low, high, count)
    ax.yaxis.set_major_locator(FixedLocator(ticks))
    ax.yaxis.set_minor_locator(FixedLocator([]))
    if scientific and ticks and max(ticks) > 0:
        exponent = int(math.floor(math.log10(max(ticks))))
        if exponent != 0:
            mantissas = [t / 10**exponent for t in ticks]
            decimals = next(
                (d for d in (0, 1, 2) if all(abs(round(m, d) - m) < 1e-9 for m in mantissas)),
                2,
            )
            ax.yaxis.set_major_formatter(
                FuncFormatter(lambda v, _: f"{v / 10**exponent:.{decimals}f}")
            )
            ax.text(
                0.0,
                1.03,
                rf"$\times10^{{{exponent}}}$",
                transform=ax.transAxes,
                ha="left",
                va="bottom",
                fontsize=6.2,
                color=INK,
            )
            return
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{int(round(v))}"))


def integer_ticks(axis, ticks: list[float]) -> None:
    """Fixed ticks with integer labels, used on both axes of a log scatter."""
    axis.set_major_locator(FixedLocator(ticks))
    axis.set_minor_locator(FixedLocator([]))
    axis.set_major_formatter(FuncFormatter(lambda v, _: f"{int(round(v))}"))


def panel_letter(ax, text: str, *, y: float = -0.34, size: float = 7.4) -> None:
    """Bold ``(A)``-style label centred below a panel."""
    ax.annotate(
        text,
        xy=(0.5, y),
        xycoords="axes fraction",
        ha="center",
        va="top",
        fontsize=size,
        fontweight="bold",
    )


def save(fig, directory, name: str, *, png_dpi: float | None = 300) -> None:
    """Write `name`.pdf and `name`.png, then close the figure.

    The PDF is what the paper includes; the PNG is a preview. `png_dpi` of
    `None` falls back to the rcParam, which the full-page topology figure
    uses to keep its small type legible when previewed.
    """
    directory.mkdir(parents=True, exist_ok=True)
    fig.savefig(directory / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(directory / f"{name}.png", dpi=png_dpi, bbox_inches="tight")
    plt.close(fig)
