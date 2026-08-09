"""Figure style and building blocks.

House rules (applied by `use_paper_style`):

* Figure widths are quoted in inches that match a two-column ACL/NeurIPS layout
  (3.25in single column, 6.9in full width) so nothing is rescaled in LaTeX and
  font sizes in the figure match the body text.
* Series identity is never carried by colour alone: every multi-series plot gets
  both a marker shape and a direct label at the end of the line.
* No truncated value axes on bar charts, and zero is always in view when the
  sign of a quantity is the point being made.

The categorical hues are the first, second, third and seventh slots of a
palette validated for colour-vision deficiency (adjacent- and all-pairs OKLab
ΔE, chroma floor, contrast against the paper's white surface).
"""

from __future__ import annotations

import os
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

# Validated categorical slots (light surface).
BLUE = "#2a78d6"
ORANGE = "#eb6834"
AQUA = "#1baf7a"
VIOLET = "#4a3aa7"
RED = "#e34948"
YELLOW = "#eda100"
GREY = "#5a5a5a"
INK = "#1a1a1a"
MUTED = "#767676"
RULE = "#d8d8d8"

FAMILY_COLORS: Dict[str, str] = {
    "SmolLM2": BLUE,
    "Qwen2.5": ORANGE,
    "Qwen3": AQUA,
    "Llama-3.2": VIOLET,
    "Qwen2.5-Coder": GREY,
    "OLMo-2": RED,
    "Falcon3": YELLOW,
}
FAMILY_MARKERS: Dict[str, str] = {
    "SmolLM2": "o",
    "Qwen2.5": "s",
    "Qwen3": "^",
    "Llama-3.2": "D",
    "Qwen2.5-Coder": "v",
    "OLMo-2": "P",
    "Falcon3": "X",
}

COL_WIDTH = 3.25
FULL_WIDTH = 6.9


def use_paper_style() -> None:
    mpl.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.02,
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
            "font.size": 8,
            "axes.titlesize": 8.5,
            "axes.labelsize": 8,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "legend.fontsize": 7.5,
            "axes.edgecolor": "#4a4a4a",
            "axes.linewidth": 0.7,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.labelcolor": INK,
            "text.color": INK,
            "xtick.color": "#4a4a4a",
            "ytick.color": "#4a4a4a",
            "xtick.major.width": 0.7,
            "ytick.major.width": 0.7,
            "xtick.major.size": 3,
            "ytick.major.size": 3,
            "grid.color": RULE,
            "grid.linewidth": 0.6,
            "legend.frameon": False,
            "lines.linewidth": 1.6,
            "lines.markersize": 4.5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def save(fig, path_stem: str, formats: Sequence[str] = ("pdf", "png")) -> List[str]:
    """Write vector + raster copies of a figure; returns the paths written."""
    os.makedirs(os.path.dirname(path_stem) or ".", exist_ok=True)
    out = []
    for ext in formats:
        p = f"{path_stem}.{ext}"
        fig.savefig(p)
        out.append(p)
    plt.close(fig)
    return out


def zero_rule(ax, y: float = 0.0, label: Optional[str] = None) -> None:
    """A reference line at y (usually zero), drawn behind the data."""
    ax.axhline(y, color="#9a9a9a", linewidth=0.9, linestyle=(0, (4, 3)), zorder=0)
    if label:
        ax.annotate(
            label,
            xy=(0.995, y),
            xycoords=("axes fraction", "data"),
            ha="right",
            va="bottom",
            fontsize=6.8,
            color=MUTED,
        )


def band(ax, x, lo, hi, color: str, alpha: float = 0.14) -> None:
    ax.fill_between(x, lo, hi, color=color, alpha=alpha, linewidth=0, zorder=1)


def direct_label(
    ax, x: float, y: float, text: str, color: str, dx: float = 0.55, va: str = "center"
) -> None:
    ax.annotate(
        text,
        xy=(x, y),
        xytext=(dx, 0),
        textcoords="offset fontsize",
        color=color,
        fontsize=7.2,
        va=va,
        ha="left",
        fontweight="medium",
    )


def labels_no_overlap(
    ax,
    items: Sequence[Tuple[float, float, str, str]],
    min_sep_frac: float = 0.085,
    dx: float = 0.55,
) -> None:
    """Place end-of-series labels, nudging them apart when they would collide.

    `items` are (x, y, text, colour) in data coordinates. Labels are pushed
    apart vertically by at least `min_sep_frac` of the axis height and a thin
    leader is drawn to any label that had to move, so the reader can still tell
    which mark it belongs to.
    """
    if not items:
        return
    lo, hi = ax.get_ylim()
    span = hi - lo
    min_sep = min_sep_frac * span
    order = sorted(range(len(items)), key=lambda i: items[i][1])
    placed = [items[i][1] for i in order]
    for k in range(1, len(placed)):
        if placed[k] - placed[k - 1] < min_sep:
            placed[k] = placed[k - 1] + min_sep
    # Re-centre so the block does not drift off the top of the axis.
    overshoot = placed[-1] - (hi - 0.02 * span)
    if overshoot > 0:
        placed = [p - overshoot for p in placed]
    for k, i in enumerate(order):
        x, y, text, color = items[i]
        yl = placed[k]
        ax.annotate(
            text,
            xy=(x, yl),
            xytext=(dx, 0),
            textcoords="offset fontsize",
            color=color,
            fontsize=7.2,
            va="center",
            ha="left",
            fontweight="medium",
        )
        if abs(yl - y) > 0.02 * span:
            ax.plot(
                [x, x], [y, yl], color=color, linewidth=0.5, alpha=0.55,
                zorder=1, solid_capstyle="butt",
            )


def param_axis(ax, values_b: Iterable[float]) -> None:
    """Log-scaled parameter-count axis with readable tick labels."""
    vals = sorted(set(float(v) for v in values_b))
    ax.set_xscale("log")
    ticks = [0.135, 0.25, 0.5, 1.0, 2.0, 3.0]
    ticks = [t for t in ticks if min(vals) * 0.7 <= t <= max(vals) * 1.4]
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"{t:g}B" if t >= 1 else f"{int(t*1000)}M" for t in ticks])
    ax.minorticks_off()


def grouped_bars(
    ax,
    groups: Sequence[str],
    series: Sequence[Tuple[str, Sequence[float], Optional[Sequence[Tuple[float, float]]], str]],
    width: float = 0.8,
    gap: float = 0.06,
) -> None:
    """Grouped bars with a 2px surface gap between neighbours and CI whiskers.

    `series` entries are (label, values, cis_or_None, colour).
    """
    n = len(series)
    slot = width / n
    xs = np.arange(len(groups), dtype=float)
    for i, (label, values, cis, color) in enumerate(series):
        off = -width / 2 + slot * (i + 0.5)
        pos = xs + off
        ax.bar(
            pos,
            values,
            width=slot - gap,
            color=color,
            label=label,
            linewidth=0,
            zorder=2,
        )
        if cis is not None:
            lo = np.array([c[0] for c in cis], dtype=float)
            hi = np.array([c[1] for c in cis], dtype=float)
            v = np.asarray(values, dtype=float)
            ax.errorbar(
                pos,
                v,
                yerr=[v - lo, hi - v],
                fmt="none",
                ecolor="#3a3a3a",
                elinewidth=0.8,
                capsize=1.8,
                capthick=0.8,
                zorder=3,
            )
    ax.set_xticks(xs)
    ax.set_xticklabels(groups)
