"""Shared matplotlib styling for report figures.

One place for colors/chrome so every notebook's charts look like the same
report. Sequential blue for magnitude (heatmaps); three categorical hues for
classifier comparisons (never more than logreg/svm/nb -- 3 series -- in one
figure, so we don't need a wider categorical ramp).
"""
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

CATEGORICAL = ["#2a78d6", "#1baf7a", "#eda100"]  # blue, aqua, yellow
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
AXIS = "#c3c2b7"
SURFACE = "#fcfcfb"

# Sequential blue ramp (light -> dark), for heatmaps / magnitude encodings.
SEQUENTIAL_BLUE = LinearSegmentedColormap.from_list(
    "sequential_blue",
    ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"],
)


def style_axes(ax):
    """Recessive gridlines/spines, muted ticks -- apply to every report figure."""
    ax.set_facecolor(SURFACE)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(AXIS)
    ax.spines["bottom"].set_color(AXIS)
    ax.tick_params(colors=INK_SECONDARY, labelsize=9)
    ax.grid(axis="y", color=GRIDLINE, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    return ax


def new_fig(figsize=(8, 5)):
    fig, ax = plt.subplots(figsize=figsize, facecolor=SURFACE)
    style_axes(ax)
    return fig, ax
