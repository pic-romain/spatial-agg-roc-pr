"""Centralized plotting style for article figures.

Import this module at the top of any figure script in this folder and call
:func:`apply_style` once, before creating figures::

    from style import apply_style, MODEL_COLORS, LW, savefig
    apply_style()

It has no third-party style dependency and deliberately matches the look of
the scripts in the top-level ``plots/`` folder: plain matplotlib defaults
(sans-serif, outward ticks, no grid) with the project's shared conventions
layered on -- the HRES/GraphCast/Pangu colour assignment, the 12/13/10 font
sizes, line widths, axis labels and save settings -- so every article figure
in ``z_dcp-figures`` matches the rest of the paper.
"""

import string

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# --------------------------------------------------------------------------- #
# Shared palette and drawing constants
# --------------------------------------------------------------------------- #
# Model -> colour. Kept identical to the historical convention
# (colors = ['black', 'crimson', 'forestgreen', 'orange']) so new figures stay
# consistent with ones already in the paper.
MODEL_COLORS = {
    "HRES": "black",
    "GraphCast": "crimson",
    "Pangu": "forestgreen",
}
# Positional palette (HRES, GraphCast, Pangu, spare) for code that indexes by
# model position rather than by name.
COLORS = ["black", "crimson", "forestgreen", "orange"]

# Line style per (PAV, interpolation) variant, shared by the "pairs" plots.
VARIANT_LS = {
    ("PAV", "interp"): "solid",
    ("PAV", "nointerp"): "dashed",
    ("noPAV", "interp"): "dotted",
    ("noPAV", "nointerp"): "dashdot",
}

LW = 1.5          # default line width for curves
ALPHA = 1.0       # default curve alpha
PANEL_SIZE = 4.0  # side length (inches) of one square ROC/PR panel
DPI = 300         # raster export resolution

# Canonical axis labels, so every ROC/PR panel is worded identically.
LABELS = {
    "fpr": "False alarm rate",
    "tpr": "Hit rate",
    "recall": "Recall",
    "precision": "Precision",
}


# --------------------------------------------------------------------------- #
# Style activation
# --------------------------------------------------------------------------- #
def apply_style():
    """Activate the centralized article style.

    Safe to call more than once. Starts from matplotlib's defaults (to match
    the look of the top-level ``plots/`` scripts) and only overrides the
    project's shared conventions: font sizes, line width, palette and export
    dpi.
    """
    mpl.rcParams.update(mpl.rcParamsDefault)  # start from a clean default look
    mpl.rcParams.update({
        # Figure / export
        "savefig.dpi": DPI,
        "savefig.bbox": "tight",
        # Fonts (points), matching the hard-coded sizes used across plots/:
        # labels 12, titles 13, legend/ticks 10.
        "axes.titlesize": 13,
        "axes.labelsize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        # Lines
        "lines.linewidth": LW,
        # Colour cycle; models are also coloured explicitly via MODEL_COLORS.
        "axes.prop_cycle": mpl.cycler(color=COLORS),
    })


def panel_grid(nrows, ncols, **kwargs):
    """``plt.subplots`` with the standard square-panel figure size."""
    kwargs.setdefault("figsize", (PANEL_SIZE * ncols, PANEL_SIZE * nrows))
    return plt.subplots(nrows, ncols, **kwargs)


def model_legend_handles(models=("HRES", "GraphCast", "Pangu"), lw=LW, ls="solid"):
    """Line2D handles coloured per model, for a shared figure legend."""
    return [
        Line2D([0], [0], color=MODEL_COLORS[m], lw=lw, ls=ls, label=m)
        for m in models
    ]


def add_panel_labels(axes, labels=None, offset=(-6, 6), loc="upper left",
                     fontweight="bold", **kwargs):
    """Annotate each axis with a subfigure label: (a), (b), (c), ...

    Labels are placed in reading order (row-major) just outside the top corner
    given by ``loc`` -- in the margin above the tick labels, so they clear
    long centred panel titles. ``axes`` may be a single Axes, a list, or the
    2-D array returned by :func:`plt.subplots`.

    Parameters
    ----------
    labels : sequence of str, optional
        Custom labels; defaults to ``['(a)', '(b)', ...]``.
    offset : (float, float)
        Offset in points from the chosen corner.
    loc : {'upper left', 'upper right'}
        Which corner to anchor to.
    """
    axes_flat = np.atleast_1d(np.asarray(axes, dtype=object)).ravel()
    if labels is None:
        labels = [f"({c})" for c in string.ascii_lowercase]

    # Anchor at the top corner and extend *outward* (left/right) into the
    # margin so the label never sits under a centred title.
    xy, ha = {"upper left": ((0.0, 1.0), "right"),
              "upper right": ((1.0, 1.0), "left")}[loc]
    kwargs.setdefault("fontsize", mpl.rcParams["axes.titlesize"])

    for ax, lab in zip(axes_flat, labels):
        ax.annotate(lab, xy=xy, xycoords="axes fraction",
                    xytext=offset, textcoords="offset points",
                    ha=ha, va="bottom", fontweight=fontweight, **kwargs)
    return axes


def savefig(fig, path, **kwargs):
    """Save ``fig`` with the project defaults (dpi, tight bbox)."""
    kwargs.setdefault("dpi", DPI)
    kwargs.setdefault("bbox_inches", "tight")
    fig.savefig(path, **kwargs)
