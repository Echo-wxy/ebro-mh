"""Shared figure style for all figures produced by this package.

Layout target: a single-column page with a usable text width of about
16.5 cm; "narrow" figures are sized at 8.0 cm.
Fonts: Liberation Serif (metric-compatible substitute for Times New Roman).
Output: 600 dpi PNG, vector PDF, and LZW TIFF.

Palette (one colour, one meaning, shared by all four figures):
    MAUVE    EBRO-MH itself and its parts
    STEEL    comparator baseline models
    MINT     data and partitions
    APRICOT  calibration, thresholds and outputs
    DENY     unverified claims / components that never executed
    NEUTRAL  axes, arrows, de-emphasised items
"""
from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt

CM = 1 / 2.54
WIDE = 16.5 * CM
NARROW = 8.0 * CM

SERIF = "Liberation Serif"

# Fixed palette, given as (fill, stroke). Strokes are clean hues: high value,
# medium-high chroma, no black and no grey mixed in. Emphasis is obtained by
# raising saturation within the same hue, never by darkening towards black.
MINT = ("#D7ECE3", "#4FA689")
MAUVE = ("#DCD2EC", "#8E72B8")
STEEL = ("#D6E4F0", "#4F7FA8")
APRICOT = ("#FAE6C4", "#E3A03F")
DENY = ("#F7DEDB", "#C7534A")   # reserved: unverified / never executed
NEUTRAL = "#8A8A8A"             # reserved: "not the focus / comparison only"
FAINT = "#BDBDBD"
AXIS = "#333333"

MAUVE_HI = "#6E51A0"            # same hue, higher saturation, for emphasis
STEEL_HI = "#33628B"

# Backwards-compatible aliases used by the earlier scripts
SLATE = STEEL
WARN = DENY[1]


def use_style(base: float = 8.0) -> None:
    """base is the body font size in points (8 for narrow, 9.5 for wide)."""
    mpl.rcParams.update({
        "font.family": "serif",
        "font.serif": [SERIF],
        "mathtext.fontset": "custom",
        "mathtext.rm": SERIF,
        "mathtext.it": f"{SERIF}:italic",
        "font.size": base,
        "axes.labelsize": base,
        "axes.titlesize": base,
        "xtick.labelsize": base - 1,
        "ytick.labelsize": base - 1,
        "legend.fontsize": base - 1,
        "axes.linewidth": 0.6,
        "axes.edgecolor": AXIS,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": False,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.major.size": 2.5,
        "ytick.major.size": 2.5,
        "xtick.color": AXIS,
        "ytick.color": AXIS,
        "text.color": AXIS,
        "axes.labelcolor": AXIS,
        "legend.frameon": False,
        "lines.linewidth": 1.0,
        "figure.dpi": 120,
        "savefig.dpi": 600,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def panel_label(fig, ax, letter, y=1.012, size=9.5):
    """Panel letter at the top left of a panel, above any legend."""
    box = ax.get_position()
    fig.text(box.x0, y, "(%s)" % letter, fontsize=size, fontweight="bold",
             ha="left", va="bottom", color=AXIS)


def save(fig, stem: str, outdir: str = "figure") -> None:
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(out / f"{stem}.{ext}", bbox_inches="tight", pad_inches=0.05,
                    transparent=False, facecolor="white")
    fig.savefig(out / f"{stem}.tiff", bbox_inches="tight", pad_inches=0.05,
                transparent=False, facecolor="white",
                pil_kwargs={"compression": "tiff_lzw"})
    plt.close(fig)
