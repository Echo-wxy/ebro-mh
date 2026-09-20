"""temporal-validation calibration, two panels.

(a) reliability curves for EBRO-MH and EBM
(b) calibration gap (observed rate minus mean predicted probability) for the
    same bins, which makes the direction and size of the miscalibration
    readable instead of leaving it to be eyeballed against the diagonal.

Source: scripts/data/calibration_bins.csv (10 equal-count bins per method,
temporal validation NHANES 2015-2016 -> 2017-2018).
"""
import pandas as pd
import matplotlib.pyplot as plt

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ebro_mh import figstyle as fs

fs.use_style(base=9.5)
bins = pd.read_csv(Path(__file__).resolve().parents[1] / "data" / "results" / "calibration_bins.csv")

M = {"EBRO-MH": dict(color=fs.MAUVE[1], ls="-", lw=1.7, marker="o", ms=3.4,
                     mfc=fs.MAUVE[1], mec=fs.MAUVE[1], z=5),
     "EBM": dict(color=fs.STEEL[1], ls=(0, (4.5, 2.2)), lw=1.3, marker="s",
                 ms=3.0, mfc="white", mec=fs.STEEL[1], z=4)}

fig, axes = plt.subplots(1, 2, figsize=(16.5 * fs.CM, 9.3 * fs.CM))
fig.subplots_adjust(top=0.885, bottom=0.115, left=0.060, right=0.988, wspace=0.24)

# ---------------- panel (a): reliability curves ----------------
ax = axes[0]
lim = 0.64
ax.plot([0, lim], [0, lim], ls=(0, (1, 2.5)), lw=0.7, color=fs.NEUTRAL, zorder=1)
ax.text(0.50, 0.535, "Perfect calibration", fontsize=7, color=fs.NEUTRAL,
        rotation=45, rotation_mode="anchor", ha="center", va="bottom")
for name in ("EBM", "EBRO-MH"):
    d = bins[bins.method == name].sort_values("mean_predicted_probability")
    s = M[name]
    ax.plot(d.mean_predicted_probability, d.observed_rate, lw=s["lw"], ls=s["ls"],
            color=s["color"], marker=s["marker"], ms=s["ms"], mfc=s["mfc"],
            mec=s["mec"], mew=0.8, zorder=s["z"], label=name)
ax.set_xlim(0, lim)
ax.set_ylim(0, lim)
ax.set_xticks([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
ax.set_yticks([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
ax.set_xlabel("Mean predicted probability")
ax.set_ylabel("Observed outcome rate")
ax.set_aspect("equal", adjustable="box")

# ---------------- panel (b): calibration gap ----------------
ax = axes[1]
ax.axhline(0, ls=(0, (1, 2.5)), lw=0.7, color=fs.NEUTRAL, zorder=1)
ax.text(0.628, 0.005, "No gap", fontsize=7, color=fs.NEUTRAL, ha="right", va="bottom")
for name in ("EBM", "EBRO-MH"):
    d = bins[bins.method == name].sort_values("mean_predicted_probability")
    gap = d.observed_rate - d.mean_predicted_probability
    s = M[name]
    ax.plot(d.mean_predicted_probability, gap, lw=s["lw"], ls=s["ls"],
            color=s["color"], marker=s["marker"], ms=s["ms"], mfc=s["mfc"],
            mec=s["mec"], mew=0.8, zorder=s["z"], label=name)
ax.set_xlim(0, lim)
ax.set_ylim(-0.09, 0.09)
ax.set_xticks([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
ax.set_yticks([-0.08, -0.04, 0.00, 0.04, 0.08])
ax.set_xlabel("Mean predicted probability")
ax.set_ylabel("Observed rate minus predicted probability")
ax.annotate("Risk understated", xy=(0.018, 0.074), fontsize=7, color=fs.NEUTRAL,
            ha="left", va="center")
ax.annotate("Risk overstated", xy=(0.018, -0.078), fontsize=7, color=fs.NEUTRAL,
            ha="left", va="center")

# ---------------- legend above, panel letters above that ----------------
handles, labels = axes[0].get_legend_handles_labels()
order = [labels.index(x) for x in ("EBRO-MH", "EBM")]
fig.legend([handles[i] for i in order], [labels[i] for i in order],
           loc="upper center", bbox_to_anchor=(0.5, 0.985), ncol=2,
           handlelength=2.2, columnspacing=1.8, handletextpad=0.5,
           borderaxespad=0.0)
for ax, letter in zip(axes, "ab"):
    fs.panel_label(fig, ax, letter, y=0.995, size=9.5)

fs.save(fig, "calibration", outdir=str(Path(__file__).resolve().parents[1] / "outputs"))
print("wrote outputs/calibration.png, .pdf and .tiff")
