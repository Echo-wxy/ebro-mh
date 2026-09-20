"""fixed-split sensitivity analyses, three panels.

Both curves are EBRO-MH, so both carry the EBRO-MH hue; the two datasets are
separated by line style and marker fill, and the default setting used in all
other analyses is marked in every panel.

Source: scripts/data/sensitivity_auc.csv
"""
import pandas as pd
import matplotlib.pyplot as plt

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ebro_mh import figstyle as fs

fs.use_style(base=9.5)
d = pd.read_csv(Path(__file__).resolve().parents[1] / "data" / "results" / "sensitivity_auc.csv")

PANELS = [("training_fraction", "Fraction of the fitting partition", 1.00, False),
          ("particles", "Number of particles", 20, True),
          ("batch_size", "Validation batch size", 64, True)]

STYLE = {"PIDD": dict(ls="-", lw=1.5, marker="o", ms=3.6, mfc=fs.MAUVE[1], mew=0.9),
         "NHANES diabetes": dict(ls=(0, (4.5, 2.2)), lw=1.3, marker="s", ms=3.6,
                                 mfc="white", mew=0.9)}

fig, axes = plt.subplots(1, 3, figsize=(16.5 * fs.CM, 6.4 * fs.CM), sharey=True)
fig.subplots_adjust(top=0.845, bottom=0.215, left=0.062, right=0.993, wspace=0.09)

for ax, (key, xlabel, default, logx) in zip(axes, PANELS):
    for spine in ax.spines.values():          # closed frame for panelled figures
        spine.set_visible(True)
        spine.set_linewidth(0.6)
        spine.set_color(fs.AXIS)

    ax.axvline(default, color=fs.FAINT, lw=0.7, ls=(0, (1, 2.5)), zorder=1)

    for name in ("PIDD", "NHANES diabetes"):
        s = dict(STYLE[name])
        sub = d[(d.dataset == name) & (d.experiment == key)].sort_values("value")
        ax.plot(sub.value, sub.auc, color=fs.MAUVE[1], mec=fs.MAUVE[1],
                zorder=3 if name == "PIDD" else 4, label=name, **s)

    if logx:
        ax.set_xscale("log")
        ticks = sorted(d[d.experiment == key].value.unique())
        ax.set_xticks(ticks)
        ax.set_xticklabels(["%d" % t for t in ticks])
        ax.minorticks_off()
        ax.set_xlim(min(ticks) * 0.80, max(ticks) * 1.25)
    else:
        ax.set_xticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_xlim(-0.03, 1.08)

    ax.set_xlabel(xlabel)
    ax.set_ylim(0.54, 0.80)
    ax.set_yticks([0.55, 0.60, 0.65, 0.70, 0.75, 0.80])
    ax.tick_params(axis="y", which="both", right=False)

    # default marker, placed in every panel, away from the curves
    xt = ax.get_xlim()
    ax.annotate("Default", xy=(default, 0.5445), xytext=(-3, 0), fontsize=7.5,
                textcoords="offset points", color=fs.NEUTRAL, ha="right",
                va="bottom", zorder=5)

axes[0].set_ylabel("AUC")

handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.985),
           ncol=2, handlelength=2.4, columnspacing=2.0, handletextpad=0.5,
           borderaxespad=0.0)
for ax, letter in zip(axes, "abc"):
    fs.panel_label(fig, ax, letter, y=0.995, size=9.5)

fs.save(fig, "sensitivity", outdir=str(Path(__file__).resolve().parents[1] / "outputs"))
print("wrote outputs/sensitivity.png, .pdf and .tiff")
