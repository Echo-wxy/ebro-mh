"""Decision curves for every evaluation setting.

Net benefit is computed on predictions averaged across repeats (or across the
10 locked refits for the external cohort), over threshold probabilities from
0.02 to 0.50, for each method and for the treat-all strategy. The comparator in
each setting is the method whose repeat-averaged predictions had the highest
AUC: XGBoost for NHANES diabetes and EBM otherwise.

    python scripts/decision_curves.py

Writes data/results/decision_curves.csv and draws the two decision-curve figures
into outputs/.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

R = ROOT / "data" / "results"
M = R / "main_analyses"
THRESHOLDS = np.arange(0.02, 0.51, 0.01)
COMPARATOR = {
    "PIDD cross-validation": "EBM",
    "NHANES diabetes cross-validation": "XGBoost",
    "NHANES CKD cross-validation": "EBM",
    "Temporal validation": "EBM",
    "Geographic external validation": "EBM",
}


def net_benefit(y, p):
    n, prev = len(y), y.mean()
    rows = []
    for t in THRESHOLDS:
        flag = p >= t
        tp = np.sum(flag & (y == 1))
        fp = np.sum(flag & (y == 0))
        rows.append((t, tp / n - fp / n * (t / (1 - t)), prev - (1 - prev) * (t / (1 - t))))
    return pd.DataFrame(rows, columns=["threshold", "net_benefit", "treat_all"])


def averaged(df, ycol, pcol):
    return df.groupby("participant_id").agg(y=(ycol, "first"), p=(pcol, "mean")).reset_index()


def compute() -> pd.DataFrame:
    sources = {
        "PIDD cross-validation": M / "pidd" / "oof_predictions.csv.gz",
        "NHANES diabetes cross-validation": M / "nhanes_dm" / "oof_predictions.csv.gz",
        "NHANES CKD cross-validation": M / "nhanes_ckd" / "oof_predictions.csv.gz",
        "Temporal validation": M / "temporal" / "temporal_predictions.csv.gz",
    }
    out = []
    for analysis, path in sources.items():
        d = pd.read_csv(path)
        for method in d.method.unique():
            s = averaged(d[d.method == method], "y_true", "y_prob")
            c = net_benefit(s.y.values, s.p.values)
            c["analysis"], c["method"] = analysis, method
            out.append(c)
    e = pd.read_csv(R / "external_validation" / "ckd_external_predictions_averaged.csv")
    for method in e.method.unique():
        s = e[e.method == method]
        c = net_benefit(s.y.values, s.probability.values)
        c["analysis"], c["method"] = "Geographic external validation", method
        out.append(c)
    return pd.concat(out, ignore_index=True)


def summarize(d: pd.DataFrame) -> pd.DataFrame:
    """Shortfall of EBRO-MH against the prespecified comparator in each setting."""
    rows = []
    for analysis, comp in COMPARATOR.items():
        sub = d[d.analysis == analysis]
        e = sub[sub.method == "EBRO-MH"].set_index("threshold").net_benefit
        c = sub[sub.method == comp].set_index("threshold").net_benefit
        diff = c - e
        i20 = int(np.argmin(np.abs(diff.index.values - 0.20)))
        rows.append(dict(analysis=analysis, comparator=comp,
                         shortfall_at_0_20=round(float(diff.iloc[i20]), 4),
                         max_excess_of_ebro=round(float(max(0.0, (-diff).max())), 4),
                         lowest_threshold_always_below=float(min(t for t in diff.index
                                                                 if (diff[diff.index >= t] > 1e-9).all()))))
    return pd.DataFrame(rows)


def plot(d: pd.DataFrame, out_dir: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from ebro_mh import figstyle as fs

    def panel(ax, analysis):
        comp = COMPARATOR[analysis]
        sub = d[d.analysis == analysis]
        ta = sub[sub.method == comp]
        ax.axhline(0, color=fs.FAINT, lw=0.8)
        ax.plot(ta.threshold, ta.treat_all, color=fs.NEUTRAL, lw=1.1, ls=(0, (4.5, 2.2)), label="Treat all")
        ax.plot(ta.threshold, ta.net_benefit, color=fs.STEEL[1], lw=1.4, ls=(0, (4.5, 2.2)), label=comp)
        e = sub[sub.method == "EBRO-MH"]
        ax.plot(e.threshold, e.net_benefit, color=fs.MAUVE[1], lw=1.8, label="EBRO-MH")
        ax.set_xlim(0.02, 0.50)
        ax.set_xlabel("Threshold probability")

    fs.use_style(base=9.5)
    fig, axes = plt.subplots(1, 2, figsize=(16.5 * fs.CM, 7.6 * fs.CM))
    for ax, a in zip(axes, ["Temporal validation", "Geographic external validation"]):
        panel(ax, a)
    axes[0].set_ylabel("Net benefit")
    axes[0].legend(frameon=False, fontsize=8)
    fs.save(fig, "decision_curves_main", outdir=str(out_dir))

    fig, axes = plt.subplots(1, 3, figsize=(16.5 * fs.CM, 6.6 * fs.CM))
    for ax, a in zip(axes, ["PIDD cross-validation", "NHANES diabetes cross-validation", "NHANES CKD cross-validation"]):
        panel(ax, a)
        ax.set_title(a.replace(" cross-validation", ""), fontsize=8)
    axes[0].set_ylabel("Net benefit")
    fs.save(fig, "decision_curves_cross_validation", outdir=str(out_dir))


def main():
    d = compute()
    d.to_csv(R / "decision_curves.csv", index=False)
    s = summarize(d)
    s.to_csv(R / "decision_curves_summary.csv", index=False)
    print(s.to_string(index=False))
    out = ROOT / "outputs"
    out.mkdir(exist_ok=True)
    plot(d, out)
    print("\nWrote data/results/decision_curves.csv, data/results/decision_curves_summary.csv, and figures in outputs/")


if __name__ == "__main__":
    main()
