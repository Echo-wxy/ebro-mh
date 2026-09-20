"""One-command demo on synthetic data.

Fits the rule ensemble and a logistic regression comparator on a synthetic
development cohort, applies both to a synthetic external cohort with a shifted
case mix, writes a metrics table, and draws the calibration figure.

    python scripts/run_demo.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ebro_mh.baselines.comparators import fit_locked
from ebro_mh.data.synthetic import FEATURES, make_development_and_external
from ebro_mh.metrics import calibration_bins, classification_metrics

OUT = Path(__file__).resolve().parents[1] / "outputs"


def split(d, seed):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(d))
    n_val = int(0.15 * len(d))
    n_cal = int(0.15 * len(d))
    val = d.iloc[idx[:n_val]]
    cal = d.iloc[idx[n_val:n_val + n_cal]]
    fit = d.iloc[idx[n_val + n_cal:]]
    return fit, val, cal


def main(seed: int = 7) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    cfg = yaml.safe_load((Path(__file__).resolve().parents[1] / "configs" / "quick.yaml").read_text())
    models_cfg = cfg["models"]

    dev, ext = make_development_and_external(seed=seed)
    fit, val, cal = split(dev, seed)

    rows, curves = [], {}
    for name in ("ebro_mh", "logistic_regression"):
        model = fit_locked(
            name, fit, fit["ckd"].values, val, val["ckd"].values,
            cal, cal["ckd"].values, FEATURES, models_cfg, seed,
        )
        p = model.predict_proba(ext)
        yhat = model.predict(ext)
        m = classification_metrics(ext["ckd"].values, p, yhat)
        rows.append({"method": name, **m})
        curves[name] = calibration_bins(ext["ckd"].values, p, n_bins=10)

    table = pd.DataFrame(rows)
    table.to_csv(OUT / "demo_metrics.csv", index=False)
    print(table.to_string(index=False))

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from ebro_mh import figstyle as fs

        fs.use_style(base=9.5)
        fig, ax = plt.subplots(figsize=(8.0 * fs.CM, 8.0 * fs.CM))
        ax.plot([0, 1], [0, 1], ls=(0, (1, 2.5)), lw=0.7, color=fs.NEUTRAL)
        for name, style in (("ebro_mh", fs.MAUVE[1]), ("logistic_regression", fs.STEEL[1])):
            c = curves[name]
            ax.plot(c["mean_pred"], c["observed"], marker="o", ms=3.4, lw=1.5,
                    color=style, label=name.replace("_", " "))
        ax.set_xlabel("Mean predicted probability")
        ax.set_ylabel("Observed outcome rate")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect("equal", adjustable="box")
        fig.legend(loc="upper center", ncol=2, borderaxespad=0.0)
        fs.save(fig, "demo_calibration", outdir=str(OUT))
        print(f"wrote {OUT/'demo_calibration.png'}")
    except Exception as exc:  # plotting is optional for the demo
        print(f"figure skipped: {exc}")


if __name__ == "__main__":
    main()
