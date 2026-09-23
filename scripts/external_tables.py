"""Tables for the geographic external validation.

Computes, from the recorded outputs in data/results/external_validation/:
  - means with 95% t-based confidence intervals across the 10 locked refits for
    every performance measure (the external-validation performance table);
  - baseline characteristics of the development and external cohorts (baseline table);
  - discrimination by sex and age group with 95% percentile bootstrap intervals
    from 2000 resamples on refit-averaged predictions (subgroup table).

    python scripts/external_tables.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
E = ROOT / "data" / "results" / "external_validation"
OUT = ROOT / "data" / "results"
METRICS = ["auc", "f1", "brier", "ece", "log_loss", "calibration_intercept", "calibration_slope"]


def t_interval(a):
    a = np.asarray(a, float)
    m = a.mean()
    se = a.std(ddof=1) / np.sqrt(len(a))
    lo, hi = stats.t.interval(0.95, len(a) - 1, loc=m, scale=se)
    return m, lo, hi


def performance() -> pd.DataFrame:
    d = pd.read_csv(E / "ckd_external_metrics_by_seed.csv")
    rows = []
    for method, g in d.groupby("method"):
        row = {"method": method}
        for m in METRICS:
            mean, lo, hi = t_interval(g[m].values)
            row[f"{m}_mean"], row[f"{m}_ci_low"], row[f"{m}_ci_high"] = mean, lo, hi
        rows.append(row)
    return pd.DataFrame(rows)


def baseline() -> pd.DataFrame:
    dev = pd.read_csv(E / "nhanes_ckd_external_development.csv")
    ext = pd.read_csv(E / "knhanes_2019_2020_ckd_external.csv")
    rows = [("No. of participants", f"{len(dev)}", f"{len(ext)}"),
            ("Chronic kidney disease, No. (%)", f"{int(dev.ckd.sum())} ({dev.ckd.mean()*100:.1f})",
             f"{int(ext.ckd.sum())} ({ext.ckd.mean()*100:.1f})"),
            ("Female, No. (%)", f"{int((dev.sex == 2).sum())} ({(dev.sex == 2).mean()*100:.1f})",
             f"{int((ext.sex == 2).sum())} ({(ext.sex == 2).mean()*100:.1f})")]
    for label, col in [("Age, y", "age"), ("Diabetes duration, y", "diabetes_duration"), ("Body mass index", "bmi"),
                       ("Systolic blood pressure, mm Hg", "sbp"), ("Diastolic blood pressure, mm Hg", "dbp"),
                       ("Glycated hemoglobin, %", "hba1c"), ("Total cholesterol, mg/dL", "total_cholesterol"),
                       ("HDL cholesterol, mg/dL", "hdl"), ("Triglycerides, mg/dL", "triglycerides"),
                       ("Uric acid, mg/dL", "uric_acid"), ("Estimated GFR, mL/min/1.73 m2", "egfr")]:
        cells = [label]
        for d in (dev, ext):
            s = pd.to_numeric(d[col], errors="coerce")
            cells.append(f"{s.mean():.1f} ({s.std():.1f})")
        rows.append(tuple(cells))
    cells = ["Urine albumin-to-creatinine ratio, median (IQR), mg/g"]
    for d in (dev, ext):
        s = pd.to_numeric(d.uacr, errors="coerce")
        cells.append(f"{s.median():.1f} ({s.quantile(.25):.1f}-{s.quantile(.75):.1f})")
    rows.append(tuple(cells))
    return pd.DataFrame(rows, columns=["characteristic", "development_cohort", "external_cohort"])


def subgroups(seed: int = 0, n_boot: int = 2000) -> pd.DataFrame:
    ext = pd.read_csv(E / "knhanes_2019_2020_ckd_external.csv")[["participant_id", "sex", "age"]]
    p = pd.read_csv(E / "ckd_external_predictions_averaged.csv").merge(ext, on="participant_id")
    p["agegrp"] = np.where(p.age >= 65, "65 years or older", "20-64 years")
    p["sexlab"] = np.where(p.sex == 2, "Female", "Male")
    rng = np.random.default_rng(seed)
    rows = []
    for method in ["EBRO-MH", "EBM", "Logistic regression"]:
        sub = p[p.method == method]
        for col in ["sexlab", "agegrp"]:
            for level, g in sub.groupby(col):
                y, s = g.y.values, g.probability.values
                auc = roc_auc_score(y, s)
                idx = np.arange(len(y))
                boot = []
                for _ in range(n_boot):
                    j = rng.choice(idx, len(idx), replace=True)
                    if len(np.unique(y[j])) == 2:
                        boot.append(roc_auc_score(y[j], s[j]))
                lo, hi = np.percentile(boot, [2.5, 97.5])
                rows.append(dict(method=method, subgroup=level, n=len(g), events=int(y.sum()),
                                 auc=round(auc, 4), ci_low=round(lo, 4), ci_high=round(hi, 4)))
    return pd.DataFrame(rows)


def main():
    perf = performance()
    perf.to_csv(OUT / "external_performance_with_ci.csv", index=False)
    base = baseline()
    base.to_csv(OUT / "external_baseline_characteristics.csv", index=False)
    sg = subgroups()
    sg.to_csv(OUT / "external_subgroup_auc.csv", index=False)
    pd.set_option("display.width", 160)
    print(perf[["method", "auc_mean", "auc_ci_low", "auc_ci_high"]].round(4).to_string(index=False))
    print()
    print(base.to_string(index=False))
    print()
    print(sg.to_string(index=False))
    print("\nWrote external_performance_with_ci.csv, external_baseline_characteristics.csv, "
          "external_subgroup_auc.csv in data/results/")


if __name__ == "__main__":
    main()
