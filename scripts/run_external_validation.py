from __future__ import annotations
from pathlib import Path
import time
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, f1_score
from statsmodels.stats.multitest import multipletests
from ebro_mh.baselines.comparators import fit_locked
from ebro_mh.metrics import classification_metrics, calibration_bins
from ebro_mh.common import ensure_dir, write_json

MODEL_NAMES = [
    "logistic_regression", "random_forest", "xgboost", "rulefit",
    "bayesian_rule_lists", "ebm", "ebro_mh"
]
DISPLAY = {
    "logistic_regression": "Logistic regression",
    "random_forest": "Random forest",
    "xgboost": "XGBoost",
    "rulefit": "RuleFit",
    "bayesian_rule_lists": "Bayesian Rule Lists",
    "ebm": "EBM",
    "ebro_mh": "EBRO-MH",
}


def inner_split(d, ycol, seed, val_frac, cal_frac):
    train, temp = train_test_split(
        d,
        test_size=val_frac + cal_frac,
        stratify=d[ycol],
        random_state=seed,
    )
    rel = cal_frac / (val_frac + cal_frac)
    val, cal = train_test_split(
        temp,
        test_size=rel,
        stratify=temp[ycol],
        random_state=seed + 17,
    )
    return train, val, cal


def _mean_ci95(values):
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return np.nan, np.nan, np.nan, np.nan
    mean = float(np.mean(x))
    sd = float(np.std(x, ddof=1)) if len(x) > 1 else 0.0
    if len(x) > 1:
        se = sd / np.sqrt(len(x))
        q = float(stats.t.ppf(0.975, df=len(x) - 1))
        lo, hi = mean - q * se, mean + q * se
    else:
        lo = hi = mean
    return mean, sd, float(lo), float(hi)


def summarize_seed_metrics(by_seed: pd.DataFrame) -> pd.DataFrame:
    metric_cols = [
        "auc", "f1", "brier", "ece", "log_loss",
        "calibration_intercept", "calibration_slope", "fit_seconds",
    ]
    rows = []
    for method, g in by_seed.groupby("method", sort=False):
        row = {
            "method": method,
            "seeds": int(g["seed"].nunique()),
            "development_n": int(g["development_n"].iloc[0]),
            "development_positive": int(g["development_positive"].iloc[0]),
            "external_n": int(g["external_n"].iloc[0]),
            "external_positive": int(g["external_positive"].iloc[0]),
            "external_prevalence": float(g["external_prevalence"].iloc[0]),
        }
        for c in metric_cols:
            mean, sd, lo, hi = _mean_ci95(g[c])
            row[f"{c}_mean"] = mean
            row[f"{c}_sd"] = sd
            row[f"{c}_ci95_low"] = lo
            row[f"{c}_ci95_high"] = hi
        rows.append(row)
    return pd.DataFrame(rows).sort_values("auc_mean", ascending=False).reset_index(drop=True)


def run_ckd_external(dev, test, features, cfg, outdir):
    ycol = "ckd"
    out = ensure_dir(outdir)
    rows, preds, bins_rows, fitted_meta = [], [], [], []

    for si, seed in enumerate(cfg["seed_list"], 1):
        fit, val, cal = inner_split(
            dev,
            ycol,
            seed,
            cfg["evaluation"]["inner_validation_fraction"],
            cfg["evaluation"]["inner_calibration_fraction"],
        )
        for mi, name in enumerate(MODEL_NAMES, 1):
            print(
                f"[CKD] seed {si}/{len(cfg['seed_list'])} | model {mi}/7 {DISPLAY[name]}",
                flush=True,
            )
            t0 = time.perf_counter()
            model = fit_locked(
                name,
                fit,
                fit[ycol].astype(int),
                val,
                val[ycol].astype(int),
                cal,
                cal[ycol].astype(int),
                features,
                cfg["models"],
                seed,
            )
            p = model.predict_proba(test)
            b = model.predict(test)
            elapsed = time.perf_counter() - t0
            m = classification_metrics(
                test[ycol].astype(int), p, b, cfg["evaluation"]["ece_bins"]
            )
            rows.append(
                {
                    "seed": seed,
                    "method": DISPLAY[name],
                    "development_n": len(dev),
                    "development_positive": int(dev[ycol].sum()),
                    "external_n": len(test),
                    "external_positive": int(test[ycol].sum()),
                    "external_prevalence": float(test[ycol].mean()),
                    "threshold": model.threshold,
                    "fit_seconds": elapsed,
                    **m,
                }
            )
            for pid, yy, pp, bb in zip(
                test.participant_id, test[ycol].astype(int), p, b
            ):
                preds.append(
                    {
                        "seed": seed,
                        "method": DISPLAY[name],
                        "participant_id": pid,
                        "y": int(yy),
                        "probability": float(pp),
                        "prediction": int(bb),
                    }
                )
            cb = calibration_bins(
                test[ycol].astype(int), p, cfg["evaluation"]["ece_bins"]
            )
            cb["seed"] = seed
            cb["method"] = DISPLAY[name]
            bins_rows.extend(cb.to_dict("records"))
            fitted_meta.append(
                {
                    "seed": seed,
                    "method": DISPLAY[name],
                    "threshold": float(model.threshold),
                    "features": list(features),
                }
            )

    by_seed = pd.DataFrame(rows)
    pred = pd.DataFrame(preds)
    by_seed.to_csv(out / "ckd_external_metrics_by_seed.csv", index=False)
    pred.to_csv(out / "ckd_external_predictions.csv", index=False)
    pd.DataFrame(bins_rows).to_csv(
        out / "ckd_external_calibration_bins_by_seed.csv", index=False
    )
    write_json(fitted_meta, out / "ckd_locked_model_manifest.json")

    # Summary table: mean metrics across the prespecified seed-specific locked refits.
    seed_summary = summarize_seed_metrics(by_seed)
    seed_summary.to_csv(out / "TABLE_CKD_EXTERNAL_VALIDATION.csv", index=False)

    # Participant-level average predictions are used only for paired bootstrap and
    # a single descriptive calibration curve, matching the paired
    # comparison strategy across repeated fits.
    agg = pred.groupby(["participant_id", "method"], as_index=False).agg(
        y=("y", "first"),
        probability=("probability", "mean"),
        prediction=("prediction", lambda x: int(np.mean(x) >= 0.5)),
    )
    agg.to_csv(out / "ckd_external_predictions_averaged.csv", index=False)

    final_bins = []
    averaged_metrics = []
    for method, g in agg.groupby("method", sort=False):
        m = classification_metrics(
            g.y.astype(int),
            g.probability,
            g.prediction,
            cfg["evaluation"]["ece_bins"],
        )
        averaged_metrics.append({"method": method, **m})
        cb = calibration_bins(
            g.y.astype(int), g.probability, cfg["evaluation"]["ece_bins"]
        )
        cb["method"] = method
        final_bins.extend(cb.to_dict("records"))

    pd.DataFrame(averaged_metrics).to_csv(
        out / "ckd_external_metrics_on_averaged_predictions.csv", index=False
    )
    pd.DataFrame(final_bins).to_csv(
        out / "ckd_external_calibration_bins_final.csv", index=False
    )
    return seed_summary, agg, by_seed


def auc_bootstrap_ci(agg, resamples=2000, seed=20260917):
    rng = np.random.default_rng(seed)
    rows = []
    for method, g in agg.groupby("method"):
        y = g.y.to_numpy()
        p = g.probability.to_numpy()
        obs = roc_auc_score(y, p)
        vals = []
        n = len(y)
        for _ in range(resamples):
            ii = rng.integers(0, n, n)
            if len(np.unique(y[ii])) < 2:
                continue
            vals.append(roc_auc_score(y[ii], p[ii]))
        lo, hi = np.quantile(vals, [0.025, 0.975])
        rows.append(
            {
                "method": method,
                "auc_on_averaged_predictions": obs,
                "bootstrap_ci95_low": float(lo),
                "bootstrap_ci95_high": float(hi),
                "bootstrap_resamples_used": len(vals),
            }
        )
    return pd.DataFrame(rows)


def paired_bootstrap_ebro_vs_ebm(agg, resamples=2000, seed=20260918):
    p = agg.pivot(index="participant_id", columns="method", values="probability")
    b = agg.pivot(index="participant_id", columns="method", values="prediction")
    required = {"EBRO-MH", "EBM"}
    if not required.issubset(set(p.columns)):
        raise RuntimeError(
            f"Paired bootstrap requires {sorted(required)}, found {sorted(p.columns)}"
        )
    y = (
        agg.drop_duplicates("participant_id")
        .set_index("participant_id")["y"]
        .loc[p.index]
        .to_numpy()
    )
    rng = np.random.default_rng(seed)
    n = len(y)
    rows = []

    for metric in ["auc", "f1"]:
        if metric == "auc":
            obs = roc_auc_score(y, p["EBRO-MH"]) - roc_auc_score(y, p["EBM"])
        else:
            obs = f1_score(y, b["EBRO-MH"], zero_division=0) - f1_score(
                y, b["EBM"], zero_division=0
            )

        vals = []
        for _ in range(resamples):
            ii = rng.integers(0, n, n)
            if metric == "auc":
                if len(np.unique(y[ii])) < 2:
                    continue
                v = roc_auc_score(y[ii], p["EBRO-MH"].to_numpy()[ii]) - roc_auc_score(
                    y[ii], p["EBM"].to_numpy()[ii]
                )
            else:
                v = f1_score(
                    y[ii], b["EBRO-MH"].to_numpy()[ii], zero_division=0
                ) - f1_score(y[ii], b["EBM"].to_numpy()[ii], zero_division=0)
            vals.append(v)

        vals = np.asarray(vals, dtype=float)
        lo, hi = np.quantile(vals, [0.025, 0.975])
        p_raw = 2 * min(np.mean(vals <= 0), np.mean(vals >= 0))
        # Avoid reporting impossible P=0 from a finite bootstrap Monte Carlo sample.
        p_floor = 1.0 / (len(vals) + 1.0)
        p_raw = max(float(p_raw), p_floor)
        rows.append(
            {
                "metric": metric,
                "comparator": "EBM",
                "difference_ebro_minus_ebm": float(obs),
                "ci95_low": float(lo),
                "ci95_high": float(hi),
                "p_raw": min(p_raw, 1.0),
                "bootstrap_resamples_used": len(vals),
            }
        )

    out = pd.DataFrame(rows)
    reject, padj, _, _ = multipletests(out.p_raw, method="holm")
    out["p_holm"] = padj
    out["reject_holm_0.05"] = reject
    return out
