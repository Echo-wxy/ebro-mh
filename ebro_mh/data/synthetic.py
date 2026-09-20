"""Synthetic cohort generator.

Produces tables with the same columns, units, and outcome definition as the
survey cohorts, so the demo and the tests run without any restricted file.
No value in this module is taken from real data.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

FEATURES = [
    "age", "sex", "diabetes_duration", "bmi", "sbp", "dbp", "hba1c",
    "total_cholesterol", "hdl", "triglycerides", "uric_acid",
]


def make_cohort(n: int = 900, seed: int = 0, positive_rate: float = 0.30) -> pd.DataFrame:
    """Return a synthetic cohort with an outcome column named ``ckd``."""
    rng = np.random.default_rng(seed)
    d = pd.DataFrame({
        "participant_id": np.arange(n),
        "age": rng.normal(61, 11, n).clip(20, 79),
        "sex": rng.integers(1, 3, n),
        "diabetes_duration": rng.gamma(2.0, 5.0, n).clip(0, 50),
        "bmi": rng.normal(29, 6, n).clip(15, 60),
        "sbp": rng.normal(128, 17, n).clip(80, 210),
        "dbp": rng.normal(73, 11, n).clip(40, 130),
        "hba1c": rng.normal(7.3, 1.5, n).clip(4, 15),
        "total_cholesterol": rng.normal(172, 41, n).clip(80, 400),
        "hdl": rng.normal(48, 13, n).clip(15, 120),
        "triglycerides": rng.lognormal(4.9, 0.6, n).clip(30, 2000),
        "uric_acid": rng.normal(5.3, 1.5, n).clip(1, 14),
    })
    # A monotone risk score drives the outcome; coefficients are arbitrary.
    z = (0.030 * (d.age - 60) + 0.020 * (d.bmi - 29) + 0.018 * (d.sbp - 128)
         + 0.120 * (d.hba1c - 7.3) + 0.040 * (d.diabetes_duration - 10)
         + 0.090 * (d.uric_acid - 5.3))
    z = z + rng.normal(0, 1.0, n)
    cut = np.quantile(z, 1.0 - positive_rate)
    d["ckd"] = (z >= cut).astype(int)
    return d


def make_development_and_external(seed: int = 0):
    """Return a development cohort and an external cohort with a shifted case mix."""
    dev = make_cohort(n=900, seed=seed, positive_rate=0.37)
    ext = make_cohort(n=600, seed=seed + 1, positive_rate=0.28)
    ext["bmi"] = ext["bmi"] - 6.0          # the external cohort is leaner
    ext["age"] = (ext["age"] + 4.0).clip(20, 79)
    return dev, ext
