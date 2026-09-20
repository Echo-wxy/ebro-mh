import sys
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ebro_mh.data.synthetic import FEATURES, make_cohort
from ebro_mh.methods.rule_ensemble import EBROMH

CFG = yaml.safe_load((Path(__file__).resolve().parents[1] / "configs" / "quick.yaml").read_text())["models"]["ebro_mh"]


def _fit():
    d = make_cohort(n=600, seed=4)
    fit, val, cal = d.iloc[:400], d.iloc[400:500], d.iloc[500:]
    model = EBROMH(CFG, random_state=0).fit(
        fit[FEATURES], fit.ckd.values, val[FEATURES], val.ckd.values,
        cal[FEATURES], cal.ckd.values,
    )
    return model, d


def test_particle_count_and_rule_length():
    model, _ = _fit()
    assert len(model.rules_) <= CFG["particles"]
    assert all(len(r.conditions) <= CFG["max_conditions"] for r in model.rules_)


def test_weights_are_normalized():
    model, _ = _fit()
    assert np.isclose(float(np.sum(model.weights_)), 1.0, atol=1e-8)


def test_probabilities_are_in_range():
    model, d = _fit()
    p = model.predict_proba(d[FEATURES])[:, 1]
    assert p.min() >= 0.0 and p.max() <= 1.0
