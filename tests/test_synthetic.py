import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ebro_mh.data.synthetic import FEATURES, make_cohort, make_development_and_external


def test_columns_and_size():
    d = make_cohort(n=200, seed=1)
    assert len(d) == 200
    for c in FEATURES + ["ckd"]:
        assert c in d.columns
    assert d[FEATURES].notna().all().all()


def test_outcome_rate_is_controlled():
    d = make_cohort(n=2000, seed=2, positive_rate=0.25)
    assert abs(d.ckd.mean() - 0.25) < 0.02


def test_external_cohort_has_shifted_case_mix():
    dev, ext = make_development_and_external(seed=3)
    assert ext.ckd.mean() < dev.ckd.mean()
    assert ext.bmi.mean() < dev.bmi.mean()
