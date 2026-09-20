# ebro-mh

A sequential Monte Carlo rule ensemble for binary clinical risk prediction, with the comparator models, evaluation protocol, and figure code used alongside it.

## Overview

Each particle carries one conjunctive threshold rule of at most three conditions, a binary rule label, and a normalized weight. Candidate rules are generated at empirical quantiles of each predictor and as two-condition conjunctions on distinct predictors, filtered by training support and by a Laplace-smoothed positive rate above the fitting-set prevalence, and sampled according to an empirical prior score that penalizes rule length and logical infeasibility and rewards mutual information with the outcome. Weights are then updated in log space over sequential validation batches using each rule's mean binary cross-entropy. Resampling and rule modification are conditional: they execute only when the effective sample size falls below a configured fraction of the particle count. Predictions are the weight-sum of firing rules, mapped to probabilities by Platt scaling fitted on a held-out calibration partition, with the classification threshold selected on the validation partition by the Youden index and then locked.

| Component | Module |
|---|---|
| Rule representation, initialization, scoring | `ebro_mh/methods/rule_ensemble.py` |
| Sequential weighting, effective sample size, conditional resampling | `ebro_mh/methods/rule_ensemble.py` |
| Platt scaling and threshold locking | `ebro_mh/calibration.py` |
| Comparator models under fixed configurations | `ebro_mh/baselines/comparators.py` |
| Discrimination, calibration, and bin-level metrics | `ebro_mh/metrics.py` |
| Survey cohort builders | `ebro_mh/data/nhanes.py`, `ebro_mh/data/knhanes.py` |
| Synthetic cohort generator | `ebro_mh/data/synthetic.py` |
| Figure style shared by all plots | `ebro_mh/figstyle.py` |

## Data

No restricted file is redistributed here. The synthetic generator produces tables with the same columns, units, and outcome definition, so the demo and the tests run without any download.

| Dataset | Source | License / access | Where it goes |
|---|---|---|---|
| National Health and Nutrition Examination Survey, United States | National Center for Health Statistics, https://wwwn.cdc.gov/nchs/nhanes/ | Public use files, free download | `data/raw/nhanes/` |
| Korea National Health and Nutrition Examination Survey | Korea Disease Control and Prevention Agency, https://knhanes.kdca.go.kr/ | Public use files, registration and acceptance of the data use terms required; **files may not be redistributed** | `data/raw/knhanes/` |
| Pima Indians Diabetes Database | OpenML dataset 37, https://www.openml.org/d/37 | Public benchmark | `data/raw/pidd/` |

See `data/README.md` for the exact file names and the variable derivations.

## Repository layout

```
ebro_mh/          package: method, baselines, calibration, metrics, cohort builders, figure style
configs/          fixed model configurations (default.yaml, quick.yaml)
scripts/          demo, external-validation run, figure scripts
data/             synthetic demo data, recorded result tables used by the figure scripts, and data documentation
tests/            unit tests and an end-to-end pipeline test
outputs/          written by the scripts; empty in the repository
```

## Usage

Quick demo, synthetic data, under a minute:

```bash
pip install -e .
python scripts/run_demo.py
```

It writes `outputs/demo_metrics.csv` and `outputs/demo_calibration.png`.

Full external-validation run, which requires the survey files described above:

```bash
python scripts/run_external_validation.py --config configs/default.yaml
```

Figures:

```bash
python scripts/plot_calibration.py
python scripts/plot_sensitivity.py
```

`configs/default.yaml` holds the fixed configuration used for all reported runs; `configs/quick.yaml` reduces the comparator settings so the demo and the tests finish quickly. Neither performs a data-dependent hyperparameter search.

## Dependencies

Python 3.9 or newer, CPU only. `pip install -e .` installs numpy, pandas, scikit-learn, xgboost, statsmodels, pyyaml, and matplotlib. The `rulefit`, `bayesian_rule_lists`, and `ebm` comparators additionally require `imodels` and `interpret`; they are optional extras and are not needed for the demo or the tests.

## License

MIT, see `LICENSE`.

## Citation

See `CITATION.cff`.
