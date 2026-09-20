# Data

Nothing restricted is stored in this repository. `data/synthetic/` holds generated tables only; run `python -c "from ebro_mh.data.synthetic import make_cohort; make_cohort().to_csv('data/synthetic/cohort.csv', index=False)"` to regenerate them.

## United States survey

Download the public use files from the National Center for Health Statistics (https://wwwn.cdc.gov/nchs/nhanes/) for the 2015-2016 and 2017-March 2020 prepandemic releases and place the XPT files in `data/raw/nhanes/`. `ebro_mh/data/nhanes.py` builds the cohorts: nonpregnant adults aged 20 to 79 years; diabetes defined by self-reported physician diagnosis or use of glucose-lowering medication; chronic kidney disease defined as an estimated glomerular filtration rate below 60 mL/min/1.73 m2 by the 2021 race-free creatinine equation, or a urine albumin-to-creatinine ratio of at least 30 mg/g.

## Korean survey

Register at https://knhanes.kdca.go.kr/, accept the data use terms, and download the 2019 and 2020 SPSS archives into `data/raw/knhanes/`. **These files may not be redistributed**, which is why they are absent here and why `data/raw/` is in `.gitignore`. `ebro_mh/data/knhanes.py` builds the cohort with the same outcome definition and the same eleven predictors, harmonized to the same units. Note the coding conventions handled in that module: the diagnosis item is coded 0 for no and 1 for yes, with 8 for not applicable and 9 for unknown, while the age-at-diagnosis item uses 888 and 999 as its missing codes, so the generic missing-code mask must not be applied to it.

## Pima Indians Diabetes Database

Fetch OpenML dataset 37 (https://www.openml.org/d/37) into `data/raw/pidd/`. Zero values for glucose, blood pressure, skinfold thickness, insulin, and body mass index are treated as missing and imputed by fitting-partition medians.

## Predictors

Chronic kidney disease task, eleven predictors: age, sex, diabetes duration, body mass index, systolic and diastolic blood pressure, glycated hemoglobin, total cholesterol, high-density lipoprotein cholesterol, triglycerides, uric acid. Serum creatinine and urine albumin are excluded because they define the outcome.

## Recorded results

`data/results/` holds the recorded outputs of every run behind the reported numbers, so that any value can be traced back to a run without rerunning anything. Large tables are gzipped; `pandas.read_csv` reads them directly.

- `data/results/main_analyses/` — repeated cross-validation on the three tasks, temporal validation, component and initialization analyses, effective sample size threshold analysis, fixed-split sensitivity analyses, and the particle traces. `summary/` holds the aggregated metrics, paired comparisons, calibration bins, and subgroup metrics.
- `data/results/external_validation/` — the geographic external validation: per-participant predictions for every seed and method, per-seed metrics, calibration bins, the paired bootstrap output, the cohort tables, and the run manifests.
- `calibration_bins.csv` and `sensitivity_auc.csv` in `data/results/` are the two tables the figure scripts read.

Participant identifiers are the survey sequence numbers of the public use files. No name, address, date of birth, or any other direct identifier is present, and nothing here can be linked to a person beyond what the public files already allow.
