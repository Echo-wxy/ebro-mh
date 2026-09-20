from __future__ import annotations
from pathlib import Path
import zipfile
import numpy as np
import pandas as pd
from ..common import ensure_dir, numeric
from .nhanes import ckd_epi_2021

ALIASES = {
    "ID": ["ID"],
    "SEX": ["SEX"],
    "AGE": ["AGE"],
    "DE1_DG": ["DE1_DG"],
    "DE1_AG": ["DE1_AG"],
    "DE1_PT": ["DE1_PT"],
    "HE_PRG": ["HE_PRG"],
    "HE_BMI": ["HE_BMI"],
    "HE_SBP": ["HE_SBP"],
    "HE_DBP": ["HE_DBP"],
    "HE_HBA1C": ["HE_HBA1C"],
    "HE_CHOL": ["HE_CHOL"],
    "HE_HDL": ["HE_HDL_ST2","HE_HDL_S"],
    "HE_TG": ["HE_TG"],
    "HE_CREA": ["HE_CREA"],
    "HE_UACID": ["HE_UACID"],
    "HE_UCREA": ["HE_UCREA"],
    "HE_UALB": ["HE_UALB"],
}

def _resolve_columns(d):
    resolved, missing = {}, []
    for logical, candidates in ALIASES.items():
        hit = next((c for c in candidates if c in d.columns), None)
        if hit is None:
            missing.append(f"{logical} (tried {candidates})")
        else:
            resolved[logical] = hit
    if missing:
        raise KeyError("Missing KNHANES fields: " + "; ".join(missing))
    return resolved

def _find_sav(raw_root: Path, year: int):
    target = f"HN{str(year)[-2:]}_all.sav".lower()
    for p in raw_root.rglob("*.sav"):
        if p.name.lower() == target:
            return p
    zips = list(raw_root.glob(f"HN{str(year)[-2:]}_ALL(SPSS).zip"))
    if not zips:
        raise FileNotFoundError(f"No KNHANES {year} SAV or ZIP found in {raw_root}")
    out = ensure_dir(raw_root / f"extracted_{year}")
    with zipfile.ZipFile(zips[0]) as z:
        z.extractall(out)
    for p in out.rglob("*.sav"):
        if p.name.lower() == target:
            return p
    raise FileNotFoundError(f"{target} not found after extraction")

def _read_sav(path):
    import pyreadstat
    d, meta = pyreadstat.read_sav(str(path), apply_value_formats=False)
    d.columns = [str(c).upper() for c in d.columns]
    return d

def _clean_code(s):
    """Missing codes for KNHANES yes/no items: 8 = not applicable, 9 = don't know."""
    s = numeric(s)
    return s.mask(s.isin([8,9,88,99,888,999,9999]))

def _clean_age_code(s):
    """Missing codes for KNHANES age-at-diagnosis items.

    Only 888 and 999 are missing codes here. The generic mask must not be used:
    8 and 9 are valid ages at diagnosis for childhood-onset diabetes and are
    present in the 2019-2020 files.
    """
    s = numeric(s)
    return s.mask(s.isin([888, 999, 9999]))

def diabetes_outcome(d):
    dg = _clean_code(d.get("DE1_DG"))
    pt = _clean_code(d.get("DE1_PT"))
    # KNHANES 2019-2020 codes DE1_DG as 0 = no, 1 = yes (8 = not applicable,
    # 9 = don't know), not 1 = yes / 2 = no.
    eligible = dg.isin([0, 1])
    positive = (dg == 1) | (pt == 1)
    y = pd.Series(np.nan, index=d.index, dtype=float)
    y.loc[eligible] = 0.0
    y.loc[eligible & positive] = 1.0
    return y

def build_year(raw_root: Path, year: int):
    d = _read_sav(_find_sav(raw_root, year))
    c = _resolve_columns(d)
    out = pd.DataFrame(index=d.index)
    out["participant_id"] = d[c["ID"]].astype(str)
    out["cycle"] = str(year)
    out["age"] = numeric(d[c["AGE"]])
    out["sex"] = numeric(d[c["SEX"]])
    out["pregnant"] = numeric(d[c["HE_PRG"]])
    out["bmi"] = numeric(d[c["HE_BMI"]])
    out["sbp"] = numeric(d[c["HE_SBP"]])
    out["dbp"] = numeric(d[c["HE_DBP"]])
    out["hba1c"] = numeric(d[c["HE_HBA1C"]])
    out["total_cholesterol"] = numeric(d[c["HE_CHOL"]])
    out["hdl"] = numeric(d[c["HE_HDL"]])
    out["triglycerides"] = numeric(d[c["HE_TG"]])
    out["uric_acid"] = numeric(d[c["HE_UACID"]])
    out["serum_creatinine"] = numeric(d[c["HE_CREA"]])
    out["urine_creatinine"] = numeric(d[c["HE_UCREA"]])
    out["urine_albumin"] = numeric(d[c["HE_UALB"]])

    # KNHANES urine albumin is mg/L and urine creatinine is mg/dL.
    # UACR (mg/g) = urine albumin (mg/L) / urine creatinine (mg/dL) * 100.
    out["uacr"] = out["urine_albumin"] / out["urine_creatinine"] * 100.0
    out["diabetes"] = diabetes_outcome(d)

    dxage = _clean_age_code(d[c["DE1_AG"]])
    out["diabetes_duration"] = out["age"] - dxage
    out.loc[out["diabetes_duration"] < 0, "diabetes_duration"] = np.nan

    out["egfr"] = ckd_epi_2021(out["serum_creatinine"], out["age"], out["sex"])
    out["ckd"] = np.where(
        out["egfr"].notna() & out["uacr"].notna(),
        ((out["egfr"] < 60) | (out["uacr"] >= 30)).astype(float),
        np.nan,
    )
    return out

def build_ckd_external(raw_root: Path, cfg):
    x = pd.concat([build_year(raw_root, 2019), build_year(raw_root, 2020)], ignore_index=True)
    x = x[(x.age >= cfg["cohort"]["min_age"]) & (x.age <= cfg["cohort"]["max_age"])].copy()
    if cfg["cohort"]["exclude_pregnant"]:
        x = x[~(x.pregnant == 1)].copy()
    x = x[(x.diabetes == 1) & x.ckd.notna()].copy()
    return x
