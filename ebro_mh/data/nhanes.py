from __future__ import annotations
from pathlib import Path
import requests
import numpy as np
import pandas as pd
from ..common import ensure_dir, numeric

CYCLES = {
    "2015-2016": {
        "base": "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2015/DataFiles",
        "files": ["DEMO_I","DIQ_I","BMX_I","BPX_I","BIOPRO_I","HDL_I","GHB_I","ALB_CR_I"],
        "bp": "ausc",
    },
    "2017-2020": {
        "base": "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2017/DataFiles",
        "files": ["P_DEMO","P_DIQ","P_BMX","P_BPXO","P_BIOPRO","P_HDL","P_GHB","P_ALB_CR"],
        "bp": "osc",
    },
}

def download_cycle(cycle: str, raw_root: Path):
    spec = CYCLES[cycle]
    out = ensure_dir(Path(raw_root) / cycle)
    s = requests.Session()
    s.headers.update({"User-Agent": "Mozilla/5.0 EBRO-MH external validation"})
    for stem in spec["files"]:
        p = out / f"{stem}.xpt"
        if p.exists() and p.stat().st_size > 1024:
            continue
        url = f"{spec['base']}/{stem}.xpt"
        print("downloading", url, flush=True)
        r = s.get(url, timeout=120)
        r.raise_for_status()
        p.write_bytes(r.content)
    return out

def _read_xpt(path):
    return pd.read_sas(path, format="xport", encoding="latin-1")

def _merge(paths):
    frames = []
    for p in paths:
        x = _read_xpt(p)
        x.columns = [str(c).upper() for c in x.columns]
        frames.append(x)
    d = frames[0]
    for x in frames[1:]:
        keep = [c for c in x.columns if c == "SEQN" or c not in d.columns]
        d = d.merge(x[keep], on="SEQN", how="left", validate="one_to_one")
    return d

def _rowmean(d, cols):
    xs = [numeric(d[c]) if c in d else pd.Series(np.nan, index=d.index) for c in cols]
    return pd.concat(xs, axis=1).mean(axis=1, skipna=True)

def diabetes_outcome(d):
    """
    Locked secondary external-validation protocol.
    Eligible: valid DIQ010 response (1=yes, 2=no, 3=borderline).
    Positive: DIQ010=1 OR current insulin use OR current oral diabetes medication.
    This is used only to define the diabetic CKD development population.
    """
    diag = numeric(d.get("DIQ010"))
    insulin = numeric(d.get("DIQ050"))
    pills = numeric(d.get("DIQ070"))
    eligible = diag.isin([1,2,3])
    positive = (diag == 1) | (insulin == 1) | (pills == 1)
    y = pd.Series(np.nan, index=d.index, dtype=float)
    y.loc[eligible] = 0.0
    y.loc[eligible & positive] = 1.0
    return y

def ckd_epi_2021(scr_mg_dl, age, sex):
    scr = np.asarray(scr_mg_dl, dtype=float)
    age = np.asarray(age, dtype=float)
    sex = np.asarray(sex, dtype=float)
    female = sex == 2
    k = np.where(female, 0.7, 0.9)
    alpha = np.where(female, -0.241, -0.302)
    ratio = scr / k
    egfr = (
        142
        * np.minimum(ratio, 1) ** alpha
        * np.maximum(ratio, 1) ** (-1.200)
        * (0.9938 ** age)
        * np.where(female, 1.012, 1.0)
    )
    egfr[~np.isfinite(scr) | ~np.isfinite(age) | ~np.isfinite(sex)] = np.nan
    return egfr

def build_cycle(cycle: str, raw_root: Path, download=True):
    if download:
        download_cycle(cycle, raw_root)
    spec = CYCLES[cycle]
    folder = Path(raw_root) / cycle
    paths = [folder / f"{stem}.xpt" for stem in spec["files"]]
    missing = [str(p) for p in paths if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing NHANES files: " + ", ".join(missing))

    d = _merge(paths)
    out = pd.DataFrame({"participant_id": d["SEQN"].astype("Int64").astype(str)})
    out["cycle"] = cycle
    out["age"] = numeric(d.get("RIDAGEYR"))
    out["sex"] = numeric(d.get("RIAGENDR"))
    out["pregnant"] = numeric(d.get("RIDEXPRG"))
    out["bmi"] = numeric(d.get("BMXBMI"))

    if spec["bp"] == "ausc":
        out["sbp"] = _rowmean(d, ["BPXSY2","BPXSY3"])
        out["dbp"] = _rowmean(d, ["BPXDI2","BPXDI3"])
    else:
        out["sbp"] = _rowmean(d, ["BPXOSY2","BPXOSY3"])
        out["dbp"] = _rowmean(d, ["BPXODI2","BPXODI3"])

    out["hba1c"] = numeric(d.get("LBXGH"))
    out["total_cholesterol"] = numeric(d.get("LBXSCH"))
    out["hdl"] = numeric(d.get("LBDHDD"))
    out["triglycerides"] = numeric(d.get("LBXSTR"))
    out["uric_acid"] = numeric(d.get("LBXSUA"))
    out["serum_creatinine"] = numeric(d.get("LBXSCR"))
    out["uacr"] = numeric(d.get("URDACT"))
    out["diabetes"] = diabetes_outcome(d)

    dxage = numeric(d.get("DID040"))
    dxage = dxage.mask(dxage.isin([666,777,999]))
    out["diabetes_duration"] = out["age"] - dxage
    out.loc[out["diabetes_duration"] < 0, "diabetes_duration"] = np.nan

    out["egfr"] = ckd_epi_2021(out["serum_creatinine"], out["age"], out["sex"])
    out["ckd"] = np.where(
        out["egfr"].notna() & out["uacr"].notna(),
        ((out["egfr"] < 60) | (out["uacr"] >= 30)).astype(float),
        np.nan,
    )
    return out

def build_ckd_development(raw_root: Path, cfg, download=True):
    frames = []
    for cycle in ["2015-2016", "2017-2020"]:
        d = build_cycle(cycle, raw_root, download=download)
        d = d[(d.age >= cfg["cohort"]["min_age"]) & (d.age <= cfg["cohort"]["max_age"])].copy()
        if cfg["cohort"]["exclude_pregnant"]:
            d = d[~(d.pregnant == 1)].copy()
        frames.append(d)

    x = pd.concat(frames, ignore_index=True)
    # Secondary external validation is explicitly restricted to adults with diabetes
    # because diabetes duration is part of the existing 11-predictor CKD specification.
    x = x[(x.diabetes == 1) & x.ckd.notna()].copy()
    return x
