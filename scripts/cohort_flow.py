"""Participant flow for the United States survey cohorts.

Rebuilds every US cohort used in the analyses from the public NHANES files and
prints the number of participants remaining after each eligibility step, with the
number excluded at that step. The final counts are checked against the recorded
cohort sizes.

    python scripts/cohort_flow.py --raw data/raw/nhanes

Files already present under --raw are reused; missing files are downloaded from
the National Center for Health Statistics. Results are written to
outputs/cohort_flow_us.csv.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ebro_mh.data import nhanes  # noqa: E402

# The 2017-2018 cycle is used only as the temporal test cohort.
nhanes.CYCLES.setdefault("2017-2018", {
    "base": "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2017/DataFiles",
    "files": ["DEMO_J", "DIQ_J", "BMX_J", "BPX_J", "BIOPRO_J", "HDL_J", "GHB_J", "ALB_CR_J"],
    "bp": "ausc",
})

REPORTED = {
    "Diabetes cohort (2015-2016 and 2017-March 2020)": (13731, 2163),
    "CKD cohort, adults with diagnosed diabetes": (1879, 697),
    "Temporal training cohort (2015-2016)": (5271, 814),
    "Temporal test cohort (2017-2018)": (5084, 825),
}


def flow(frame: pd.DataFrame, cohort: str, steps):
    """Apply eligibility steps in order and record the count after each one."""
    rows, current = [], frame
    rows.append(dict(cohort=cohort, step="All survey participants", n=len(current), excluded=0))
    for label, keep in steps:
        nxt = current[keep(current)]
        rows.append(dict(cohort=cohort, step=label, n=len(nxt), excluded=len(current) - len(nxt)))
        current = nxt
    return rows, current


ADULT = ("Aged 20 to 79 years", lambda d: (d.age >= 20) & (d.age <= 79))
NOT_PREGNANT = ("Not pregnant", lambda d: ~(d.pregnant == 1))
DM_OUTCOME = ("Diabetes outcome available", lambda d: d.diabetes.notna())
DIAGNOSED = ("Diagnosed diabetes", lambda d: d.diabetes == 1)
CKD_OUTCOME = ("Complete CKD outcome (eGFR and UACR)", lambda d: d.ckd.notna())


def run(frames: dict) -> pd.DataFrame:
    rows = []
    main = pd.concat([frames["2015-2016"], frames["2017-2020"]], ignore_index=True)

    r, dm = flow(main, "Diabetes cohort (2015-2016 and 2017-March 2020)", [ADULT, NOT_PREGNANT, DM_OUTCOME])
    rows += r
    r, ckd = flow(main, "CKD cohort, adults with diagnosed diabetes", [ADULT, NOT_PREGNANT, DIAGNOSED, CKD_OUTCOME])
    rows += r
    r, tr = flow(frames["2015-2016"], "Temporal training cohort (2015-2016)", [ADULT, NOT_PREGNANT, DM_OUTCOME])
    rows += r
    r, te = flow(frames["2017-2018"], "Temporal test cohort (2017-2018)", [ADULT, NOT_PREGNANT, DM_OUTCOME])
    rows += r

    out = pd.DataFrame(rows)
    finals = {
        "Diabetes cohort (2015-2016 and 2017-March 2020)": (len(dm), int(dm.diabetes.sum())),
        "CKD cohort, adults with diagnosed diabetes": (len(ckd), int(ckd.ckd.sum())),
        "Temporal training cohort (2015-2016)": (len(tr), int(tr.diabetes.sum())),
        "Temporal test cohort (2017-2018)": (len(te), int(te.diabetes.sum())),
    }
    # Why participants with diagnosed diabetes lacked a CKD outcome
    dx = main[ADULT[1](main)]
    dx = dx[NOT_PREGNANT[1](dx)]
    dx = dx[DIAGNOSED[1](dx)]
    detail = dict(missing_egfr=int(dx.egfr.isna().sum()), missing_uacr=int(dx.uacr.isna().sum()),
                  missing_both=int((dx.egfr.isna() & dx.uacr.isna()).sum()))
    return out, finals, detail


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="data/raw/nhanes")
    ap.add_argument("--no-download", action="store_true")
    args = ap.parse_args()
    raw = Path(args.raw)
    frames = {c: nhanes.build_cycle(c, raw, download=not args.no_download)
              for c in ("2015-2016", "2017-2020", "2017-2018")}
    out, finals, detail = run(frames)

    Path("outputs").mkdir(exist_ok=True)
    out.to_csv("outputs/cohort_flow_us.csv", index=False)
    pd.set_option("display.width", 140)
    print(out.to_string(index=False))
    print("\nCheck against the recorded cohort sizes:")
    for k, (n, pos) in finals.items():
        rn, rpos = REPORTED[k]
        flag = "match" if (n, pos) == (rn, rpos) else "DIFFERS"
        print(f"  {k:52s} n={n:6d} positive={pos:5d}   reported {rn}/{rpos}   {flag}")
    print("\nParticipants with diagnosed diabetes but no complete CKD outcome:", detail)
    print("\nWrote outputs/cohort_flow_us.csv")


if __name__ == "__main__":
    main()
