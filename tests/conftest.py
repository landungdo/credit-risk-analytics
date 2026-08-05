"""
Shared pytest fixtures.

The key fixture, `synthetic_sample`, generates a small Lending-Club-shaped
dataset on the fly and writes it to a temporary data/sample.csv. This lets the
data-dependent tests (risk mart, champion/challenger, out-of-time split) run in
CI without shipping the real, licensed dataset — so those tests no longer skip
on a clean checkout.

The synthetic data is deterministic (fixed seed) and constructed so that the
relationships the tests assert on actually hold: higher grades default more, and
recent vintages are less resolved (right-censoring).
"""

import numpy as np
import pandas as pd
import pytest

GRADES = ["A", "B", "C", "D", "E", "F", "G"]
# Rising default probability by grade, so grade A is safer than grade G
GRADE_DEFAULT_P = {"A": 0.05, "B": 0.10, "C": 0.18, "D": 0.28,
                   "E": 0.35, "F": 0.45, "G": 0.55}
PURPOSES = ["debt_consolidation", "credit_card", "home_improvement",
            "small_business", "medical", "major_purchase"]
STATES = ["CA", "TX", "NY", "FL", "IL", "OH", "PA", "MA", "NJ", "WA"]
HOME = ["RENT", "MORTGAGE", "OWN"]
VERIF = ["Verified", "Not Verified", "Source Verified"]


def _make_synthetic(n: int = 1200, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    # Vintages 2013-2018 with a resolution rate that falls for recent years
    years = rng.choice([2013, 2014, 2015, 2016, 2017, 2018], size=n,
                       p=[0.1, 0.2, 0.25, 0.2, 0.15, 0.1])
    resolved_prob = {2013: 1.0, 2014: 0.95, 2015: 0.90,
                     2016: 0.67, 2017: 0.40, 2018: 0.11}

    grades = rng.choice(GRADES, size=n, p=[0.20, 0.28, 0.27, 0.14, 0.07, 0.03, 0.01])
    months = rng.choice(["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                         "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"], size=n)

    rows = []
    for i in range(n):
        g = grades[i]
        yr = years[i]
        is_resolved = rng.random() < resolved_prob[yr]
        if is_resolved:
            defaulted = rng.random() < GRADE_DEFAULT_P[g]
            status = "Charged Off" if defaulted else "Fully Paid"
        else:
            status = "Current"

        rows.append({
            "loan_amnt": float(rng.integers(1000, 40000)),
            "term": rng.choice([" 36 months", " 60 months"]),
            "int_rate": round(5 + GRADES.index(g) * 3 + rng.normal(0, 1), 2),
            "installment": round(float(rng.integers(50, 1500)), 2),
            "grade": g,
            "sub_grade": f"{g}{rng.integers(1, 6)}",
            "emp_length": rng.choice(["< 1 year", "3 years", "10+ years", None]),
            "home_ownership": rng.choice(HOME),
            "annual_inc": float(rng.integers(20000, 200000)),
            "verification_status": rng.choice(VERIF),
            "issue_d": f"{months[i]}-{yr}",
            "loan_status": status,
            "purpose": rng.choice(PURPOSES),
            "addr_state": rng.choice(STATES),
            "dti": round(float(rng.uniform(0, 40)), 2),
            "delinq_2yrs": float(rng.integers(0, 3)),
            "earliest_cr_line": f"{rng.choice(months)}-{rng.integers(1990, 2010)}",
            "open_acc": float(rng.integers(2, 30)),
            "pub_rec": float(rng.integers(0, 2)),
            "revol_bal": float(rng.integers(0, 50000)),
            "revol_util": round(float(rng.uniform(0, 100)), 1),
            "total_acc": float(rng.integers(5, 50)),
        })
    return pd.DataFrame(rows)


@pytest.fixture
def synthetic_sample(tmp_path, monkeypatch):
    """
    Write a synthetic sample to a temp data/sample.csv and chdir into it, so code
    that reads the default "data/sample.csv" path works unchanged. Returns the
    path to the CSV.
    """
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    csv_path = data_dir / "sample.csv"
    _make_synthetic().to_csv(csv_path, index=False)
    monkeypatch.chdir(tmp_path)
    return str(csv_path)
