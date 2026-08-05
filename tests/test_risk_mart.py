"""
Tests for the SQL risk mart. Uses the synthetic_sample fixture so they run in
CI without the real dataset.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.risk_mart import build_database, run_query, QUERIES


def test_all_queries_run(synthetic_sample):
    """Every named query executes and returns a non-empty result."""
    conn = build_database(synthetic_sample)
    for name in QUERIES:
        df = run_query(conn, name)
        assert len(df) > 0, f"query {name} returned no rows"


def test_bad_rate_increases_with_grade(synthetic_sample):
    """Grade A should be safer than grade G (monotone risk gradient)."""
    conn = build_database(synthetic_sample)
    df = run_query(conn, "02_bad_rate_by_grade").set_index("grade")
    assert df.loc["A", "bad_rate"] < df.loc["G", "bad_rate"]


def test_resolution_rate_drops_for_recent_vintages(synthetic_sample):
    """Recent vintages should be less resolved than older ones (censoring)."""
    conn = build_database(synthetic_sample)
    df = run_query(conn, "04_resolution_rate_by_vintage").set_index("vintage_year")
    assert df.loc["2018", "resolved_pct"] < df.loc["2014", "resolved_pct"]


def test_loan_status_mapping_consistent(synthetic_sample):
    """
    The SQL bad-rate mapping must match the Python target mapping: only
    'Charged Off' counts as bad among resolved loans, matching src.oot_split.
    """
    from src.oot_split import BAD_STATUS, GOOD_STATUS
    conn = build_database(synthetic_sample)
    df = run_query(conn, "02_bad_rate_by_grade")
    # n_resolved should equal count of Charged Off + Fully Paid only
    import pandas as pd
    raw = pd.read_csv(synthetic_sample)
    resolved = raw[raw["loan_status"].isin(BAD_STATUS + GOOD_STATUS)]
    assert df["n_resolved"].sum() == len(resolved)
