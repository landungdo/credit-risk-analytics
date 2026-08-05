"""
Tests for the SQL risk mart.

Runs against data/sample.csv if present; skipped in a clean CI checkout without
the data file.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.risk_mart import build_database, run_query, QUERIES

SAMPLE = Path("data/sample.csv")
pytestmark = pytest.mark.skipif(
    not SAMPLE.exists(), reason="data/sample.csv not present in this checkout"
)


def test_all_queries_run():
    """Every named query executes and returns a non-empty result."""
    conn = build_database(str(SAMPLE))
    for name in QUERIES:
        df = run_query(conn, name)
        assert len(df) > 0, f"query {name} returned no rows"


def test_bad_rate_increases_with_grade():
    """Grade A should be safer than grade G (monotone risk gradient)."""
    conn = build_database(str(SAMPLE))
    df = run_query(conn, "02_bad_rate_by_grade").set_index("grade")
    assert df.loc["A", "bad_rate"] < df.loc["G", "bad_rate"]


def test_resolution_rate_drops_for_recent_vintages():
    """Recent vintages should be less resolved than older ones (censoring)."""
    conn = build_database(str(SAMPLE))
    df = run_query(conn, "04_resolution_rate_by_vintage").set_index("vintage_year")
    assert df.loc["2018", "resolved_pct"] < df.loc["2014", "resolved_pct"]
