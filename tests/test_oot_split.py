"""
Tests for the out-of-time split and target definition.

These run against the sample data if it is present; otherwise they are
skipped, so the suite still passes in a clean CI checkout without the data.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.oot_split import (
    load_resolved_loans,
    out_of_time_split,
    CENSORED_CUTOFF,
    TRAIN_END,
)

SAMPLE = Path("data/sample.csv")
pytestmark = pytest.mark.skipif(
    not SAMPLE.exists(), reason="data/sample.csv not present in this checkout"
)


def test_target_is_binary():
    df = load_resolved_loans(str(SAMPLE))
    assert set(df["target"].unique()).issubset({0, 1})


def test_censored_vintages_are_excluded():
    """No loan issued on/after the censoring cutoff should survive."""
    df = load_resolved_loans(str(SAMPLE))
    assert (df["issue_d"] < CENSORED_CUTOFF).all()


def test_splits_are_chronological_and_disjoint():
    """Train must be strictly before validation, which is before test."""
    df = load_resolved_loans(str(SAMPLE))
    train, val, test = out_of_time_split(df)

    assert train["issue_d"].max() < val["issue_d"].min()
    assert val["issue_d"].max() < test["issue_d"].min()


def test_train_is_before_cutoff():
    df = load_resolved_loans(str(SAMPLE))
    train, _, _ = out_of_time_split(df)
    assert (train["issue_d"] < TRAIN_END).all()
