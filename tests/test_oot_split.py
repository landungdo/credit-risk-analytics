"""
Tests for the out-of-time split and target definition. Uses synthetic_sample so
they run in CI without the real dataset.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.oot_split import (
    load_resolved_loans,
    out_of_time_split,
    CENSORED_CUTOFF,
    TRAIN_END,
)


def test_target_is_binary(synthetic_sample):
    df = load_resolved_loans(synthetic_sample)
    assert set(df["target"].unique()).issubset({0, 1})


def test_censored_vintages_are_excluded(synthetic_sample):
    df = load_resolved_loans(synthetic_sample)
    assert (df["issue_d"] < CENSORED_CUTOFF).all()


def test_splits_are_chronological_and_disjoint(synthetic_sample):
    df = load_resolved_loans(synthetic_sample)
    train, val, test = out_of_time_split(df)
    assert train["issue_d"].max() < val["issue_d"].min()
    assert val["issue_d"].max() < test["issue_d"].min()


def test_train_is_before_cutoff(synthetic_sample):
    df = load_resolved_loans(synthetic_sample)
    train, _, _ = out_of_time_split(df)
    assert (train["issue_d"] < TRAIN_END).all()
