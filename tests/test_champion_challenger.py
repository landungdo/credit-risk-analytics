"""
Tests for the champion/challenger comparison.

Runs against data/sample.csv if present; skipped in a clean CI checkout.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.champion_challenger import compare

SAMPLE = Path("data/sample.csv")
pytestmark = pytest.mark.skipif(
    not SAMPLE.exists(), reason="data/sample.csv not present in this checkout"
)


def test_comparison_has_both_models():
    result = compare(str(SAMPLE))
    assert len(result) == 2
    assert "CHAMPION (Logistic)" in result["model"].values
    assert "CHALLENGER (XGBoost)" in result["model"].values


def test_metrics_are_in_valid_range():
    result = compare(str(SAMPLE))
    assert (result["auc"].between(0.5, 1.0)).all()
    assert (result["ks"].between(0.0, 1.0)).all()
    assert (result["brier"].between(0.0, 1.0)).all()
    assert (result["latency_ms"] > 0).all()
