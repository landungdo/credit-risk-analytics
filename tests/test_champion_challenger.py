"""
Tests for the champion/challenger comparison. Uses the synthetic_sample fixture
so they run in CI without the real dataset.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.champion_challenger import compare


def test_comparison_has_both_models(synthetic_sample):
    result = compare(synthetic_sample)
    assert len(result) == 2
    models = result["model"].values
    assert any("Logistic" in m for m in models)
    assert any("XGBoost" in m for m in models)


def test_xgboost_is_the_champion(synthetic_sample):
    """The champion (explained model) must be XGBoost, matching SHAP explanation."""
    result = compare(synthetic_sample)
    champion_row = result[result["model"].str.contains("CHAMPION")].iloc[0]
    assert "XGBoost" in champion_row["model"]


def test_metrics_are_in_valid_range(synthetic_sample):
    result = compare(synthetic_sample)
    assert (result["auc"].between(0.4, 1.0)).all()
    assert (result["ks"].between(0.0, 1.0)).all()
    assert (result["brier"].between(0.0, 1.0)).all()
    assert (result["latency_ms"] > 0).all()
