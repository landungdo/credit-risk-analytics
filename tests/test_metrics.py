"""
Tests for discrimination metrics (AUC, KS).

Uses small synthetic score/label sets with known properties so the metrics'
correctness can be checked without training a model.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.metrics import auc_score, ks_statistic


def test_perfect_separation():
    """A score that perfectly ranks defaults above non-defaults -> AUC 1, KS 1."""
    y_true = [0, 0, 0, 1, 1, 1]
    y_score = [0.1, 0.2, 0.3, 0.7, 0.8, 0.9]
    assert auc_score(y_true, y_score) == 1.0
    assert ks_statistic(y_true, y_score) == 1.0


def test_random_scores_are_mediocre():
    """Scores unrelated to labels give AUC near 0.5 and low KS."""
    y_true = [0, 1, 0, 1, 0, 1, 0, 1]
    y_score = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
    auc = auc_score(y_true, y_score)
    assert 0.4 <= auc <= 0.6


def test_ks_is_between_zero_and_one():
    y_true = [0, 0, 1, 0, 1, 1, 0, 1]
    y_score = [0.2, 0.4, 0.6, 0.3, 0.8, 0.7, 0.1, 0.9]
    ks = ks_statistic(y_true, y_score)
    assert 0.0 <= ks <= 1.0
