"""
Tests for the decision policy simulator.
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.policy import simulate_policy, optimal_threshold, assign_decision


def _toy_data():
    # 6 loans: 3 low-risk repaid, 3 high-risk (2 default)
    pd_scores = np.array([0.05, 0.08, 0.12, 0.30, 0.35, 0.40])
    y_true = np.array([0, 0, 0, 1, 1, 0])
    loan_amnt = np.array([1000.0] * 6)
    int_rate = np.array([12.0] * 6)
    return pd_scores, y_true, loan_amnt, int_rate


def test_higher_threshold_approves_more():
    pd_scores, y_true, amnt, rate = _toy_data()
    policy = simulate_policy(pd_scores, y_true, amnt, rate,
                             thresholds=[0.10, 0.50])
    low = policy[policy["threshold"] == 0.10]["approval_rate"].iloc[0]
    high = policy[policy["threshold"] == 0.50]["approval_rate"].iloc[0]
    assert high > low


def test_approved_default_rate_is_a_fraction():
    pd_scores, y_true, amnt, rate = _toy_data()
    policy = simulate_policy(pd_scores, y_true, amnt, rate)
    assert (policy["approved_default_rate"] >= 0).all()
    assert (policy["approved_default_rate"] <= 1).all()


def test_optimal_threshold_returns_max_profit():
    pd_scores, y_true, amnt, rate = _toy_data()
    policy = simulate_policy(pd_scores, y_true, amnt, rate)
    best = optimal_threshold(policy)
    assert best["total_profit"] == policy["total_profit"].max()


def test_decision_bands():
    assert assign_decision(0.05, 0.10, 0.30) == "APPROVE"
    assert assign_decision(0.20, 0.10, 0.30) == "MANUAL_REVIEW"
    assert assign_decision(0.50, 0.10, 0.30) == "DECLINE"
