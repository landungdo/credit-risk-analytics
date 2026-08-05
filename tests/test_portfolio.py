"""
Tests for the portfolio risk module (expected loss, Basel capital).

Checks the loss arithmetic on hand-computable inputs and verifies the
structural properties of the Basel capital formula.
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.portfolio import expected_loss, basel_capital_requirement, portfolio_summary


def test_expected_loss_arithmetic():
    """EL = PD * LGD * EAD, computed per loan."""
    pd_scores = np.array([0.1, 0.2])
    exposure = np.array([1000.0, 2000.0])
    el = expected_loss(pd_scores, exposure, lgd=0.5)
    # 0.1*0.5*1000 = 50 ; 0.2*0.5*2000 = 200
    assert np.allclose(el, [50.0, 200.0])


def test_capital_is_positive_and_below_exposure():
    """Capital charge should be positive and never exceed the exposure."""
    pd_scores = np.array([0.05, 0.20, 0.50])
    exposure = np.array([1000.0, 1000.0, 1000.0])
    capital = basel_capital_requirement(pd_scores, exposure)
    assert np.all(capital > 0)
    assert np.all(capital < exposure)


def test_portfolio_summary_totals():
    """Summary totals should match the sum of per-loan figures."""
    pd_scores = np.array([0.1, 0.3])
    exposure = np.array([1000.0, 1000.0])
    summary = portfolio_summary(pd_scores, exposure, lgd=0.5)
    assert summary["n_loans"] == 2
    assert summary["total_exposure"] == 2000.0
    # EL = 0.1*0.5*1000 + 0.3*0.5*1000 = 50 + 150 = 200
    assert np.isclose(summary["expected_loss"], 200.0)


def test_higher_pd_means_higher_expected_loss():
    """Monotonicity: raising PD raises expected loss."""
    exposure = np.array([1000.0])
    low = expected_loss(np.array([0.1]), exposure)[0]
    high = expected_loss(np.array([0.4]), exposure)[0]
    assert high > low
