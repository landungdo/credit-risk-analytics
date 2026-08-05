"""
Tests for the PSI drift metric.

Verifies the defining properties: PSI is ~0 for identical distributions and
grows as two distributions diverge.
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.psi import population_stability_index


def test_identical_distributions_give_near_zero():
    rng = np.random.default_rng(0)
    scores = rng.uniform(0, 1, 5000)
    psi = population_stability_index(scores, scores)
    assert psi < 0.01


def test_shifted_distribution_gives_higher_psi():
    rng = np.random.default_rng(0)
    baseline = rng.normal(0.3, 0.1, 5000)
    small_shift = rng.normal(0.32, 0.1, 5000)
    large_shift = rng.normal(0.55, 0.1, 5000)

    psi_small = population_stability_index(baseline, small_shift)
    psi_large = population_stability_index(baseline, large_shift)

    assert psi_large > psi_small
    assert psi_large > 0.25  # a big mean shift should register as significant


def test_psi_is_non_negative():
    rng = np.random.default_rng(1)
    a = rng.uniform(0, 1, 2000)
    b = rng.uniform(0, 1, 2000)
    assert population_stability_index(a, b) >= 0
