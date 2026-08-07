"""
Shared pytest fixtures.

The key fixture, `synthetic_sample`, writes a small Lending-Club-shaped dataset
to a temporary data/sample.csv so data-dependent tests (risk mart,
champion/challenger, out-of-time split) run in CI without the real licensed
dataset. The generator itself lives in src/demo_data.py, the single source of
truth shared with the Docker demo-data script.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.demo_data import _make_synthetic

@pytest.fixture
def synthetic_sample(tmp_path, monkeypatch):
    """
    Write a synthetic sample to a temp data/sample.csv and chdir into it, so code
    that reads the default "data/sample.csv" path works unchanged. Returns the
    path to the CSV.
    """
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    csv_path = data_dir / "sample.csv"
    _make_synthetic().to_csv(csv_path, index=False)
    monkeypatch.chdir(tmp_path)
    return str(csv_path)
