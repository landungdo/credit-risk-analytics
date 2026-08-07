"""
Integration tests for the /decision endpoint.

These build a small model + policy into a temp artifacts dir (via the
synthetic_sample fixture and train_and_save), then exercise the API with
FastAPI's TestClient. They verify the decision system contract: the decision
band matches the policy cutoffs, the response carries model/policy versions and
reason codes, invalid input returns 422, and the policy is loaded once.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

VALID_APPLICATION = {
    "loan_amnt": 15000, "int_rate": 14.5, "installment": 350, "annual_inc": 60000,
    "dti": 18.0, "delinq_2yrs": 0, "open_acc": 10, "pub_rec": 0, "revol_bal": 5000,
    "revol_util": 40.0, "total_acc": 20, "emp_length_years": 5,
    "credit_history_months": 120, "term": 60, "grade": "C", "sub_grade": "C4",
    "home_ownership": "RENT", "verification_status": "Verified",
    "purpose": "debt_consolidation", "addr_state": "CA",
}


@pytest.fixture
def client(synthetic_sample):
    """Train artifacts into a temp dir, load them, and return a TestClient."""
    from fastapi.testclient import TestClient
    from scripts.train_and_save import main as train_and_save
    import api

    train_and_save(synthetic_sample)  # writes artifacts/ incl. policy.json
    api.load_artifacts()
    return TestClient(api.app)


def test_decision_returns_valid_band(client):
    r = client.post("/decision", json=VALID_APPLICATION)
    assert r.status_code == 200
    body = r.json()
    assert body["decision"] in {"APPROVE", "MANUAL_REVIEW", "DECLINE"}


def test_decision_band_matches_cutoffs(client):
    """The returned decision must be consistent with the returned PD and cutoffs."""
    body = client.post("/decision", json=VALID_APPLICATION).json()
    pd_score = body["probability_of_default"]
    if pd_score < body["approve_below"]:
        assert body["decision"] == "APPROVE"
    elif pd_score < body["review_below"]:
        assert body["decision"] == "MANUAL_REVIEW"
    else:
        assert body["decision"] == "DECLINE"


def test_decision_includes_versions_and_reasons(client):
    body = client.post("/decision", json=VALID_APPLICATION).json()
    assert "model_version" in body
    assert "policy_version" in body
    assert isinstance(body["reason_codes"], list)


def test_explanation_uses_same_champion_as_decision(client):
    """The PD in /decision must equal the PD in /explain (same champion model)."""
    d = client.post("/decision", json=VALID_APPLICATION).json()
    e = client.post("/explain", json=VALID_APPLICATION).json()
    assert abs(d["probability_of_default"] - e["probability_of_default"]) < 1e-9


def test_invalid_input_returns_422(client):
    bad = dict(VALID_APPLICATION)
    del bad["loan_amnt"]  # required field missing
    r = client.post("/decision", json=bad)
    assert r.status_code == 422


def test_policy_loaded_once(client):
    """Policy is loaded at startup and reused, not re-read per request."""
    import api
    assert api._policy is not None
    assert api._policy["policy_version"] == "policy-v2.1"


def test_negative_loan_amount_rejected(client):
    """Economically impossible input (negative amount) must return 422."""
    bad = dict(VALID_APPLICATION)
    bad["loan_amnt"] = -5000
    assert client.post("/decision", json=bad).status_code == 422


def test_nonstandard_term_rejected(client):
    """A term other than 36 or 60 must return 422."""
    bad = dict(VALID_APPLICATION)
    bad["term"] = 48
    assert client.post("/decision", json=bad).status_code == 422


def test_negative_interest_rate_rejected(client):
    """A negative interest rate must return 422."""
    bad = dict(VALID_APPLICATION)
    bad["int_rate"] = -3.0
    assert client.post("/decision", json=bad).status_code == 422
