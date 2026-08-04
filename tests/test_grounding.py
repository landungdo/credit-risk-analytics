"""
Tests for the SHAP-grounding guard in llm_explain.validate_grounding.

These run without an API key: they feed hand-written explanations to the
validator and assert it accepts grounded ones and rejects ones that introduce
factors SHAP did not surface. This is the safety-critical behavior of the
explanation layer, so it is tested directly.
"""

import sys
from pathlib import Path

# Allow `import src.llm_explain` when run from the project root
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.llm_explain import validate_grounding

DRIVERS = [
    {"feature": "sub_grade", "description": "assigned loan sub-grade", "direction": "increases_risk"},
    {"feature": "int_rate", "description": "interest rate on the loan", "direction": "increases_risk"},
    {"feature": "term", "description": "loan term", "direction": "increases_risk"},
    {"feature": "dti", "description": "debt-to-income ratio", "direction": "increases_risk"},
]


def test_grounded_reason_passes():
    reason = (
        "The application was declined due to the assigned loan sub-grade, the "
        "high interest rate, the 60-month term, and the debt-to-income ratio."
    )
    result = validate_grounding(reason, DRIVERS)
    assert result["grounded"] is True
    assert result["violations"] == []


def test_grounded_reason_varied_phrasing_passes():
    reason = (
        "Your application was not approved primarily because of the sub-grade "
        "assigned to the loan and its interest rate. The loan term and debt to "
        "income ratio also contributed."
    )
    result = validate_grounding(reason, DRIVERS)
    assert result["grounded"] is True


def test_invented_factors_are_caught():
    reason = (
        "The application was declined due to the sub-grade, interest rate, and "
        "the applicant employment history and home ownership status."
    )
    result = validate_grounding(reason, DRIVERS)
    assert result["grounded"] is False
    # Both invented factors should be flagged
    assert any("employment" in v for v in result["violations"])
    assert any("home ownership" in v for v in result["violations"])
