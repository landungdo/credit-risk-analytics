"""
FastAPI service for the credit risk model.

Endpoints:
- GET  /health                 liveness check
- POST /predict                calibrated PD for one application
- POST /explain                PD + top SHAP drivers for one application
- POST /decision               PD + approve/review/decline decision + versions
- POST /portfolio/summary      expected loss + capital for a batch

The trained artifacts are loaded once at startup (see scripts/train_and_save.py).
Input is validated with Pydantic so malformed requests get a clear 422 rather
than a 500.
"""

import pickle
import json
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.explain import build_explainer, top_drivers
from src.portfolio import portfolio_summary
from src.policy import assign_decision

ARTIFACT_DIR = Path("artifacts")

app = FastAPI(title="Credit Risk API", version="2.1")

# Loaded at startup
_model = None
_calibrated = None
_feature_columns = None
_explainer = None
_policy = None


@app.on_event("startup")
def load_artifacts():
    global _model, _calibrated, _feature_columns, _explainer, _policy
    with open(ARTIFACT_DIR / "model.pkl", "rb") as f:
        _model = pickle.load(f)
    with open(ARTIFACT_DIR / "calibrated.pkl", "rb") as f:
        _calibrated = pickle.load(f)
    with open(ARTIFACT_DIR / "feature_columns.pkl", "rb") as f:
        _feature_columns = pickle.load(f)
    with open(ARTIFACT_DIR / "policy.json") as f:
        _policy = json.load(f)
    _explainer = build_explainer(_model)


class Application(BaseModel):
    """One loan application. Fields mirror the engineered feature schema."""
    loan_amnt: float
    int_rate: float
    installment: float
    annual_inc: float
    dti: float | None = None
    delinq_2yrs: float = 0
    open_acc: float = 0
    pub_rec: float = 0
    revol_bal: float = 0
    revol_util: float | None = None
    total_acc: float = 0
    emp_length_years: float | None = None
    credit_history_months: float | None = None
    term: int = Field(..., description="36 or 60")
    grade: str
    sub_grade: str
    home_ownership: str
    verification_status: str
    purpose: str
    addr_state: str


def _to_frame(app_in: Application) -> pd.DataFrame:
    """Turn one Application into a single-row feature frame in the right order."""
    row = pd.DataFrame([app_in.model_dump()])[_feature_columns]
    categorical = [
        "term", "grade", "sub_grade", "home_ownership",
        "verification_status", "purpose", "addr_state",
    ]
    for col in categorical:
        if col in row.columns and col != "term":
            row[col] = row[col].astype("category")
    return row


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": _model is not None}


@app.post("/predict")
def predict(application: Application):
    if _calibrated is None:
        raise HTTPException(503, "Model not loaded")
    row = _to_frame(application)
    pd_score = float(_calibrated.predict_proba(row)[:, 1][0])
    return {"probability_of_default": pd_score}


@app.post("/explain")
def explain(application: Application):
    if _explainer is None:
        raise HTTPException(503, "Model not loaded")
    row = _to_frame(application)
    pd_score = float(_calibrated.predict_proba(row)[:, 1][0])
    drivers = top_drivers(_explainer, row)
    return {"probability_of_default": pd_score, "drivers": drivers}


@app.post("/decision")
def decision(application: Application):
    """
    Full credit decision for one application: PD, an approve/review/decline
    decision from the frozen policy, the model and policy versions, and reason
    codes derived from the SAME XGBoost champion that produced the PD — so the
    explanation describes the decision actually made.
    """
    if _calibrated is None or _policy is None:
        raise HTTPException(503, "Model or policy not loaded")

    row = _to_frame(application)
    pd_score = float(_calibrated.predict_proba(row)[:, 1][0])

    band = assign_decision(
        pd_score, _policy["approve_below"], _policy["review_below"]
    )

    # Reason codes: the risk-increasing SHAP drivers of THIS prediction
    drivers = top_drivers(_explainer, row)
    reason_codes = [
        d["feature"] for d in drivers if d["direction"] == "increases_risk"
    ]

    return {
        "probability_of_default": pd_score,
        "decision": band,
        "approve_below": _policy["approve_below"],
        "review_below": _policy["review_below"],
        "reason_codes": reason_codes,
        "model_version": _policy["model_version"],
        "policy_version": _policy["policy_version"],
    }


class PortfolioRequest(BaseModel):
    applications: list[Application]


@app.post("/portfolio/summary")
def portfolio(request: PortfolioRequest):
    if _calibrated is None:
        raise HTTPException(503, "Model not loaded")
    rows = pd.concat([_to_frame(a) for a in request.applications], ignore_index=True)
    for col in rows.columns:
        if rows[col].dtype.name == "object":
            rows[col] = rows[col].astype("category")
    pd_scores = _calibrated.predict_proba(rows)[:, 1]
    exposure = rows["loan_amnt"].values
    return portfolio_summary(pd_scores, exposure)
