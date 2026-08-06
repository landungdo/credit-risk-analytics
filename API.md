# API

FastAPI service exposing the credit risk model.

## Run locally

```bash
# 1. Train and persist the model artifacts
python scripts/train_and_save.py

# 2. Start the API
uvicorn api:app --reload
```

Interactive docs are auto-generated at `http://localhost:8000/docs`.

## Run with Docker

```bash
docker build -t credit-risk-api .
docker run -p 8000:8000 credit-risk-api
```

## Endpoints

### `GET /health`
Liveness check.
```json
{"status": "ok", "model_loaded": true}
```

### `POST /predict`
Calibrated probability of default for one application.

Request body: an `Application` object (see fields below). Response:
```json
{"probability_of_default": 0.1721}
```

### `POST /explain`
PD plus the top SHAP drivers behind it.
```json
{
  "probability_of_default": 0.1721,
  "drivers": [
    {"feature": "sub_grade", "description": "assigned loan sub-grade",
     "value": "C4", "shap": -0.18, "direction": "decreases_risk"},
    {"feature": "term", "description": "loan term",
     "value": 60, "shap": 0.09, "direction": "increases_risk"}
  ]
}
```

### `POST /decision`
The full credit decision for one application: PD, an approve / manual-review /
decline decision from the frozen policy (cutoffs selected on the 2015 validation
book), the model and policy versions, and reason codes derived from the same
XGBoost champion that produced the PD — so the explanation describes the decision
actually made.
```json
{
  "probability_of_default": 0.172,
  "decision": "APPROVE",
  "approve_below": 0.20,
  "review_below": 0.25,
  "reason_codes": ["term", "int_rate", "dti"],
  "model_version": "xgb-champion-v2.1",
  "policy_version": "policy-v2.1"
}
```
The policy is loaded once at startup from `artifacts/policy.json`. Invalid input
returns HTTP 422.

### `POST /portfolio/summary`
Expected loss and Basel-style capital for a batch of applications.

Request body: `{"applications": [ <Application>, ... ]}`. Response:
```json
{
  "n_loans": 3,
  "total_exposure": 45000,
  "expected_loss": 3484,
  "capital_requirement": 8812,
  "avg_pd": 0.172
}
```

## Application fields

| Field | Type | Notes |
|---|---|---|
| loan_amnt | float | requested amount |
| int_rate | float | interest rate (%) |
| installment | float | monthly installment |
| annual_inc | float | annual income |
| dti | float? | debt-to-income ratio |
| delinq_2yrs | float | delinquencies in past 2 years |
| open_acc | float | open credit lines |
| pub_rec | float | derogatory public records |
| revol_bal | float | revolving balance |
| revol_util | float? | revolving utilization (%) |
| total_acc | float | total credit lines |
| emp_length_years | float? | 0-10 |
| credit_history_months | float? | credit history length |
| term | int | 36 or 60 |
| grade | str | A-G |
| sub_grade | str | e.g. C4 |
| home_ownership | str | RENT / MORTGAGE / OWN / ... |
| verification_status | str | Verified / Not Verified / ... |
| purpose | str | e.g. debt_consolidation |
| addr_state | str | 2-letter state code |

Fields marked `?` are optional (the model handles missing values natively).
