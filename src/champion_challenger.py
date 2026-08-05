"""
Champion / challenger model comparison.

Choosing a production model is not just "highest AUC wins". A risk team weighs
several dimensions under model governance. This module compares the XGBoost
CHAMPION (the model that makes the decision *and* is explained via SHAP, so the
reason codes describe the actual decision) against a logistic-regression
CHALLENGER (the simple, interpretable scorecard-style baseline a bank would
default to) across four dimensions:

  1. Discrimination   - AUC, KS on the out-of-time test
  2. Calibration      - Brier score on calibrated probabilities (lower is better)
  3. Latency          - median single-prediction time (ms)
  4. Interpretability - a qualitative note (SHAP attributions vs. coefficients)

Design note on consistency: the champion is the model that is both deployed for
the decision and explained. Earlier the logistic model was framed as champion
while SHAP explained XGBoost — that split meant the reason codes would not
describe the deployed decision. XGBoost is therefore the champion here, so the
SHAP-based adverse-action reasons explain the same model that decides. The
logistic challenger is kept as an interpretable benchmark: because it is within
~0.01 AUC, it documents that the extra complexity is a deliberate, bounded choice
made to support per-decision explanations, not an unexamined default.
"""

import time

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import brier_score_loss

try:
    from src.model import build_split, train_model
    from src.metrics import auc_score, ks_statistic
    from src.features import CATEGORICAL_COLS, NUMERIC_COLS
except ModuleNotFoundError:
    from model import build_split, train_model
    from metrics import auc_score, ks_statistic
    from features import CATEGORICAL_COLS, NUMERIC_COLS


def _train_logistic(X_train, y_train):
    numeric = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])
    pre = ColumnTransformer([
        ("num", numeric, NUMERIC_COLS + ["emp_length_years", "credit_history_months"]),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_COLS),
    ])
    pipe = Pipeline([("pre", pre), ("lr", LogisticRegression(max_iter=1000))])
    Xt = X_train.copy()
    for c in CATEGORICAL_COLS:
        Xt[c] = Xt[c].astype(str)
    pipe.fit(Xt, y_train)
    return pipe


def _measure_latency(predict_fn, X_row, n: int = 100) -> float:
    """Median single-prediction latency in milliseconds."""
    times = []
    for _ in range(n):
        start = time.perf_counter()
        predict_fn(X_row)
        times.append((time.perf_counter() - start) * 1000)
    return float(np.median(times))


def compare(sample_path: str = "data/sample.csv") -> pd.DataFrame:
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.frozen import FrozenEstimator

    (X_train, y_train), (X_val, y_val), (X_test, y_test) = build_split(sample_path)

    # Champion: XGBoost, calibrated on validation
    xgb_model = train_model(X_train, y_train, X_val, y_val)
    xgb_cal = CalibratedClassifierCV(FrozenEstimator(xgb_model), method="isotonic")
    xgb_cal.fit(X_val, y_val)
    xgb_p = xgb_cal.predict_proba(X_test)[:, 1]

    # Challenger: logistic regression (needs string categoricals), also calibrated
    lr_model = _train_logistic(X_train, y_train)
    X_val_str = X_val.copy()
    X_test_str = X_test.copy()
    for c in CATEGORICAL_COLS:
        X_val_str[c] = X_val_str[c].astype(str)
        X_test_str[c] = X_test_str[c].astype(str)
    lr_cal = CalibratedClassifierCV(FrozenEstimator(lr_model), method="isotonic")
    lr_cal.fit(X_val_str, y_val)
    lr_p = lr_cal.predict_proba(X_test_str)[:, 1]

    # Brier is now computed on calibrated probabilities for both, a fair comparison
    rows = [
        {
            "model": "CHALLENGER (Logistic)",
            "auc": auc_score(y_test, lr_p),
            "ks": ks_statistic(y_test, lr_p),
            "brier": brier_score_loss(y_test, lr_p),
            "latency_ms": _measure_latency(
                lambda r: lr_cal.predict_proba(r), X_test_str.iloc[[0]]
            ),
            "interpretability": "High (linear coefficients, odds ratios)",
        },
        {
            "model": "CHAMPION (XGBoost)",
            "auc": auc_score(y_test, xgb_p),
            "ks": ks_statistic(y_test, xgb_p),
            "brier": brier_score_loss(y_test, xgb_p),
            "latency_ms": _measure_latency(
                lambda r: xgb_cal.predict_proba(r), X_test.iloc[[0]]
            ),
            "interpretability": "Medium (SHAP attribution explains the deployed decision)",
        },
    ]
    return pd.DataFrame(rows)


if __name__ == "__main__":
    result = compare()

    print("=== Champion / Challenger comparison (out-of-time test, 2016) ===\n")
    for _, r in result.iterrows():
        print(f"{r['model']}")
        print(f"  AUC:              {r['auc']:.4f}")
        print(f"  KS:               {r['ks']:.4f}")
        print(f"  Brier:            {r['brier']:.4f}")
        print(f"  Latency (median): {r['latency_ms']:.2f} ms")
        print(f"  Interpretability: {r['interpretability']}")
        print()

    # result row 0 is the logistic challenger, row 1 the XGBoost champion
    logistic = result[result["model"].str.contains("Logistic")].iloc[0]
    xgb = result[result["model"].str.contains("XGBoost")].iloc[0]
    auc_gain = xgb["auc"] - logistic["auc"]
    print(f"Champion (XGBoost) AUC edge over challenger (logistic): {auc_gain:+.4f}")
    print()
    print("Recommendation: XGBoost is the champion - it makes the decision and is")
    print("explained via SHAP, so the reason codes describe the deployed model. Its")
    print("edge over the logistic challenger is small (~0.01 AUC), which is stated")
    print("openly: the complexity is justified by the need for per-decision SHAP")
    print("attribution, not by a large accuracy gain. The logistic challenger stays")
    print("as an interpretable benchmark and a fallback.")
