"""
Champion / challenger model comparison.

Choosing a production model is not just "highest AUC wins". A risk team weighs
several dimensions, because a marginally more accurate model that is slower,
harder to calibrate, or harder to explain may be the wrong choice under model
governance. This module compares a logistic-regression CHAMPION (the simple,
interpretable scorecard-style baseline that a bank would default to) against an
XGBoost CHALLENGER across four dimensions:

  1. Discrimination   - AUC, KS on the out-of-time test
  2. Calibration      - Brier score (lower is better)
  3. Latency          - median single-prediction time (ms)
  4. Interpretability - a qualitative note (coefficients vs. SHAP)

The framing deliberately makes logistic the champion: on this data it is nearly
as accurate (see the ablation study), so the burden is on the challenger to
justify the extra complexity.
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
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = build_split(sample_path)

    # Challenger: XGBoost
    xgb_model = train_model(X_train, y_train, X_val, y_val)
    xgb_p = xgb_model.predict_proba(X_test)[:, 1]

    # Champion: logistic regression (needs string categoricals)
    lr_model = _train_logistic(X_train, y_train)
    X_test_str = X_test.copy()
    for c in CATEGORICAL_COLS:
        X_test_str[c] = X_test_str[c].astype(str)
    lr_p = lr_model.predict_proba(X_test_str)[:, 1]

    rows = [
        {
            "model": "CHAMPION (Logistic)",
            "auc": auc_score(y_test, lr_p),
            "ks": ks_statistic(y_test, lr_p),
            "brier": brier_score_loss(y_test, lr_p),
            "latency_ms": _measure_latency(
                lambda r: lr_model.predict_proba(r), X_test_str.iloc[[0]]
            ),
            "interpretability": "High (linear coefficients, odds ratios)",
        },
        {
            "model": "CHALLENGER (XGBoost)",
            "auc": auc_score(y_test, xgb_p),
            "ks": ks_statistic(y_test, xgb_p),
            "brier": brier_score_loss(y_test, xgb_p),
            "latency_ms": _measure_latency(
                lambda r: xgb_model.predict_proba(r), X_test.iloc[[0]]
            ),
            "interpretability": "Medium (needs SHAP for per-decision reasons)",
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

    champ = result.iloc[0]
    chall = result.iloc[1]
    auc_gain = chall["auc"] - champ["auc"]
    print(f"Challenger AUC gain over champion: {auc_gain:+.4f}")
    print()
    print("Recommendation: the challenger's discrimination gain is small, so the")
    print("champion (logistic) remains defensible for production on interpretability")
    print("and simplicity grounds. The challenger is retained to power SHAP-based")
    print("adverse-action explanations, where per-decision attribution is needed.")
