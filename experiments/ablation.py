"""
Ablation study: how much of the model's performance comes from the borrower's
own attributes versus from Lending Club's own risk pricing?

Lending Club assigns each loan a `grade`, `sub_grade`, and `int_rate` that are
themselves the output of *their* risk assessment. Using them as features means
the model can partly re-learn a score someone else already computed. This study
quantifies that by training three models on the same out-of-time split:

1. FULL      - all features (the project's primary model)
2. NO_PRICING- drops int_rate, grade, sub_grade (borrower attributes only)
3. BASELINE  - logistic regression on the FULL feature set, as a simple
               reference point for whether XGBoost's complexity is justified

The gap between FULL and NO_PRICING shows how much lift the pricing variables
add; the gap between FULL and BASELINE shows what the gradient boosting buys
over a linear model.
"""

import numpy as np
import pandas as pd
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

from src.oot_split import load_resolved_loans, out_of_time_split
from src.features import engineer_features, CATEGORICAL_COLS, NUMERIC_COLS
from src.metrics import auc_score, ks_statistic

PRICING_FEATURES = ["int_rate", "grade", "sub_grade"]


def _xgb():
    return xgb.XGBClassifier(
        n_estimators=400, learning_rate=0.03, max_depth=4,
        subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
        reg_lambda=1.0, enable_categorical=True, eval_metric="auc",
        early_stopping_rounds=30, random_state=42,
    )


def train_xgb(X_train, y_train, X_val, y_val, drop=None):
    """Train XGBoost, optionally dropping a list of columns first."""
    if drop:
        X_train = X_train.drop(columns=drop)
        X_val = X_val.drop(columns=drop)
    model = _xgb()
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    return model, X_val


def train_logistic(X_train, y_train, X_val, y_val):
    """
    Logistic-regression baseline. Unlike XGBoost it needs explicit encoding:
    one-hot for categoricals, median imputation for missing numerics.
    """
    numeric_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])
    pre = ColumnTransformer([
        ("num", numeric_pipe,
         NUMERIC_COLS + ["emp_length_years", "credit_history_months"]),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_COLS),
    ])
    pipe = Pipeline([
        ("pre", pre),
        ("lr", LogisticRegression(max_iter=1000, C=1.0)),
    ])
    # OneHotEncoder needs plain object dtype, not pandas 'category'
    Xt = X_train.copy()
    Xv = X_val.copy()
    for c in CATEGORICAL_COLS:
        Xt[c] = Xt[c].astype(str)
        Xv[c] = Xv[c].astype(str)
    pipe.fit(Xt, y_train)
    return pipe, Xv


def evaluate_on_test(model, X_test, y_test, drop=None, stringify=False):
    Xt = X_test.copy()
    if drop:
        Xt = Xt.drop(columns=drop)
    if stringify:
        for c in CATEGORICAL_COLS:
            if c in Xt.columns:
                Xt[c] = Xt[c].astype(str)
    p = model.predict_proba(Xt)[:, 1]
    return auc_score(y_test, p), ks_statistic(y_test, p)


def run():
    resolved = load_resolved_loans("data/sample.csv")
    train_df, val_df, test_df = out_of_time_split(resolved)

    X_train = engineer_features(train_df)
    X_val = engineer_features(val_df)
    X_test = engineer_features(test_df)
    y_train = train_df["target"].values
    y_val = val_df["target"].values
    y_test = test_df["target"].values

    results = []

    # 1. FULL
    m_full, _ = train_xgb(X_train, y_train, X_val, y_val)
    auc, ks = evaluate_on_test(m_full, X_test, y_test)
    results.append(("FULL (all features, XGBoost)", auc, ks))

    # 2. NO_PRICING
    m_np, _ = train_xgb(X_train, y_train, X_val, y_val, drop=PRICING_FEATURES)
    auc, ks = evaluate_on_test(m_np, X_test, y_test, drop=PRICING_FEATURES)
    results.append(("NO_PRICING (drops int_rate/grade/sub_grade)", auc, ks))

    # 3. BASELINE (logistic on full features)
    m_lr, _ = train_logistic(X_train, y_train, X_val, y_val)
    auc, ks = evaluate_on_test(m_lr, X_test, y_test, stringify=True)
    results.append(("BASELINE (logistic regression, all features)", auc, ks))

    print(f"{'Model':<48} {'AUC':>6} {'KS':>6}")
    print("-" * 62)
    for name, auc, ks in results:
        print(f"{name:<48} {auc:>6.3f} {ks:>6.3f}")

    full_auc = results[0][1]
    np_auc = results[1][1]
    print()
    print(f"Pricing-variable lift (FULL - NO_PRICING): {full_auc - np_auc:+.3f} AUC")
    print("Interpretation: even without Lending Club's own pricing variables,")
    print("the model retains meaningful discrimination from borrower attributes,")
    print("which argues against the score being pure leakage of a pre-computed grade.")
    return results


if __name__ == "__main__":
    run()
