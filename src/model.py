"""
Baseline probability-of-default model: XGBoost on the out-of-time split.

Trains on pre-2015 loans, tunes/monitors on the 2015 validation vintage,
and reports out-of-time performance on the held-out 2016 test vintage.
Categorical features are passed natively (enable_categorical=True) rather
than one-hot encoded.
"""

import xgboost as xgb

try:
    from src.oot_split import load_resolved_loans, out_of_time_split
    from src.features import engineer_features
    from src.metrics import evaluate
except ModuleNotFoundError:
    from oot_split import load_resolved_loans, out_of_time_split
    from features import engineer_features
    from metrics import evaluate


def train_model(X_train, y_train, X_val, y_val) -> xgb.XGBClassifier:
    """Train an XGBoost classifier with native categorical support."""
    model = xgb.XGBClassifier(
        n_estimators=400,
        learning_rate=0.03,
        max_depth=4,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=5,
        reg_lambda=1.0,
        enable_categorical=True,     # use pandas 'category' dtype directly
        eval_metric="auc",
        early_stopping_rounds=30,    # stop when validation AUC stops improving
        random_state=42,
    )
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    return model


def build_split(sample_path: str = "data/sample.csv"):
    """Load, label, split, and engineer features for all three sets."""
    resolved = load_resolved_loans(sample_path)
    train_df, val_df, test_df = out_of_time_split(resolved)

    X_train = engineer_features(train_df)
    X_val = engineer_features(val_df)
    X_test = engineer_features(test_df)

    y_train = train_df["target"].values
    y_val = val_df["target"].values
    y_test = test_df["target"].values

    return (X_train, y_train), (X_val, y_val), (X_test, y_test)


if __name__ == "__main__":
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = build_split()

    print(f"Train: {len(X_train)} | Validation: {len(X_val)} | Test: {len(X_test)}")
    print(f"Best iteration will be chosen on the 2015 validation vintage.\n")

    model = train_model(X_train, y_train, X_val, y_val)
    print(f"Best iteration: {model.best_iteration}\n")

    # In-sample and out-of-time discrimination
    evaluate(y_train, model.predict_proba(X_train)[:, 1], "Train")
    evaluate(y_val, model.predict_proba(X_val)[:, 1], "Validation 2015")
    evaluate(y_test, model.predict_proba(X_test)[:, 1], "Test 2016 (OOT)")
