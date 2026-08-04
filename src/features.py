"""
Feature engineering for the credit risk model.

Categorical columns are cast to pandas 'category' dtype rather than
one-hot encoded — both XGBoost and LightGBM handle categoricals natively,
which avoids a sparse ~50-column explosion from addr_state alone.

Missing values (emp_length, dti, revol_util) are left as NaN on purpose:
tree-based models handle missing values natively and often learn a more
informative split from "missing" than from an imputed guess.
"""

import pandas as pd

CATEGORICAL_COLS = [
    "term", "grade", "sub_grade", "home_ownership",
    "verification_status", "purpose", "addr_state",
]

NUMERIC_COLS = [
    "loan_amnt", "int_rate", "installment", "annual_inc", "dti",
    "delinq_2yrs", "open_acc", "pub_rec", "revol_bal", "revol_util", "total_acc",
]

EMP_LENGTH_MAP = {
    "< 1 year": 0, "1 year": 1, "2 years": 2, "3 years": 3, "4 years": 4,
    "5 years": 5, "6 years": 6, "7 years": 7, "8 years": 8, "9 years": 9,
    "10+ years": 10,
}


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Return a model-ready feature matrix (target is handled separately)."""
    out = df.copy()

    # "36 months" -> 36 (int)
    out["term"] = out["term"].str.extract(r"(\d+)").astype(int)

    # Ordinal string -> numeric; missing stays NaN (tree models handle it)
    out["emp_length_years"] = out["emp_length"].map(EMP_LENGTH_MAP)

    # Credit history length in months at time of application —
    # a standard, high-signal feature in credit scoring
    out["earliest_cr_line"] = pd.to_datetime(out["earliest_cr_line"], format="%b-%Y")
    out["credit_history_months"] = (
        (out["issue_d"] - out["earliest_cr_line"]).dt.days / 30.44
    ).round(1)

    feature_cols = NUMERIC_COLS + ["emp_length_years", "credit_history_months"] + CATEGORICAL_COLS
    features = out[feature_cols].copy()

    for col in CATEGORICAL_COLS:
        features[col] = features[col].astype("category")

    return features


if __name__ == "__main__":
    from oot_split import load_resolved_loans

    resolved = load_resolved_loans("data/sample.csv")
    features = engineer_features(resolved)

    print("Feature matrix shape:", features.shape)
    print()
    print(features.dtypes)
    print()
    print(features.head())
    print()
    print("Missing values remaining (expected — tree models handle these natively):")
    print(features.isnull().sum()[features.isnull().sum() > 0])
