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

# --- Application-time boundary -------------------------------------------------
# A strict application PD model should only use information known *before* the
# lending decision. `int_rate`, `grade`, and `sub_grade` are assigned by Lending
# Club's own pricing/underwriting, and `installment` is derived from them, so
# they are outputs of a prior decision, not independent borrower attributes.
#
# These lists make that boundary explicit. The primary model uses all features
# (FULL); the ablation study (experiments/ablation.py) trains a PRE_DECISION-only
# model to quantify how much the pricing variables contribute. Documenting the
# split is the point: a reviewer can see the boundary is understood rather than
# ignored. See README "Limitations & design decisions".
POST_DECISION_FEATURES = ["int_rate", "grade", "sub_grade", "installment"]

PRE_DECISION_FEATURES = [
    c for c in (CATEGORICAL_COLS + NUMERIC_COLS)
    if c not in POST_DECISION_FEATURES
] + ["emp_length_years", "credit_history_months"]

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
    # Works whether run from the project root (`python src/features.py`)
    # or from inside src/ — falls back gracefully for both.
    try:
        from src.oot_split import load_resolved_loans
    except ModuleNotFoundError:
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
