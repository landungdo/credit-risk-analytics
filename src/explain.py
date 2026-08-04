"""
SHAP-based explanation of individual PD predictions.

For each applicant, TreeExplainer attributes the prediction to individual
features. We extract the top drivers pushing the prediction toward default
(positive SHAP) and toward repayment (negative SHAP). These structured
drivers are the *only* factual basis the LLM layer (llm_explain.py) is
allowed to use, which keeps the natural-language reason grounded.
"""

import numpy as np
import shap

# Human-readable descriptions for feature names, used when composing reasons.
FEATURE_DESCRIPTIONS = {
    "loan_amnt": "requested loan amount",
    "int_rate": "interest rate on the loan",
    "installment": "monthly installment",
    "annual_inc": "annual income",
    "dti": "debt-to-income ratio",
    "delinq_2yrs": "number of delinquencies in the past 2 years",
    "open_acc": "number of open credit lines",
    "pub_rec": "number of derogatory public records",
    "revol_bal": "revolving balance",
    "revol_util": "revolving line utilization rate",
    "total_acc": "total number of credit lines",
    "emp_length_years": "length of employment",
    "credit_history_months": "length of credit history",
    "term": "loan term",
    "grade": "assigned loan grade",
    "sub_grade": "assigned loan sub-grade",
    "home_ownership": "home ownership status",
    "verification_status": "income verification status",
    "purpose": "stated purpose of the loan",
    "addr_state": "state of residence",
}


def build_explainer(model):
    """Create a SHAP TreeExplainer for the trained model."""
    return shap.TreeExplainer(model)


def top_drivers(explainer, X_row, top_n: int = 4):
    """
    Return the top drivers for a single applicant as a list of dicts.

    Each driver: {feature, description, value, shap, direction}
    - direction "increases_risk"  -> positive SHAP (pushes toward default)
    - direction "decreases_risk"  -> negative SHAP (pushes toward repayment)
    """
    shap_values = explainer.shap_values(X_row)
    sv = shap_values[0]
    feature_names = list(X_row.columns)

    order = np.argsort(np.abs(sv))[::-1][:top_n]

    drivers = []
    for i in order:
        feat = feature_names[i]
        drivers.append({
            "feature": feat,
            "description": FEATURE_DESCRIPTIONS.get(feat, feat),
            "value": X_row.iloc[0, i],
            "shap": float(sv[i]),
            "direction": "increases_risk" if sv[i] > 0 else "decreases_risk",
        })
    return drivers


if __name__ == "__main__":
    try:
        from src.model import build_split, train_model
    except ModuleNotFoundError:
        from model import build_split, train_model

    (X_train, y_train), (X_val, y_val), (X_test, y_test) = build_split()
    model = train_model(X_train, y_train, X_val, y_val)

    explainer = build_explainer(model)

    # Show drivers for the first 3 test applicants
    for idx in range(3):
        pd_score = model.predict_proba(X_test.iloc[[idx]])[:, 1][0]
        print(f"\nApplicant {idx} — predicted default probability: {pd_score:.1%}")
        drivers = top_drivers(explainer, X_test.iloc[[idx]])
        for d in drivers:
            arrow = "↑ risk" if d["direction"] == "increases_risk" else "↓ risk"
            print(f"  {arrow}  {d['description']} = {d['value']} (SHAP {d['shap']:+.3f})")
