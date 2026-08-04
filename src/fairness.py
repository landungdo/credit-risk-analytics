"""
Fairness audit for the PD model.

The Lending Club data contains no direct protected attributes (race, gender),
so this audit uses proxy groups that are observable in the data:
- income bracket (low / mid / high, by tertile)
- broad US region (derived from state of residence)

For each group we compute, at a fixed approval threshold:
- approval rate (share scored below the PD cutoff)
- the disparate impact ratio vs. the most-approved group — the "four-fifths
  rule" from US fair-lending guidance flags a ratio below 0.80
- observed default rate among approved loans (calibration parity: is the model
  equally accurate across groups, or does one group carry hidden extra risk?)

This is intended as a screening tool that surfaces disparities for review, not
a certification of fairness.
"""

import numpy as np
import pandas as pd

# Coarse census-style region mapping for US states
REGION_MAP = {
    "CT": "Northeast", "ME": "Northeast", "MA": "Northeast", "NH": "Northeast",
    "RI": "Northeast", "VT": "Northeast", "NJ": "Northeast", "NY": "Northeast",
    "PA": "Northeast",
    "IL": "Midwest", "IN": "Midwest", "MI": "Midwest", "OH": "Midwest",
    "WI": "Midwest", "IA": "Midwest", "KS": "Midwest", "MN": "Midwest",
    "MO": "Midwest", "NE": "Midwest", "ND": "Midwest", "SD": "Midwest",
    "DE": "South", "FL": "South", "GA": "South", "MD": "South", "NC": "South",
    "SC": "South", "VA": "South", "DC": "South", "WV": "South", "AL": "South",
    "KY": "South", "MS": "South", "TN": "South", "AR": "South", "LA": "South",
    "OK": "South", "TX": "South",
    "AZ": "West", "CO": "West", "ID": "West", "MT": "West", "NV": "West",
    "NM": "West", "UT": "West", "WY": "West", "AK": "West", "CA": "West",
    "HI": "West", "OR": "West", "WA": "West",
}


def assign_groups(df: pd.DataFrame) -> pd.DataFrame:
    """Add income_bracket and region columns used as fairness proxy groups."""
    out = df.copy()
    out["income_bracket"] = pd.qcut(
        out["annual_inc"], q=3, labels=["low", "mid", "high"]
    )
    out["region"] = out["addr_state"].map(REGION_MAP).fillna("Other")
    return out


def audit_group(df: pd.DataFrame, pd_scores, y_true, group_col: str, threshold: float):
    """
    Compute approval rate, disparate impact, and approved-default rate per group.

    `threshold`: applications with predicted PD below this are "approved".
    """
    work = df.copy()
    work["pd_score"] = pd_scores
    work["y_true"] = y_true
    work["approved"] = work["pd_score"] < threshold

    rows = []
    for group, g in work.groupby(group_col, observed=True):
        approval_rate = g["approved"].mean()
        approved = g[g["approved"]]
        approved_default_rate = (
            approved["y_true"].mean() if len(approved) > 0 else np.nan
        )
        rows.append({
            "group": group,
            "n": len(g),
            "approval_rate": approval_rate,
            "approved_default_rate": approved_default_rate,
        })

    result = pd.DataFrame(rows)
    # Disparate impact: each group's approval rate vs. the highest one
    max_rate = result["approval_rate"].max()
    result["disparate_impact"] = result["approval_rate"] / max_rate
    result["flag_4_5_rule"] = result["disparate_impact"] < 0.80
    return result.sort_values("approval_rate", ascending=False).reset_index(drop=True)


if __name__ == "__main__":
    try:
        from src.model import build_split, train_model
    except ModuleNotFoundError:
        from model import build_split, train_model

    (X_train, y_train), (X_val, y_val), (X_test, y_test) = build_split()
    model = train_model(X_train, y_train, X_val, y_val)

    # Rebuild the raw test frame to recover group columns (state, income)
    try:
        from src.oot_split import load_resolved_loans, out_of_time_split
    except ModuleNotFoundError:
        from oot_split import load_resolved_loans, out_of_time_split
    resolved = load_resolved_loans("data/sample.csv")
    _, _, test_df = out_of_time_split(resolved)
    test_df = assign_groups(test_df)

    pd_scores = model.predict_proba(X_test)[:, 1]

    # Approve the ~70% lowest-risk applications (illustrative cutoff)
    threshold = np.quantile(pd_scores, 0.70)
    print(f"Approval threshold (PD < {threshold:.3f}), approving ~70% overall\n")

    for col in ["income_bracket", "region"]:
        print(f"=== Fairness by {col} ===")
        audit = audit_group(test_df, pd_scores, y_test, col, threshold)
        with pd.option_context("display.float_format", "{:.3f}".format):
            print(audit.to_string(index=False))
        flagged = audit[audit["flag_4_5_rule"]]
        if len(flagged) > 0:
            print(f"⚠️  Groups flagged by four-fifths rule: {list(flagged['group'])}")
        print()
