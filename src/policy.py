"""
Decision policy simulator.

A PD model on its own only scores risk. A lending business needs a *decision*:
given each applicant's calibrated PD, which loans to approve, and where to set
the cutoff. This module turns scores into decisions and quantifies the trade-off
a credit policy actually faces:

  - approve too many  -> higher volume, but more defaults and losses
  - approve too few   -> fewer losses, but forgone interest income

For each candidate PD threshold it computes, on the approved book:
  - approval rate
  - realized default rate among approved loans
  - expected profit = interest income on good loans - loss on defaulted loans

Profit model (simplified but structurally correct):
  - a repaid loan earns approximately  int_rate * loan_amnt  (one period of interest)
  - a defaulted loan loses             LGD * loan_amnt        (principal not recovered)

These are illustrative unit economics, not a bank's full P&L, but they capture
the core tension: the optimal cutoff is the one that maximizes profit, which is
generally *not* the one that minimizes defaults.
"""

import numpy as np
import pandas as pd

DEFAULT_LGD = 0.45


def simulate_policy(pd_scores, y_true, loan_amnt, int_rate,
                    thresholds=None, lgd: float = DEFAULT_LGD) -> pd.DataFrame:
    """
    Evaluate approve/decline outcomes across a grid of PD cutoffs.

    A loan is approved if its predicted PD is below the threshold. Profit uses
    the *realized* outcome (y_true) so the reported profit is what the policy
    would actually have earned on this book.
    """
    pd_scores = np.asarray(pd_scores)
    y_true = np.asarray(y_true)
    loan_amnt = np.asarray(loan_amnt, dtype=float)
    int_rate = np.asarray(int_rate, dtype=float) / 100.0  # percent -> fraction

    if thresholds is None:
        thresholds = np.round(np.arange(0.05, 0.51, 0.05), 2)

    rows = []
    n = len(pd_scores)
    for t in thresholds:
        approved = pd_scores < t
        n_approved = int(approved.sum())
        if n_approved == 0:
            continue

        appr_default = y_true[approved]
        appr_amount = loan_amnt[approved]
        appr_rate_int = int_rate[approved]

        # Repaid loans earn one period of interest; defaults lose LGD * principal
        income = np.where(appr_default == 0, appr_rate_int * appr_amount, 0.0)
        loss = np.where(appr_default == 1, lgd * appr_amount, 0.0)
        profit = float(np.sum(income) - np.sum(loss))

        rows.append({
            "threshold": t,
            "approval_rate": n_approved / n,
            "approved_default_rate": float(appr_default.mean()),
            "total_profit": profit,
            "profit_per_approved": profit / n_approved,
        })

    return pd.DataFrame(rows)


def optimal_threshold(policy_df: pd.DataFrame) -> dict:
    """Return the threshold row that maximizes total profit."""
    best = policy_df.loc[policy_df["total_profit"].idxmax()]
    return best.to_dict()


def assign_decision(pd_score: float, approve_below: float, review_below: float) -> str:
    """
    Three-way decision band for a single applicant:
      - APPROVE       if PD < approve_below
      - MANUAL_REVIEW if approve_below <= PD < review_below
      - DECLINE       otherwise
    """
    if pd_score < approve_below:
        return "APPROVE"
    if pd_score < review_below:
        return "MANUAL_REVIEW"
    return "DECLINE"


def select_cutoff_on_validation(calibrated, X_val, val_df, lgd: float = DEFAULT_LGD):
    """
    Choose the profit-maximizing cutoff on the VALIDATION book (2015), so the
    threshold is selected on data separate from the final test. Returns the
    frozen cutoff and the validation policy table.
    """
    pd_val = calibrated.predict_proba(X_val)[:, 1]
    policy_val = simulate_policy(
        pd_val, val_df["target"].values,
        val_df["loan_amnt"].values, val_df["int_rate"].values, lgd=lgd,
    )
    best = optimal_threshold(policy_val)
    return best["threshold"], policy_val


if __name__ == "__main__":
    try:
        from src.model import build_split, train_model
        from src.calibration import calibrate
        from src.oot_split import load_resolved_loans, out_of_time_split
    except ModuleNotFoundError:
        from model import build_split, train_model
        from calibration import calibrate
        from oot_split import load_resolved_loans, out_of_time_split

    (X_train, y_train), (X_val, y_val), (X_test, y_test) = build_split()
    model = train_model(X_train, y_train, X_val, y_val)
    calibrated = calibrate(model, X_val, y_val)

    # Recover raw validation and test frames (for loan_amnt / int_rate / target)
    resolved = load_resolved_loans("data/sample.csv")
    _, val_df, test_df = out_of_time_split(resolved)

    # 1) SELECT the cutoff on the 2015 validation book (not the test book)
    frozen_cutoff, policy_val = select_cutoff_on_validation(calibrated, X_val, val_df)

    # 2) EVALUATE that frozen cutoff on the untouched 2016 test book
    pd_test = calibrated.predict_proba(X_test)[:, 1]
    policy_test = simulate_policy(
        pd_test, y_test, test_df["loan_amnt"].values, test_df["int_rate"].values,
        thresholds=[frozen_cutoff],
    )
    frozen_row = policy_test.iloc[0]

    print("=== Cutoff selected on 2015 validation, evaluated on 2016 test ===\n")
    print(f"Selected cutoff on policy-validation (2015): PD < {frozen_cutoff:.2f}")
    print(f"Frozen cutoff evaluated on OOT test (2016):  PD < {frozen_cutoff:.2f}\n")
    print("On the untouched 2016 test book at the frozen cutoff:")
    print(f"  approval rate:          {frozen_row['approval_rate']:.1%}")
    print(f"  approved default rate:  {frozen_row['approved_default_rate']:.1%}")
    print(f"  total profit:           ${frozen_row['total_profit']:,.0f}")
    print()
    print("The cutoff is chosen on validation and only then measured on test, so")
    print("the reported test profit is free of the optimism bias that arises when")
    print("the same book is used to both select and evaluate the policy.")
    print()
    print("Full validation trade-off table (where the cutoff was chosen):")
    with pd.option_context("display.float_format", lambda x: f"{x:,.3f}"):
        print(policy_val.to_string(index=False))
