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
    pd_scores = calibrated.predict_proba(X_test)[:, 1]

    resolved = load_resolved_loans("data/sample.csv")
    _, _, test_df = out_of_time_split(resolved)
    loan_amnt = test_df["loan_amnt"].values
    int_rate = test_df["int_rate"].values

    policy = simulate_policy(pd_scores, y_test, loan_amnt, int_rate)

    print("=== Decision policy trade-off (out-of-time test, 2016) ===\n")
    with pd.option_context("display.float_format", lambda x: f"{x:,.3f}"):
        print(policy.to_string(index=False))

    best = optimal_threshold(policy)
    print(f"\nProfit-maximizing cutoff: PD < {best['threshold']:.2f}")
    print(f"  approval rate:          {best['approval_rate']:.1%}")
    print(f"  approved default rate:  {best['approved_default_rate']:.1%}")
    print(f"  total profit:           ${best['total_profit']:,.0f}")
    print()
    print("Note: the profit-maximizing cutoff is not the one that minimizes")
    print("defaults - some default risk is worth taking for the interest income.")
