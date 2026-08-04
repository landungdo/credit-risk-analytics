"""
Portfolio-level credit risk aggregation.

Individual calibrated PDs are combined with standard credit-risk parameters to
produce portfolio metrics that a risk function actually reports:

- Expected Loss (EL) = PD x LGD x EAD, summed across the portfolio.
  * PD  = calibrated probability of default (from the model)
  * LGD = loss given default (fraction of exposure lost when a loan defaults)
  * EAD = exposure at default (approximated here by the loan amount)

- Unexpected Loss / capital: a simplified Basel-style capital estimate using
  the Basel IRB asset-correlation formula for retail-style exposures. This is
  a teaching approximation, not a regulatory calculation, but it uses the
  actual IRB functional form rather than an ad-hoc number.

Calibration matters here: EL scales linearly with PD, so a miscalibrated PD
feeds a biased loss number straight into the capital estimate.
"""

import numpy as np
from scipy.stats import norm

# Standard assumptions (illustrative, documented so they can be changed)
DEFAULT_LGD = 0.45          # 45% loss given default, a common unsecured assumption
CONFIDENCE_LEVEL = 0.999    # Basel IRB uses the 99.9% worst-case
ASSET_CORRELATION = 0.15    # fixed correlation for this simplified retail model


def expected_loss(pd_scores, exposure, lgd: float = DEFAULT_LGD):
    """Per-loan expected loss = PD x LGD x EAD."""
    return pd_scores * lgd * exposure


def basel_capital_requirement(pd_scores, exposure, lgd: float = DEFAULT_LGD,
                              correlation: float = ASSET_CORRELATION,
                              confidence: float = CONFIDENCE_LEVEL):
    """
    Simplified Basel IRB capital requirement per loan.

    K = LGD * [ N( (N^-1(PD) + sqrt(R) * N^-1(conf)) / sqrt(1-R) ) - PD ] * EAD

    where N is the standard normal CDF and N^-1 its inverse. This is the
    unexpected-loss capital charge (loss beyond the expected loss).
    """
    pd_clipped = np.clip(pd_scores, 1e-6, 1 - 1e-6)  # avoid inf at the tails
    inv_pd = norm.ppf(pd_clipped)
    inv_conf = norm.ppf(confidence)

    conditional_pd = norm.cdf(
        (inv_pd + np.sqrt(correlation) * inv_conf) / np.sqrt(1 - correlation)
    )
    capital_rate = lgd * (conditional_pd - pd_clipped)
    return capital_rate * exposure


def portfolio_summary(pd_scores, exposure, lgd: float = DEFAULT_LGD) -> dict:
    """Aggregate EL and capital across the whole portfolio."""
    el = expected_loss(pd_scores, exposure, lgd)
    capital = basel_capital_requirement(pd_scores, exposure, lgd)

    total_exposure = float(np.sum(exposure))
    total_el = float(np.sum(el))
    total_capital = float(np.sum(capital))

    return {
        "n_loans": len(pd_scores),
        "total_exposure": total_exposure,
        "expected_loss": total_el,
        "expected_loss_rate": total_el / total_exposure,
        "capital_requirement": total_capital,
        "capital_rate": total_capital / total_exposure,
        "avg_pd": float(np.mean(pd_scores)),
    }


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

    # Use CALIBRATED PDs — the whole point is that these feed loss numbers
    pd_scores = calibrated.predict_proba(X_test)[:, 1]

    # Exposure at default approximated by loan amount
    resolved = load_resolved_loans("data/sample.csv")
    _, _, test_df = out_of_time_split(resolved)
    exposure = test_df["loan_amnt"].values

    summary = portfolio_summary(pd_scores, exposure)

    print("=== Portfolio Risk Summary (out-of-time test, 2016) ===")
    print(f"Loans:                {summary['n_loans']:,}")
    print(f"Total exposure:       ${summary['total_exposure']:,.0f}")
    print(f"Average PD:           {summary['avg_pd']:.2%}")
    print(f"Expected loss:        ${summary['expected_loss']:,.0f} "
          f"({summary['expected_loss_rate']:.2%} of exposure)")
    print(f"Capital requirement:  ${summary['capital_requirement']:,.0f} "
          f"({summary['capital_rate']:.2%} of exposure)")
