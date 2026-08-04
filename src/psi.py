"""
Population Stability Index (PSI) monitoring.

PSI measures how much the distribution of model scores has shifted between a
baseline period (training) and a later period (out-of-time test). It is the
standard early-warning metric for model drift in credit risk:

    PSI = sum over bins of (actual% - expected%) * ln(actual% / expected%)

Common rule of thumb:
    PSI < 0.10  -> stable
    0.10-0.25   -> moderate shift, monitor
    PSI > 0.25  -> significant shift, investigate/recalibrate

The key finding this module is designed to surface: an overall PSI can look
stable while a specific subgroup has already drifted materially. Aggregate
monitoring hides that; segmented PSI reveals it.
"""

import numpy as np
import pandas as pd


def population_stability_index(expected, actual, n_bins: int = 10) -> float:
    """
    PSI between a baseline (expected) and a later (actual) score distribution.

    Bin edges are fixed from the expected distribution's quantiles so both
    distributions are compared on the same buckets.
    """
    # Quantile bin edges from the baseline; widen the outer edges to catch tails
    edges = np.quantile(expected, np.linspace(0, 1, n_bins + 1))
    edges[0], edges[-1] = -np.inf, np.inf

    expected_counts, _ = np.histogram(expected, bins=edges)
    actual_counts, _ = np.histogram(actual, bins=edges)

    expected_pct = expected_counts / len(expected)
    actual_pct = actual_counts / len(actual)

    # Small epsilon to avoid division by zero / log(0) in empty bins
    eps = 1e-6
    expected_pct = np.clip(expected_pct, eps, None)
    actual_pct = np.clip(actual_pct, eps, None)

    return float(np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct)))


def psi_by_subgroup(baseline_df, later_df, score_col: str, group_col: str,
                    n_bins: int = 10) -> pd.DataFrame:
    """
    Compute PSI within each subgroup, plus the overall PSI for reference.

    Bin edges are fixed from the *overall baseline* so subgroup PSIs are
    comparable to each other and to the overall figure.
    """
    overall = population_stability_index(
        baseline_df[score_col].values, later_df[score_col].values, n_bins
    )

    rows = [{"group": "OVERALL", "n_later": len(later_df), "psi": overall}]
    for group in sorted(set(baseline_df[group_col]) & set(later_df[group_col])):
        base_g = baseline_df.loc[baseline_df[group_col] == group, score_col].values
        late_g = later_df.loc[later_df[group_col] == group, score_col].values
        if len(base_g) < 50 or len(late_g) < 50:
            continue  # too few observations for a reliable PSI
        psi = population_stability_index(base_g, late_g, n_bins)
        rows.append({"group": group, "n_later": len(late_g), "psi": psi})

    result = pd.DataFrame(rows)
    result["status"] = pd.cut(
        result["psi"],
        bins=[-np.inf, 0.10, 0.25, np.inf],
        labels=["stable", "moderate", "significant"],
    )
    return result


if __name__ == "__main__":
    try:
        from src.model import build_split, train_model
        from src.oot_split import load_resolved_loans, out_of_time_split
        from src.fairness import assign_groups
    except ModuleNotFoundError:
        from model import build_split, train_model
        from oot_split import load_resolved_loans, out_of_time_split
        from fairness import assign_groups

    (X_train, y_train), (X_val, y_val), (X_test, y_test) = build_split()
    model = train_model(X_train, y_train, X_val, y_val)

    # Rebuild raw frames to recover group columns and attach scores
    resolved = load_resolved_loans("data/sample.csv")
    train_df, _, test_df = out_of_time_split(resolved)
    train_df = assign_groups(train_df)
    test_df = assign_groups(test_df)

    train_df["score"] = model.predict_proba(X_train)[:, 1]
    test_df["score"] = model.predict_proba(X_test)[:, 1]

    print("=== PSI: training baseline vs out-of-time test (2016) ===\n")

    for col in ["income_bracket", "region"]:
        print(f"--- Segmented by {col} ---")
        result = psi_by_subgroup(train_df, test_df, "score", col)
        with pd.option_context("display.float_format", "{:.4f}".format):
            print(result.to_string(index=False))
        print()
