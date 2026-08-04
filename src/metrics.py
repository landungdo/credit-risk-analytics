"""
Discrimination metrics for the credit risk model.

Two metrics, both standard in credit scoring:
- AUC (ROC): overall ranking quality.
- KS statistic: the maximum separation between the cumulative distributions
  of defaults and non-defaults. In credit risk a KS in the 20-40 range is
  typical for application scorecards; higher is better discrimination.
"""

import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve


def auc_score(y_true, y_score) -> float:
    """Area under the ROC curve."""
    return roc_auc_score(y_true, y_score)


def ks_statistic(y_true, y_score) -> float:
    """
    Kolmogorov-Smirnov statistic = max(TPR - FPR) across all thresholds.

    Equivalent to the maximum gap between the cumulative distributions of
    the positive (default) and negative (non-default) classes.
    """
    fpr, tpr, _ = roc_curve(y_true, y_score)
    return float(np.max(tpr - fpr))


def evaluate(y_true, y_score, label: str = "") -> dict:
    """Compute and print AUC + KS for one split."""
    auc = auc_score(y_true, y_score)
    ks = ks_statistic(y_true, y_score)
    prefix = f"[{label}] " if label else ""
    print(f"{prefix}AUC: {auc:.4f} | KS: {ks:.4f}")
    return {"auc": auc, "ks": ks}
