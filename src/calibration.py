"""
Probability calibration for the PD model.

A credit risk PD is used directly in downstream expected-loss and capital
calculations, so the predicted number must behave like a true probability:
among loans scored at 0.20, roughly 20% should actually default. A model can
rank well (high AUC) yet be poorly calibrated. We fit an isotonic calibrator
on the validation vintage and check calibration on the out-of-time test set.

Calibration quality is summarized with the Brier score (lower is better) and
visualized with a reliability diagram (predicted vs. observed default rate).
"""

import matplotlib
matplotlib.use("Agg")  # headless backend; save figures without a display
import matplotlib.pyplot as plt
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.frozen import FrozenEstimator
from sklearn.metrics import brier_score_loss

try:
    from src.model import build_split, train_model
except ModuleNotFoundError:
    from model import build_split, train_model


def calibrate(model, X_val, y_val):
    """
    Fit an isotonic calibrator on the validation set without retraining the
    base model. FrozenEstimator freezes the already-trained model so
    CalibratedClassifierCV only fits the calibration mapping on top of it
    (replaces the deprecated cv='prefit' from sklearn < 1.6).
    """
    calibrated = CalibratedClassifierCV(FrozenEstimator(model), method="isotonic")
    calibrated.fit(X_val, y_val)
    return calibrated


def reliability_diagram(y_true, p_uncalibrated, p_calibrated, out_path):
    """Plot predicted vs. observed default rate, before and after calibration."""
    frac_pos_uncal, mean_pred_uncal = calibration_curve(y_true, p_uncalibrated, n_bins=10)
    frac_pos_cal, mean_pred_cal = calibration_curve(y_true, p_calibrated, n_bins=10)

    plt.figure(figsize=(6, 6))
    plt.plot([0, 1], [0, 1], "k--", label="Perfectly calibrated")
    plt.plot(mean_pred_uncal, frac_pos_uncal, "o-", label="Uncalibrated")
    plt.plot(mean_pred_cal, frac_pos_cal, "s-", label="Calibrated (isotonic)")
    plt.xlabel("Mean predicted default probability")
    plt.ylabel("Observed default rate")
    plt.title("Reliability diagram — out-of-time test (2016)")
    plt.legend(loc="upper left")
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close()
    print(f"Saved reliability diagram to {out_path}")


if __name__ == "__main__":
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = build_split()

    model = train_model(X_train, y_train, X_val, y_val)
    calibrated = calibrate(model, X_val, y_val)

    p_uncal = model.predict_proba(X_test)[:, 1]
    p_cal = calibrated.predict_proba(X_test)[:, 1]

    brier_uncal = brier_score_loss(y_test, p_uncal)
    brier_cal = brier_score_loss(y_test, p_cal)

    print(f"Brier score (uncalibrated): {brier_uncal:.4f}")
    print(f"Brier score (calibrated):   {brier_cal:.4f}")
    print(f"Test default base rate:     {y_test.mean():.4f}")
    print(f"Mean predicted (uncal):     {p_uncal.mean():.4f}")
    print(f"Mean predicted (cal):       {p_cal.mean():.4f}")

    reliability_diagram(y_test, p_uncal, p_cal, "reliability_diagram.png")
