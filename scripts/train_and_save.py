"""
Train the model once and persist all artifacts the API needs at serve time.

Saves the calibrated model and the fitted feature metadata to disk so the API
loads a ready model at startup instead of retraining on every boot. Run this
as a build/deploy step:  python scripts/train_and_save.py
"""

import pickle
from pathlib import Path

try:
    from src.model import build_split, train_model
    from src.calibration import calibrate
except ModuleNotFoundError:
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from src.model import build_split, train_model
    from src.calibration import calibrate

ARTIFACT_DIR = Path("artifacts")


def main(sample_path: str = "data/sample.csv"):
    ARTIFACT_DIR.mkdir(exist_ok=True)

    (X_train, y_train), (X_val, y_val), (X_test, y_test) = build_split(sample_path)
    model = train_model(X_train, y_train, X_val, y_val)
    calibrated = calibrate(model, X_val, y_val)

    # The raw XGBoost model is needed for SHAP; the calibrated wrapper is
    # needed for well-calibrated PDs. Persist both, plus the feature schema.
    with open(ARTIFACT_DIR / "model.pkl", "wb") as f:
        pickle.dump(model, f)
    with open(ARTIFACT_DIR / "calibrated.pkl", "wb") as f:
        pickle.dump(calibrated, f)
    with open(ARTIFACT_DIR / "feature_columns.pkl", "wb") as f:
        pickle.dump(list(X_train.columns), f)

    # Select the decision cutoff on the 2015 validation book (not the test book)
    # and persist it as a versioned policy artifact the API loads at startup.
    import json
    from src.policy import select_cutoff_on_validation
    from src.oot_split import load_resolved_loans, out_of_time_split

    resolved = load_resolved_loans(sample_path)
    _, val_df, _ = out_of_time_split(resolved)
    approve_below, _ = select_cutoff_on_validation(calibrated, X_val, val_df)
    review_below = min(approve_below + 0.05, 1.0)  # manual-review band above approve

    policy = {
        "policy_version": "policy-v2.1",
        "model_version": "xgb-champion-v2.1",
        "approve_below": round(float(approve_below), 4),
        "review_below": round(float(review_below), 4),
        "selected_on": "2015 validation vintage",
        "note": "Cutoff chosen on validation, not on the test book.",
    }
    with open(ARTIFACT_DIR / "policy.json", "w") as f:
        json.dump(policy, f, indent=2)

    print(f"Saved artifacts to {ARTIFACT_DIR}/")
    print(f"  model.pkl, calibrated.pkl, feature_columns.pkl, policy.json")
    print(f"Feature count: {len(X_train.columns)}")
    print(f"Policy: approve_below={policy['approve_below']}, review_below={policy['review_below']}")


if __name__ == "__main__":
    main()
